import logging
from typing import List, Optional, Tuple

import regex as re

from blocks.location_block import get_loc
from blocks.other_blocks import Plan, PlanQueue
from blocks.time_block import MyDateTime
from processors.chatglm import llms_dict
from agents.brain import Brain
from blocks.retriever import Observation, Retriever


class Agent:
    name: str  # 姓名
    age: int  # 年龄
    traits: str  # 性格
    status: str  # 现在的状态
    profession: List = []  # 职业
    brain: Brain  # 记忆
    summary_description: str = ""  # Agent’s Summary Description
    summary_refresh_seconds: int = 3600  # 多长时间进行一次总结

    last_refreshed: MyDateTime  # 上次生成摘要的时间
    daily_summaries: List[str] = []  # 执行的计划摘要

    dialogue_history: List = []

    plans: PlanQueue = PlanQueue()  # 当天的计划
    schedule: List = []  # 当天的日程
    schedule_summary = ""  # 当天的日程摘要

    loc: str  # 所在位置
    time_step: int = 600  # 智能体感知周围环境的时间步，单位秒
    processer = llms_dict["chat_gpt"]

    def __init__(self, name, age, traits, status, profession, clock, loc, memories, faiss_store):
        super(Agent, self).__init__()
        self.name = name
        self.age = age
        self.traits = traits
        self.status = status
        self.profession = profession
        self.clock = clock
        now = self.clock.now()
        self.last_refreshed = now
        self.loc = loc
        self.brain = Brain(memories=Retriever(vector_store=faiss_store))
        self.brain.memorizing_batch(memories, now, place=loc)

    def _get_entity_from_observation(self, desc: str) -> str:
        """获取观察主体"""
        prompt = f"以下观察中观察的对象是什么？\n" \
                 f"对象是："
        object = self.processer.processing(prompt)
        object = "解析subject"
        return object

    def _get_entity_action(self, desc: str, entity_name: str) -> str:
        """获取行为"""
        prompt = f"观察：{desc}" \
                 f"根据上述观察推断{entity_name}在做什么\n" \
                 f"{entity_name}在做"
        action = self.processer.processing(prompt)
        action = "解析action"
        return action

    def _brief_summary(self, prompt):
        """对记忆进行简要总结"""
        summary = self.processer.processing(prompt)
        summaries = []
        try:
            summaries = summary.strip().split()
            summaries = [i.strip() for i in summaries]
        except:
            logging.error(f"解析总结: {summary}过程出错...")

        return summaries

    def _merge_memories(self, memories):
        visited = set()
        merged_memories = []
        for memo in memories:
            if memo.id in visited:
                continue
            memo.id.add(visited)
            merged_memories.append(memo)
        return merged_memories

    def summarize_for_reacting(self, agent_status, agent_name, now) -> str:
        """反应前的准备：总结相关记忆"""
        # 根据两个模版「[观察者] 和 [被观察实体] 的关系是什么」和「[被观察实体] 正在 [被观察实体的动作状态]」构造 query
        q1 = f"{self.name}和{agent_name}是什么关系？"
        q2 = f"{agent_name}正在{agent_status}"
        # 检索记忆库，获取相关记忆
        memories1 = self.brain.remembering(q1, now)
        memories2 = self.brain.remembering(q2, now)
        # 合并记忆
        memories = self._merge_memories([memories1, memories2])
        memories_str = ";".join([str(memo) for memo in memories])
        # 生成总结
        prompt = f"{self.name}关于{agent_name}的记忆如下:\n" \
                 f"{memories_str}\n\n" \
                 f"对上述记忆生成3-5条简要的总结:"
        summary = self._brief_summary(prompt)
        return ";".join(summary)

    def _reacting(self, observation: str, name, suffix: str, now: Optional[MyDateTime] = None) -> str:
        """对观察进行反应"""
        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_for_reacting(observation, name, now)
        prompt = f"{agent_summary_description}\n" \
                 f"现在是{str(now)}\n" \
                 f"{self.name}现在的状态: {self.status}\n" \
                 f"观察: {observation}\n" \
                 f"{self.name}的记忆中相关的内容:{relevant_memories_str}\n\n" \
                 f"{suffix}"

        # 判断长度
        reaction = self.processer.processing(prompt)
        return reaction

    def generate_reaction(self, observation: str, name, now):
        """生成反应"""

        def _pars_action(reaction):
            flag, mod, action = False, None, None
            try:
                reactions = reaction.strip().split("^")
                flag, mod, action = reactions
                flag = True if flag == "是" else False
            except:
                logging.error(f"解析反应：{reaction}时出错。。。")
            return flag, mod, action

        suffix = f"{self.name}应该对观察做出反应吗？如果是，应该做出什么反应？在一行中进行回复" \
                 f"如果不需要做出反应，输出格式为: 否^None^None" \
                 f"如果做出的反应是对话，输出格式为: 是^SAY^要说的话\n" \
                 f"否则输出格式为: 是^REACT^做出的反应(如果有的话)\n" \
                 f"要么什么都不做，做出反应，要么说些什么，但不能两者兼而有之。\n"
        reaction = self._reacting(observation, name, suffix, now)
        flag, mod, action = _pars_action(reaction)
        return flag, mod, action

    def generate_dialogue_response(self,
                                   observation: str,
                                   object_name,
                                   dialogue_history) -> Tuple[bool, str]:
        """生成聊天内容"""

        def _pars_reply(reply):
            flag, text = False, "对话未能成功解析，结束对话"
            try:
                flag, text = reply.strip().split("^")
                flag = True if flag == "SAY" else False
            except:
                logging.error(f"解析回复：{reply}时出错。。。")

            return flag, text

        now = self.clock.now()
        # suffix_first = f"{reaction}。 {self.name}可能会说什么？\n" \
        #                f"输出格式为: SAY:要说的话\n"

        dialogue_history = "\n".join(dialogue_history)
        suffix_next = f"对话历史：\n{dialogue_history}\n" \
                      f"{self.name}可能如何回答{object_name}?\n" \
                      f"如果要结束对话，输出格式为：GOODBYE^要说的话；" \
                      f"否则，输出格式为: SAY^要说的话\n"

        reply = self._reacting(observation, object_name, suffix_next, now)

        flag, reply = _pars_reply(reply)

        return flag, reply

    def _generate_summary_description(self, now: MyDateTime) -> str:
        """生成总结"""
        summary_contains: [str] = [(f"{self.name}的主要个性", f"给定以下陈述，人们会怎么评价{self.name}的主要个性?\n"),
                                   (f"{self.name}当前的工作日常", f"给定以下陈述，{self.name}将如何总结自己的工作日常?\n"),
                                   (f"{self.name}对最近生活的感受", f"给定以下陈述，{self.name}感到最近的生活:")]

        summary_str = ""
        for query, prompt in summary_contains:
            related_memories = self.brain.remembering(query, now)
            related_memories_str = "\n".join([str(mem) for mem in related_memories])
            prompt += related_memories_str
            summary = self.processer.processing(prompt)
            summary = "解析总结"
            summary_str += summary
        return summary_str

    def get_summary(self, now: MyDateTime, force_refresh: bool = False) -> str:
        """返回代理的摘要性描述"""
        since_refresh = now - self.last_refreshed
        if (
                not self.summary_description
                or since_refresh >= self.summary_refresh_seconds
                or force_refresh
        ):
            self.summary_description = self._generate_summary_description(now)
            self.last_refreshed = now
        age = self.age if self.age is not None else "N/A"
        summary_desc = f"姓名:{self.name}(年龄:{age})\n" \
                       f"性格:{self.traits}\n" \
                       f"职业:{','.join(self.profession)}\n" \
                       f"{self.summary_description}"

        return summary_desc

    def get_schedule_summary(self, update=True):
        """获取前一天的日程摘要"""

        if self.schedule and (not self.schedule_summary or update):
            schedule_str = "\n".join(self.schedule)
            prompt = f"{self.name}昨天的日程:\n" \
                     f"{schedule_str}\n\n" \
                     f"对李开心昨天的日程生成3-5条简要的总结:(输出格式：时间范围：主要事件)"
            summaries = self._brief_summary(prompt)
            summary = ";".join(summaries)
            yesterday = self.clock.past(1)
            month = yesterday.get("month", "N/A")
            day = yesterday.get("day", "N/A")
            weekday = yesterday.get("weekday", "N/A")
            self.schedule_summary = f"昨天是{month}月{day}日，{weekday}。{self.name}做的主要事情有:" + summary
            self.schedule = []
        return self.schedule_summary

    def parse_plan(self, plan_str):
        p = re.compile(r"(?P<start>\d{1,2}:\d{1,2})[-—](?P<end>\d{1,2}:\d{1,2})\]:\s+(?P<task>.+)")
        plans = []
        for plan in plan_str.split("\n"):
            match = p.search(plan)
            if match:
                info = match.groupdict()
                if all([isinstance(i, str) and len(i) > 1 for i in info.values()]):
                    plans.append(Plan(**info))
            else:
                logging.warning(f"模型生成的计划无法解析：{plan}")
        return plans

    def planning(self, now: MyDateTime):
        """生成计划"""
        # 1. 获取摘要性描述和前一天的日程摘要
        summary_description = self.get_summary(now=now)
        schedule_summary = self.get_schedule_summary()
        month = now.get("month", "N/A")
        day = now.get("day", "N/A")
        weekday = now.get("weekday", "N/A")
        hour = now.get("hour", "N/A")
        minute = now.get("minute", "N/A")
        # 2. 生成粗略规划
        prompt = f"{summary_description}\n" \
                 f"{schedule_summary}\n" \
                 f"大致计划示例:" \
                 f"这是张三今天从7:14开始的大致计划:\n" \
                 f"[7:14-7:45]: 起床并完成早间程序\n" \
                 f"[7:45-8:35]: 先吃早餐\n" \
                 f"[8:35-17:10]: 去学校学习\n" \
                 f"[17:10-22:30]: 玩CSGO\n" \
                 f"[22:30-7:30]: 睡觉\n\n" \
                 f"今天是{month}月{day}日，{weekday}。这是{self.name}今天从{hour}:{minute}开始的一天的简洁的大致计划:"
        rough_plans = self.processer.processing(prompt)
        rough_plans = self.parse_plan(rough_plans)
        # 3. 将粗略规划保存到计划中
        self.plans.batch_put(rough_plans)

    def disassemble_plan(self, rough_plan) -> List[Plan]:
        prompt = f"粗略计划:{rough_plan}\n" \
                 f"根据粗略计划的时间范围和活动目标，生成更细致的活动目标，每个活动限制在5-15分钟以内;\n" \
                 f"细致计划示例:\n" \
                 f"[7:14-7:20]: 起床并完成早间程序\n" \
                 f"[7:20-7:30]: 吃早餐\n\n" \
                 f"细致计划:"
        detailed_plans = self.processer.processing(prompt)
        detailed_plans = self.parse_plan(detailed_plans)
        return detailed_plans

    def _replan(self, observation, reaction):
        """重新生成计划"""
        now = self.clock.now()
        hour = now.get("hour", "N/A")
        minute = now.get("minute", "N/A")
        old_plans = str(self.plans)
        prompt = f"现在是{hour}:{minute}\n" \
                 f"{self.name}的状态:{self.status}\n" \
                 f"观察:{observation}\n" \
                 f"反应:{reaction}\n" \
                 f"{self.name}原来的计划如下:{old_plans}\n\n" \
                 f"根据{self.name}的观察、反应以及原来的计划，从现在开始为他制定一天的简洁的大致的计划,不要写任何解释:"
        new_plans = self.processer.processing(prompt)
        new_plans = self.parse_plan(new_plans)
        self.plans = PlanQueue()
        self.plans.batch_put(new_plans)

    def perceiving(self, now: MyDateTime):
        """感知周围环境"""
        loc = get_loc(self.loc)
        # 感知周围环境
        obj_list = [obj for obj in loc.guest if obj.status]
        obv_list = [f"{obj.name}: {obj.desc}" for obj in obj_list]
        # 将感知结果保存到记忆流中
        if obv_list:
            self.brain.memorizing_batch(obv_list, now)
        return obj_list

    def run_conversation(self, action_self, agent):
        """
        聊天
        """

        now = self.clock.now()
        flag, mod, action_agent = agent.generate_reaction(self, now)
        if not flag and mod != "SAY":
            logging.info(f"{agent.name}没有回应{self.name}发起得对话")
            return
        self.status = f"正在与{agent.name}进行对话"
        agent.status = f"正在与{self.name}进行对话"
        dialogue_history = []
        sentence = action_self
        turns = 0
        while True:
            logging.info(sentence)
            flag1, sentence = agent.generate_dialogue_response(sentence, self.name, dialogue_history[-5:])
            dialogue_history.append(sentence)
            logging.info(sentence)
            flag2, sentence = self.generate_dialogue_response(sentence, agent.name, dialogue_history[-5:])
            logging.info(sentence)
            dialogue_history.append(sentence)
            if not flag1 or not flag2:
                break

            turns += 1
        logging.info(f"{self.name}和{agent.name}的聊天结束")
        self.brain.memorizing_batch(dialogue_history, now)
        agent.memorizing_batch(dialogue_history, now)

    def reacting(self, mod, action, agent):
        """行动"""
        # 1. 解析行动
        # 2. 执行行动
        if mod == "SAY":
            logging.info(f"{self.name}做出的反应是对话")
            self.run_conversation(action, agent)
        else:
            logging.info(f"{self.name}做出的反应是其他行为: {action}")
            self.status = action

    def start(self, time_step: int = 600):
        now: MyDateTime = self.clock.now()
        logging.info(f"{self.name}在{str(now)}的日志".center(66, "="))
        # 1. 制定计划
        self.planning(now)
        logging.info(f"制定一天的粗略计划如下: \n{str(self.plans)}")
        # 2. 执行计划，并在每个时间步感知周围环境
        #    2.1 将自己的行动及感知到的信息保存到记忆中
        # TODO 需要保证不会陷入死循环
        while not self.plans.empty():
            # 获取计划
            plan = self.plans.get()
            # 拆解计划
            detailed_plans = self.disassemble_plan(str(plan))
            remain_time = 0
            for dp in detailed_plans:
                s, e, action = dp.get_info()
                # TODO 状态怎么更新
                self.status = action
                time_cost = (e - s).seconds
                nums, remain_time = divmod(remain_time + time_cost, time_step)
                # 感知周围环境
                flag_break = False

                now = self.clock.now()
                obv_agent_list = self.perceiving(now)
                agent = obv_agent_list[0]
                # observations = "\n".join([f"{obj.name}，{obj.desc}" for obj in agent_list])
                if obv_agent_list:
                    # 判断采取反应还是继续执行计划
                    flag, mod, action = self.generate_reaction(agent.status, agent.name, now)
                    logging.info(f"{str(now)}，{self.name}观察到: \n{agent.status}\n"
                                 f"{self.name}决定对该反应{'做出' if flag_break else '不做出'}反应: {action}")
                    # 如果需要做出反应，则更新计划
                    if flag:
                        self._replan(agent.status, action)
                        self.reacting(mod, action, agent)
                        flag_break = True

                # 把行动添加到记忆中
                self.brain.memorizing(self.status, now)
                self.schedule.append(f"{self.status}")
                # 如果需要更新计划，则退出循环
                if flag_break:
                    break

    def user_control(self, user, question):
        """用户控制"""
        pass
