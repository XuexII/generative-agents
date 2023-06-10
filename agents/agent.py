from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict

from blocks.location_block import MyLocation
from blocks.time_block import MyDateTime
from processors.chatglm import chat_glm
from .brain import Brain


class Agent(BaseModel):
    name: str  # 姓名
    age: int  # 年龄
    traits: str  # 性格
    status: str  # 现在的状态
    profession: List = []  # 职业
    brain: Brain  # 记忆
    summary_description: str = ""  # Agent’s Summary Description
    summary_refresh_seconds: int = 3600  # 多长时间进行一次总结

    last_refreshed: MyDateTime = Field(default_factory=datetime.now)  # 上次生成摘要的时间
    daily_summaries: List[str] = Field(default_factory=list)  # 执行的计划摘要

    dialogue_history: List = []

    plans: List[Dict] = []  # 当天的计划
    schedule: List = []  # 当天的日程
    schedule_summary = ""  # 当天的日程摘要

    per_plan_consumed_times: str = "30分钟"  # 每个计划最少用多长时间
    loc: MyLocation

    def _get_entity_from_observation(self, desc: str) -> str:
        """获取观察主体"""
        prompt = f"以下观察中观察的对象是什么？\n" \
                 f"对象是："
        object = chat_glm.processing(prompt)
        object = "解析subject"
        return object

    def _get_entity_action(self, desc: str, entity_name: str) -> str:
        """获取行为"""
        prompt = f"观察：{desc}" \
                 f"根据上述观察推断{entity_name}在做什么\n" \
                 f"{entity_name}在做"
        action = chat_glm.processing(prompt)
        action = "解析action"
        return action

    def _brief_summary(self, memories):
        """对记忆进行简要总结"""
        # 生成总结
        prompt = f"相关记忆如下:\n {memories}\n\n" \
                 f"对上述记忆生成3-5条简要的总结(示例格式(1))"
        summary = chat_glm.processing(prompt)
        summary = "解析总结"
        return summary

    def summarize_for_reacting(self, desc: str, now: Optional[MyDateTime] = None) -> str:
        """反应前的准备：总结相关记忆"""
        # 根据两个模版「[观察者] 和 [被观察实体] 的关系是什么」和「[被观察实体] 正在 [被观察实体的动作状态]」构造 query
        subject = self._get_entity_from_observation(desc)
        subject_action = self._get_entity_action(desc, subject)
        q1 = f"{self.name}和{subject}是什么关系？"
        q2 = f"{subject}正在{subject_action}"
        # 检索记忆库，获取相关记忆
        memories1 = self.brain.remembering(q1, now)
        memories1 = self.brain.remembering(q2, now)
        # 合并记忆
        memories = ""
        # 生成总结
        summary = self._brief_summary(memories)
        return summary

    def _reacting(self, desc: str, suffix: str, now: Optional[MyDateTime] = None) -> str:
        """对观察进行反应"""
        agent_summary_description = self.get_summary(now=now)
        agent_status = "获取想在的状态"
        relevant_memories_str = self.summarize_for_reacting(desc, now)
        prompt = f"{agent_summary_description}\n" \
                 f"现在是{str(now)}\n" \
                 f"{self.name}现在的状态: {self.status}\n" \
                 f"观察: {desc}\n" \
                 f"{self.name}的记忆中相关的内容:{relevant_memories_str}\n\n" \
                 f"{suffix}"

        # 判断长度
        reaction = chat_glm.processing(prompt)
        return reaction

    def generate_reaction(self, observation: str, now: Optional[datetime] = None) -> Tuple[bool, str]:
        suffix = f"{self.name}应该对观察做出反应吗？如果是，应该做出什么反应？在一行中进行回复" \
                 f"如果做出的反应是对话，输出格式为: SAY: 要说的话\n" \
                 f"否则输出格式为: REACT: 做出的反应(如果有的话)\n" \
                 f"要么什么都不做，做出反应，要么说些什么，但不能两者兼而有之。\n"
        reaction = self._reacting(observation, suffix, now)
        parse_reaction = "解析反应"
        # TODO 根据现在时间和反应重新生成计划
        if "REACT:" in reaction:
            return False, f"{self.name} {parse_reaction}"
        if "SAY:" in reaction:
            return True, f"{self.name} said {parse_reaction}"
        else:
            return False, reaction

    def generate_dialogue_response(self,
                                   observation: str,
                                   object_name,
                                   now: Optional[datetime] = None,
                                   is_first=True) -> Tuple[bool, str]:
        """生成聊天内容"""
        reaction = "打算做出的反应"
        suffix_first = f"{self.name}的反应: {reaction}, {self.name}可能会对{object_name}说什么？"
        dialogue_history = "\n".join(self.dialogue_history)
        suffix_next = f"对话历史：\n{dialogue_history}" \
                      f"{self.name}可能如何回答{object_name}?" \
                      f"如果要结束对话，输出格式为：GOODBYE:要说的话" \
                      f"否则，输出格式为: SAY:要说的话\n"

        if is_first:
            reply = self._reacting(observation, suffix_first, now)
        else:
            reply = self._reacting(observation, suffix_next, now)

        parse_reply = "解析回复"
        # 将对话添加到历史中
        self.dialogue_history.append({"time": now, "name": self.name, "say": parse_reply})

        if "GOODBYE:" in reply:
            return False, f"{self.name} said {parse_reply}"
        if "SAY:" in reply:
            return True, f"{self.name} said {parse_reply}"
        else:
            return False, reply

    def _generate_summary_description(self, now: Optional[datetime] = None) -> str:
        """生成总结"""
        summary_contains: [str] = [(f"{self.name}的主要个性", f"给定以下陈述，人们会怎么评价{self.name}的主要个性?\n"),
                                   (f"{self.name}当前的工作日常", f"给定以下陈述，{self.name}将如何总结自己的工作日常?\n"),
                                   (f"{self.name}对最近生活的感受", f"给定以下陈述，{self.name}感到最近的生活:")]

        summary_str = ""
        for query, prompt in summary_contains:
            related_memories = self.brain.remembering(query, now)
            related_memories_str = "\n".join([str(mem) for mem in related_memories])
            prompt += related_memories_str
            summary = chat_glm.processing(prompt)
            summary = "解析总结"
            summary_str += summary
        return summary_str

    def get_summary(self, now: Optional[MyDateTime] = None, force_refresh: bool = False) -> str:
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

    def get_full_header(
            self, force_refresh: bool = False, now: Optional[MyDateTime] = None
    ) -> str:
        """返回代理状态、摘要和当前时间的完整标题。"""
        now = datetime.now() if now is None else now
        summary = self.get_summary(force_refresh=force_refresh, now=now)
        current_time_str = now.strftime("%B %d, %Y, %I:%M %p")
        return (
            f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"
        )

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            + "\nThe {entity} is"
        )
        return (
            self.chain(prompt).run(entity=entity_name, observation=observation).strip()
        )

    def get_schedule_summary(self, now: MyDateTime, update=True):
        """获取前一天的日程摘要"""

        if self.schedule and (not self.schedule_summary or update):
            summary = self._brief_summary("\n".join(self.schedule))
            summary = "解析摘要"
            yesterday = now.past(1)
            month = yesterday.get("month", "N/A")
            day = yesterday.get("day", "N/A")
            weekday = yesterday.get("weekday", "N/A")
            self.schedule_summary = f"昨天是{month}月{day}日，{weekday}。{self.name}做的主要事情有:" + summary

            self.schedule = []
        return self.schedule_summary

    def planning(self, now: MyDateTime):
        """生成计划"""
        # 1. 获取摘要性描述和前一天的日程摘要
        summary_description = self.get_summary(now=now)
        schedule_summary = self.get_schedule_summary(now=now)
        month = now.get("month", "N/A")
        day = now.get("day", "N/A")
        weekday = now.get("weekday", "N/A")
        # 2. 生成粗略规划
        prompt = f"{summary_description}\n" \
                 f"{schedule_summary}\n" \
                 f"今天是{month}月{day}日，{weekday}。这里是{self.name}今天的大致计划(示例格式:8:00-8:30 - 洗漱、刷牙):"
        rough_plans = chat_glm.processing(prompt)
        rough_plans = "解析计划"
        # 3. 将粗略规划保存到记忆流中
        # 4. 拆解计划
        detailed_plans = []
        for rp in rough_plans:
            prompt = f"粗略计划:{rp}\n" \
                     f"根据粗略计划的时间范围和活动目标，生成更细致的活动目标，每个活动限制在30分钟以内;\n" \
                     f"细致计划(示例格式: 8:00 - 8:30 - 洗漱、刷牙):"
            dp = chat_glm.processing(prompt)
            dp = []  # "解析计划"  {"start": "8:00", "end": "8:30", "task": "起床、刷牙、吃饭"}
            detailed_plans.extend(dp)
        # 5. 保存计划
        self.plans = detailed_plans

    def _replan(self, observation, reaction, now: MyDateTime):
        """重新生成计划"""
        old_plans = "\n".join(self.plans)
        prompt = f"现在是{now}\n" \
                 f"{self.name}的状态:{self.status}\n" \
                 f"观察:{observation}\n" \
                 f"反应:{reaction}\n" \
                 f"{self.name}原来的计划如下:{old_plans}\n\n" \
                 f"根据{self.name}的观察、反应以及原来的计划，从现在开始为他制定一个大致的计划(示例格式:8:00-8:30 - 洗漱、刷牙)"
        new_plans = chat_glm.processing(prompt)
        new_plans = "解析计划"
        return new_plans

    def update_status(self):
        current_time = self.get_current_time()
        need_replan = True
        for task_temp in self.plan:
            task_to_temp = datetime.strptime(task_temp['to'], '%H:%M')
            if task_to_temp.time() > current_time.time():
                self.status = task_temp['task']
                need_replan = False
                break
        if need_replan:
            new_plan = self.make_plan()
            self.status = new_plan[0]['task']
        return self.status

    def reacting(self):
        """反应"""
        pass

    def observing(self):
        """观察"""
        pass
