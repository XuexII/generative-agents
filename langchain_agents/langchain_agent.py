import logging
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any

import regex as re
from langchain.prompts import PromptTemplate

from blocks.location_block import MyLocation, my_map
from blocks.other_blocks import Plan, PlanQueue
from langchain_agents.generative_agent import GenerativeAgent
from utils.utils_log import Logger
from langchain import LLMChain
from blocks.things_block import MyThing

weekday_map = {"1": "Monday", "2": "Tuesday", "3": "Wednesday",
               "4": "Thursday", "5": "Friday", "6": "Saturday", "7": "Sunday"}


class LangChainAgent(GenerativeAgent):
    id: str  # 唯一id
    plans: PlanQueue = PlanQueue()  # 一天的计划
    schedule: List = []  # 当天的日程
    schedule_summary = ""  # 当天的日程摘要
    loc: str  # 当前所在位置
    known_areas: List = []  # 知道的位置
    emoji: str = ""  # emoji
    has_update_plan: bool = False
    log_file: Any = None
    logger: Optional[logging.RootLogger] = None

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory
        )
        # return LLMChain(
        #     llm=self.llm, prompt=prompt, verbose=self.verbose, memory=None
        # )

    def init_agent(self, log_dir, memories: List[str]):
        os.makedirs(log_dir, exist_ok=True)
        logger = Logger(os.path.join(log_dir, f"{self.name}1.txt"),
                        clevel=logging.INFO, flevel=logging.INFO, fmt='%(message)s')
        self.logger = logger
        for memory in memories:
            self.memory.add_memory(memory)

        self.logger.info(f"{self.name}初始化成功")

    def summarize_related_memories(self, name, desc) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template(
            "{q1}?\n"
            "Context from memory:\n"
            "{relevant_memories}\n"
            "Relevant context: "
        )
        # entity_name = self._get_entity_from_observation(observation)
        # entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {name}"
        q2 = f"{name} is {desc}"
        return self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2]).strip()

    def _generate_reaction(
            self, observation: Tuple, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """对给定的观察或对话行为做出反应。"""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        name, desc = observation
        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_related_memories(name, desc)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation=f"{name} is {desc}",
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip()

    def generate_reaction(
            self, observation: Tuple, now: Optional[datetime] = None
    ) -> Tuple[str, str]:
        """React to a given observation."""
        call_to_action_template = (
                "{agent_name} can choose to do nothing, do something or say something react to the observation. Reaction should be very very brief\n"
                "If do nothing,write:\nPASS:None\n"
                + 'If reaction implies interactive behavior, must generate the words that Tommie might say, write:\nSAY: "what to say"\n'
                + "\notherwise, write:\nREACT: what will reaction"
                + "\nEither do nothing, react, or say something but not both."
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        # AAA
        name, desc = observation
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                                            f"{name} is {desc} and reacted by {result}",
                self.memory.now_key: now,
            },
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            self.replanning(f"{name} is {desc}", reaction)
            return "REACT", f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = self._clean_response(result.split("SAY:")[-1])
            # TODO 修改
            self.replanning(f"{name} is {desc}", said_value)
            return "SAY", f"{self.name} said {said_value}"
        else:
            return "PASS", result

    def generate_dialogue_response(
            self, observation: Tuple, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """生成对话内容"""
        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        name, desc = observation
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                                                f"{name} is {desc} and said {farewell}",
                    self.memory.now_key: now,
                },
            )
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                                                f"{name} is {desc} and said {response_text}",
                    self.memory.now_key: now,
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

    def _get_date_info(self, now: Optional[datetime] = None, offset: int = 0):
        now = now if now else datetime.now()
        now = now + timedelta(offset)
        _, week, weekday = now.isocalendar()
        date_str = now.strftime("%B %d,%H:%M")
        date, hour = date_str.split(",")
        return date, hour, weekday_map.get(str(weekday), "Monday")

    def get_schedule_summary(self, update=True):
        """获取前一天的日程摘要"""

        if self.schedule and (not self.schedule_summary or update):
            schedule_str = ";".join(self.schedule)
            prompt = PromptTemplate.from_template(
                "{name}'s schedule for yesterday:\n"
                "{schedule_str}\n\n"
                "Generate 3-5 brief summaries of {name}'s schedule for yesterday (output format: Time Range: Main Event):"
            )
            summaries = self.chain(prompt).run(name=self.name, schedule_str=schedule_str).strip()
            yestd, hour, yestw = self._get_date_info(offset=-1)

            self.schedule_summary = f"On {yestw} {yestd}, {self.name}" + summaries
            self.schedule = []
        return self.schedule_summary

    def parse_plan(self, plan_str):
        p = re.compile(r"(?P<start>\d{1,2}:\d{1,2})[-—](?P<end>\d{1,2}:\d{1,2})\]?:\s+(?P<task>.+)")
        plans = []
        for plan in plan_str.split("\n"):
            match = p.search(plan)
            if match:
                info = match.groupdict()
                if all([isinstance(i, str) and len(i) > 1 for i in info.values()]):
                    plans.append(Plan(**info))
            else:
                print()
                self.logger.info(f"模型生成的计划无法解析：{plan}")
        return plans

    def planning(self, now: Optional[datetime] = None):
        """生成计划"""
        # 1. 获取摘要性描述和前一天的日程摘要
        summary_description = self.get_summary(now=now)
        schedule_summary = self.get_schedule_summary()

        date, hour, weekd = self._get_date_info(now=now)
        # 2. 生成粗略规划
        prompt = PromptTemplate.from_template(
            "{summary_description}\n"
            "{schedule_summary}\n"
            "plan example format:\n"
            "[7:14-11:00]: Wake up and complete the morining routine\n"
            "[11:00-12:00]: eat something\n"
            "[8:35-17:10]: Go to school and study\n"
            "[17:10-22:30]: Play CSGO\n"
            "[22:30-7:30]: Go to sleep\n\n"
            "Today is {weekd} {date}. Here is {name}’s very brief plan today in broad strokes from {hour} today:"
        )
        plans = self.chain(prompt).run(summary_description=summary_description,
                                       schedule_summary=schedule_summary,
                                       weekd=weekd,
                                       date=date,
                                       name=self.name,
                                       hour=hour).strip()

        # plans = "[18:30-19:30]: Tommie settles into their workspace and checks emails and messages.\n[19:30-20:00]: Tommie attends a team meeting to discuss ongoing projects and tasks.\n[20:00-20:30]: Tommie starts working on their assigned programming tasks for the day.\n[20:30-21:00]: Tommie takes a short break and chats with colleagues about non-work related topics.\n[21:00-22:30]: Tommie continues working on programming tasks, focusing on debugging and testing.\n[22:30-23:00]: Tommie wraps up the day's work, organizes files, and prepares for tomorrow.\n[23:00-23:15]: Tommie says goodbye to coworkers and leaves the SpaceX office.\n[23:15-23:30]: Tommie walks or takes transportation back to Tommie's house.\n[23:30-00:00]: Tommie arrives home and unwinds, possibly engaging in a hobby or watching TV.\n[00:00-00:30]: Tommie gets ready for bed, completing the evening routine.\n[00:30-07:00]: Tommie sleeps and rests for the night."
        rough_plans = self.parse_plan(plans)
        # 3. 将粗略规划保存到计划中
        self.plans.batch_put(rough_plans)

    def replanning(self, observation, reaction, now: Optional[datetime] = None):
        """重新生成计划"""
        prompt = PromptTemplate.from_template(
            "It's {hour}\n"
            "{name}'s status: {status}\n"
            "Observation: {observation}\n"
            "Reaction: {reaction}\n"
            "{name}'s previous plans:\n"
            "{old_plans}\n"
            "plan example format:\n"
            "[7:14-11:00]: Wake up and complete the morining routine\n"
            "[11:00-12:00]: eat something\n"
            "[8:35-17:10]: Go to school and study\n"
            "[17:10-22:30]: Play CSGO\n"
            "[22:30-7:30]: Go to sleep\n\n"
            "Based on {name}'s observation,reaction,and previous plans,make new very brief plan today in broad strokes from now on, without any explanations:"
        )
        date, hour, weekd = self._get_date_info(now=now)
        old_plans = str(self.plans)
        plans = self.chain(prompt).run(hour=hour,
                                       name=self.name,
                                       status=self.status,
                                       observation=observation,
                                       reaction=reaction,
                                       old_plans=old_plans).strip()

        new_plans = self.parse_plan(plans)
        self.plans = PlanQueue()
        self.plans.batch_put(new_plans)
        self.has_update_plan = True
        self.logger.info(f"更新了原计划: \n{str(self.plans)}")
        self.logger.info("=" * 66 + "\n")

    def disassemble_plan(self, start, end, plan) -> List[Plan]:
        """
        解析计划
        """
        prompt = PromptTemplate.from_template(
            "Here is {name}'s plan from {start} to {end}: {plan}\n\n"
            "Decomposes the plan to create finer-grained actions, each action must be Be very very brief and limited within 5-15 minutes\n"
            "actions format example:\n"
            "[7:14-7:20]: getting out of bed\n"
            "[7:20-7:25]: on the way to lavatory\n"
            "[7:25-7:35]: brushing his teeth\n"
            "Here is {name}'s finer-grained actions:"
        )
        detailed_plans = self.chain(prompt).run(name=self.name,
                                                start=start,
                                                end=end,
                                                plan=plan).strip()

        detailed_plans = self.parse_plan(detailed_plans)
        dp_str = "\n".join([str(i) for i in detailed_plans])
        self.logger.info(f"将计划:{plan} 拆解为: \n{dp_str}")
        return detailed_plans

    def path_finding(self, action, now: Optional[datetime] = None, max_deep=10):
        """行动"""
        summary_description = self.get_summary(now=now)

        prompt = PromptTemplate.from_template(
            "{summary_description}\n"
            "{name} is currently in The {location}, "
            "that has {surroundings}\n"
            "{name} knows of the following areas: {known_areas}\n"
            "* Prefer to stay in the current area if the activity can be done there\n"
            "{name} is planning to {action}. Which area should {name} go to?\n\n"
            "If dont' need change the area, write:\nSTOP: None\n"
            "Otherwise, write:\nMOVE: area\n"
        )
        loc_obj = my_map.get_loc(self.loc)
        path_list = []

        known_areas = "、".join(self.known_areas)
        while loc_obj:
            path_list.append(loc_obj)

            if len(path_list) > max_deep:
                break
            location = loc_obj.get_all_path()
            surroundings = loc_obj.get_surroundings()
            result = self.chain(prompt).run(summary_description=summary_description,
                                            name=self.name,
                                            location=location,
                                            surroundings=surroundings,
                                            known_areas=known_areas,
                                            action=action
                                            ).strip()
            self.logger.info(f"寻找移动路径的结果: {result}")
            # TODO 1. 地点去重； 2. 如果2次都是同一地址直接退出
            if "STOP:" in result:
                # area = self._clean_response(result.split("STOP:")[-1])
                # loc_obj = my_map.get_loc(area)
                break
            elif "MOVE:" in result:
                area = self._clean_response(result.split("MOVE:")[-1])
                loc_obj = my_map.get_loc(area)
                if loc_obj and (loc_obj.is_leaf() or loc_obj == path_list[-1]):
                    if loc_obj == path_list[-1]:
                        loc_obj = None
                    break
            else:
                break
            known_areas = ""
        if loc_obj and loc_obj.name != self.loc:
            path_list.append(loc_obj)
        return path_list

    def update_thing_status(self, action, thing_obj: MyThing):
        """
        更新状态
        """
        # 如果是自身状态，直接更新为具体的动作
        prompt = PromptTemplate.from_template(
            "The {name}'s action is {action}\n"
            "What happens to the state of the {obj_name}?(example format: state: close)\n\n"
        )
        state = self.chain(prompt).run(name=self.name,
                                     action=action,
                                     obj_name=thing_obj.name).strip()
        if state and state.startswith("state:"):
            state = state.replace("state:", "").strip()
            thing_obj.status = state
            self.logger.info(f"{thing_obj.name}的状态更新为: {self.status}")
        else:
            self.logger.info(f"{thing_obj.name}的状态更新失败，无法解析结果：{state}")

    def get_emoji(self, task, action):
        """生成emoji"""
        prompt = PromptTemplate.from_template(
            "The {name}'s action: {action}\n\n"
            "Here is the emoji used to represent {name}'s action:"
        )
        if task:
            action = f"{task}:{action}"
        emoji = self.chain(prompt).run(name=self.name,
                                     action=action).strip()
        self.emoji = emoji
        self.logger.info(f"更新{self.name}的emoji: {self.emoji}")

    def moving(self, path_list: List[MyLocation]):
        """
        移动，在移动的过程中需要观察
        """
        original_loc = path_list[0]
        original_loc.remove(self)
        last_loc = path_list[-1]
        last_loc.accept(self)
        self.loc = last_loc.name

    def acting(self, action, now: Optional[datetime] = None) -> None:
        # 寻路
        if not now:
            now = datetime.now()
        date, hour, weekd = self._get_date_info(now=now)
        self.schedule.append(f"{hour}: {action}")
        # 寻找路径
        path_list = self.path_finding(action, now)
        if len(path_list) > 1:
            paths = "->".join([path.get_all_path() for path in path_list])
            self.logger.info(f"{self.name}开始移动，移动轨迹为: {paths}")
            self.moving(path_list)
            # 需要生成交互对象的状态
            # obj = path_list[-1]
            # if getattr(obj, "status"):
            #     obj.update_status(f"{self.name}'s action is {action}", obj.name)
        else:
            self.logger.info(f"该行动中不需要进行移动或没有找到合适的地点")

        # 将行动添加到记忆中
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} is {action}",
                self.memory.now_key: now,
            },
        )

    def run_conversation(self, agents: List[GenerativeAgent],
                         initial_observation: str,
                         max_turns=20) -> None:
        """Runs a conversation between agents."""
        mod, observation = agents[1].generate_reaction((self.name, initial_observation))
        if mod != "SAY":
            self.logger.info(f"{agents[1].name}没有回应{self.name}的对话: {observation}")
            return

        self.logger.info(f"开始对话".center(66, "="))
        turns = 0
        name = agents[1].name
        while True:
            break_dialogue = False
            for agent in agents:
                self.logger.info(observation)
                stay_in_dialogue, observation = agent.generate_dialogue_response((name, observation))
                name = agent.name
                # observation = f"{agent.name} said {reaction}"
                if not stay_in_dialogue:
                    break_dialogue = True
            if turns > max_turns or break_dialogue:
                break
            turns += 1
        self.logger.info(observation)
        self.logger.info(f"对话结束".center(66, "="))

    def reacting(self, mod, task, action, agent_list) -> None:
        """行动"""
        # 1. 解析行动
        # 2. 执行行动
        if mod == "SAY":
            self.logger.info(f"{self.name}做出的反应是对话")
            self.run_conversation([self, agent_list[0]], action)
        elif mod == "REACT":
            self.logger.info(f"{self.name}做出的反应是行动: {action}")
            if task:
                self.acting(f"{task}: {action}")
            else:
                self.acting(action)
        else:
            self.logger.info(f"{self.name}没有做出反应: {action}")
        # 更新状态
        self.status = action
        # 更新emoji
        self.get_emoji(task, action)

    def observing(self, now: Optional[datetime] = None):
        """
        观察-->前端实现，视野范围内的其它智能体和对象存入这个他的记忆
        """
        loc_obj: MyLocation = my_map.get_loc(self.loc)
        # 将观察结果添加到记忆中
        obj_list = []
        if loc_obj:
            obj_list = loc_obj.viewed(self)
        for obj in obj_list:
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                                                f"{obj.name} is {obj.status}",
                    self.memory.now_key: now,
                },
            )
        return obj_list

    def start(self, now: Optional[datetime] = None, time_step: int = 1):
        now = now if now else datetime.now()
        self.logger.info(f"{self.name}在{now.strftime('%Y-%m-%d')}的日志".center(66, "="))
        # 1. 制定计划
        self.planning(now)
        self.logger.info(f"制定一天的粗略计划如下: \n{str(self.plans)}")
        start_time = time.time()
        # 2. 执行计划，并在每个时间步感知周围环境
        #    2.1 将自己的行动及感知到的信息保存到记忆中
        self.logger.info("=" * 66 + "\n")
        # TODO 需要保证不会陷入死循环
        while not self.plans.empty():
            # 获取计划
            plan = self.plans.get()
            # 拆解计划
            start, end, task = plan.get_info()
            detailed_plans = self.disassemble_plan(start, end, str(plan))
            # 执行计划
            for dp in detailed_plans:
                time.sleep(10)
                start, end, action = dp.get_info()
                # 打印计划信息
                now = datetime.now()
                # 执行计划
                self.logger.info(f"执行计划".center(66, "="))
                self.logger.info(f"{self.name}的行动: {start}-{end}: {action}")
                self.reacting("REACT", task, action, [])
                self.logger.info(f"执行计划完成".center(66, "="))
                self.logger.info("\n")
                end_time = time.time()
                used_time = end_time - start_time
                # 感知周围环境
                obv_agent_list = []
                if used_time // time_step > 0:
                    obv_agent_list = self.observing()
                # observations = "\n".join([f"{obj.name}，{obj.desc}" for obj in agent_list])
                if obv_agent_list:
                    # 判断采取反应还是继续执行计划
                    agent = obv_agent_list[0]
                    self.logger.info(f"对观察结果进行反应".center(66, "="))
                    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    self.logger.info(f"{date_str}: {self.name}观察到: {agent.name} is {agent.status}")
                    mod, action = self.generate_reaction((agent.name, agent.status), now)
                    # 执行反应动作
                    self.reacting(mod, None, action, obv_agent_list)
                    self.logger.info(f"对观察结果反应完成".center(66, "="))
                    self.logger.info("\n")
                    if self.has_update_plan:
                        self.has_update_plan = False
                        break
