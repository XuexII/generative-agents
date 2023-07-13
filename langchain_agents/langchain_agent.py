from langchain.experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
from typing import List, Dict, Optional, Tuple, Union
from blocks.other_blocks import Plan, PlanQueue
from datetime import datetime, timedelta
from langchain.prompts import PromptTemplate
import regex as re
import logging
from blocks.location_block import MyLocation, get_loc
import time

weekday_map = {"1": "Monday", "2": "Tuesday", "3": "Wednesday",
               "4": "Thursday", "5": "Friday", "6": "Saturday", "7": "Sunday"}


class LangChainAgent(GenerativeAgent):
    plans: PlanQueue = PlanQueue()  # 一天的计划
    schedule: List = []  # 当天的日程
    loc: str  # 所在位置
    known_areas = []
    emoji: List[str] = []  # emoji

    def generate_reaction(
            self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[str, str]:
        """React to a given observation."""
        call_to_action_template = (
                "Should {agent_name} react to the observation, and if so,"
                + " what would be an appropriate reaction? Respond in one line."
                + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
                + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
                + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        # AAA
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                                            f"{observation} and reacted by {result}",
                self.memory.now_key: now,
            },
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            return "REACT", f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = self._clean_response(result.split("SAY:")[-1])
            return "SAY", f"{self.name} said {said_value}"
        else:
            return "PASS", result

    def generate_dialogue_response(
            self, observation: str, now: Optional[datetime] = None
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
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                                                f"{observation} and said {farewell}",
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
                                                f"{observation} and said {response_text}",
                    self.memory.now_key: now,
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template(
            "{q1}?\n"
            "Context from memory:\n"
            "{relevant_memories}\n"
            "Relevant context: "
        )
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = f"{entity_name} is {entity_action}"
        return self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2]).strip()

    def _get_yesterday(self, now: Optional[datetime] = None, offset: int = 0):
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
            yestd, hour, yestw = self._get_yesterday(offset=-1)

            self.schedule_summary = f"On {yestw} {yestd}, {self.name}" + summaries
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

    def planning(self, now: Optional[datetime] = None):
        """生成计划"""
        # 1. 获取摘要性描述和前一天的日程摘要
        summary_description = self.get_summary(now=now)
        schedule_summary = self.get_schedule_summary()

        date, hour, weekd = self._get_yesterday(now=now)
        # 2. 生成粗略规划
        prompt = PromptTemplate.from_template(
            "{summary_description}\n"
            "{schedule_summary}\n"
            "plan example format:"
            "[7:14-7:45]: Wake up and complete the morining routine\n"
            "[7:45-8:35]: Eat breakfast\n"
            "[8:35-17:10]: Go to school and study\n"
            "[17:10-22:30]: Play CSGO\n"
            "[22:30-7:30]: Go to sleep\n\n"
            "Today is {weekd} {date}. Here is {name}’s brief plan today in broad strokes from {hour} today:"
        )
        plans = self.chain(prompt).run(summary_description=summary_description,
                                       schedule_summary=schedule_summary,
                                       weekd=weekd,
                                       date=date,
                                       name=self.name).strip()

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
            "{old_plans}\n\n"
            "Based on {name}'s observation,reaction,and previous plans,make new brief plan today in broad strokes from now on, without any explanations:"
        )
        date, hour, weekd = self._get_yesterday(now=now)
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

    def disassemble_plan(self, overall_plan) -> List[Plan]:
        """
        解析计划
        """

        prompt = PromptTemplate.from_template(
            "overall plan: {overall_plan}\n"
            "Translate the overall plan into more detailed brief plans, each limited within 5-15 minutes\n"
            "Break down the overall plan into executable and concise actions, each action limited to 5-15 minutes."
            "plan example format:"
            "[7:14-7:20]: Wake up and complete the morining routine\n"
            "[7:20-7:30]: Eat breakfast\n\n"
            "Here is actions:"
        )
        detailed_plans = self.chain(prompt).run(overall_plan=overall_plan).strip()
        detailed_plans = self.parse_plan(detailed_plans)
        return detailed_plans

    def path_finding(self, action, now: Optional[datetime] = None, max_deep=10):
        """行动"""
        summary_description = self.get_summary(now=now)

        prompt = PromptTemplate.from_template(
            "{summary_description}\n"
            "{name} is currently in The {location})"
            "that has {surroundings}\n"
            "{name} knows of the following areas: {known_areas}"
            "* Prefer to stay in the current area if the activity can be done there\n"
            "{name} is planning to {action}. Which area should {name} go to?\n"
            "If the area that will to go can complete the plan, write:\nSTOP: area\n"
            "otherwise, write:\nNEXT: area"
        )
        loc_obj = get_loc(self.loc)
        path_list = [loc_obj]

        known_areas = "、".join(self.known_areas)
        while loc_obj and not loc_obj.is_leaf():
            if len(path_list) > max_deep:
                break
            location = str(loc_obj)
            surroundings = loc_obj.get_surroundings()
            result = self.chain(prompt).run(summary_description=summary_description,
                                            name=self.name,
                                            location=location,
                                            surroundings=surroundings,
                                            known_areas=known_areas,
                                            action=action
                                            ).strip()
            if "STOP:" in result:
                _, area = self._clean_response(result.split("STOP:")[-1])
                loc_obj = get_loc(area)
                path_list.append(loc_obj)
                break
            elif "NEXT:" in result:
                _, area = self._clean_response(result.split("STOP:")[-1])
                loc_obj = get_loc(area)
                path_list.append(loc_obj)
            else:
                break
            known_areas = ""
        return path_list

    def update_status(self, action, obj_name):
        """
        更新状态
        """
        prompt = PromptTemplate.from_template(
            "{action}\n"
            "What happens to the state of the {obj_name}?\n"
            "give the {obj_name}'s state and emoji that matches the state. (example format: state: close)"
        )
        res = self.chain(prompt).run(name=self.name,
                                     action=action,
                                     obj_name=obj_name).strip()
        self.status = res
        logging.info(f"{self.name}的状态更新为: {self.status}")

    def acting(self, action, now: Optional[datetime] = None):
        # 寻路
        path_list = self.path_finding(action, now)
        if len(path_list) > 1:
            paths = "->".join([str(path) for path in path_list])
            logging.info(f"{self.name}开始移动，移动轨迹为: {paths}")
            obj = path_list[-1]
            self.loc = str(obj)
            if getattr(obj, "status"):
                obj.update_status(f"{self.name}'s action is {action}", obj.name)
        else:
            logging.info(f"该行动中不需要进行移动或没有找到合适的地点")

    def run_conversation(self, agents: List[GenerativeAgent],
                         initial_observation: str,
                         max_turns=20) -> None:
        """Runs a conversation between agents."""
        mod, observation = agents[1].generate_reaction(initial_observation)
        if mod != "SAY":
            logging.info(f"{agents[1].name}没有回应{self.name}的对话: {observation}")
            return
        turns = 0
        while True:
            break_dialogue = False
            for agent in agents:
                logging.info(observation)
                stay_in_dialogue, observation = agent.generate_dialogue_response(observation)
                # observation = f"{agent.name} said {reaction}"
                if not stay_in_dialogue:
                    break_dialogue = True
            if turns > max_turns or break_dialogue:
                break
            turns += 1
        logging.info(observation)

    def reacting(self, mod, action, agent):
        """行动"""
        # 1. 解析行动
        # 2. 执行行动
        if mod == "SAY":
            logging.info(f"{self.name}做出的反应是对话")
            self.run_conversation(action, agent)
        else:
            logging.info(f"{self.name}做出的反应是行动: {action}")
            self.status = action

    def start(self, now: Optional[datetime] = None, time_step: int = 60):
        logging.info(f"{self.name}在{str(now)}的日志".center(66, "="))
        # 1. 制定计划
        self.planning(now)
        logging.info(f"制定一天的粗略计划如下: \n{str(self.plans)}")
        start_time = time.time()
        # 2. 执行计划，并在每个时间步感知周围环境
        #    2.1 将自己的行动及感知到的信息保存到记忆中
        logging.info("=" * 66 + "\n")
        # TODO 需要保证不会陷入死循环
        while not self.plans.empty():
            # 获取计划
            plan = self.plans.get()
            # 拆解计划
            detailed_plans = self.disassemble_plan(str(plan))
            # 执行计划
            for dp in detailed_plans:
                start, end, task = dp.get_info()
                self.status = task
                # 打印计划信息
                logging.info(f"{start}-{end}: {task}")
                used_time = time.time()
                # 感知周围环境
                obv_agent_list = []
                now = datetime.now()
                if used_time // time_step > 0:
                    obv_agent_list = self.perceiving(now)
                agent = obv_agent_list[0]
                # observations = "\n".join([f"{obj.name}，{obj.desc}" for obj in agent_list])
                if obv_agent_list:
                    # 判断采取反应还是继续执行计划
                    mod, action = self.generate_reaction(f"{self.name} saw {agent.name} is {agent.status}", now)
                    logging.info(f"{now.}，{self.name}观察到: \n{agent.status}\n"
                                 f"{self.name}决定对该反应{'做出' if flag_break else '不做出'}反应: {action}")
                    # 如果需要做出反应，则更新计划
                    flag = False
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

