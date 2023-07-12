from langchain.experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
from typing import List, Dict, Optional, Tuple, Union
from blocks.other_blocks import Plan, PlanQueue


class LangChainAgent(GenerativeAgent):
    plans: PlanQueue = PlanQueue()  # 一天的计划
    schedule: List = []  # 当天的日程
    loc: str  # 所在位置

