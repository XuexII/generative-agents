from typing import Optional, Dict, Any, Union, List, Tuple
import torch
from datetime import datetime
import logging
from collections import OrderedDict
import heapq
import torch.nn.functional as F

from pydantic import BaseModel
from queue import Queue
import math
import numpy as np
from flashtext import KeywordProcessor
from collections import defaultdict, Counter
from blocks.time_block import MyDateTime
from blocks.location_block import MyLocation
from blocks.retriever import Observation, Retriever
from processors.chatglm import chat_glm


def get_time():
    """
    TODO 获取当前时间
    """
    time: MyDateTime = MyDateTime()
    return time


def get_loc():
    """
    获取地点
    """
    loc: MyLocation = MyLocation()
    return loc


class Brain:
    """
    1. 保存记忆流
    2. 实现记忆的检索
        2.1 时近性
        2.2 重要性
        2.3 相关性
    """
    _cur: int = 0
    # memories: Dict[int: Observation] = {}  # 记忆流， 按时间存放
    memories: Retriever  # 记忆流， 按时间存放
    reflection_threshold: float  # 反思阈值
    plans: List[str] = []  # 当前计划
    aggregate_importance: float = 0.0  # 跟踪最近记忆的重要性总和
    alpha: float = 1  # 时近性的权重
    beta: float = 1  # 重要性的权重
    gamma: float = 1  # 相关性的权重

    max_tokens_limit: int = 1000
    reflecting: bool = False  # 是否正在反思

    # -------------------可能弃用---------------
    insights: List[Observation] = []  # 反思
    cumul_score: float = 0.  # 最新事件的重要性分数总和
    actions: List[Observation] = []  # 一天的经历

    def __init__(self, device, bsz=8, alpha=1, beta=1, gamma=1, threshold=800):
        self.device = device
        self.bsz = bsz  # 计算余弦相似度时的batch size
        self.alpha = alpha  # 时近性的权重
        self.beta = beta  # 重要性的权重
        self.gamma = gamma  # 相关性的权重
        self.reflection_threshold = threshold
        self.reflecting = False
        # 创建用于搜索记忆的算法
        self.kps = KeywordProcessor(case_sensitive=False)  # 大小写不敏感
        self.mtoi = {"memories": defaultdict(set), "insights": defaultdict(set)}

    def _get_topics_of_reflection(self, last_k=50):
        """
        确定反思的问题
        """
        # 1. 根据智能体记忆流中最近的100条记录，构造prompt来询问LLM
        """{记忆流中的最近 100 条记录} Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?”"""

        last_memories = [str(mem) for mem in self.memories[-last_k:]]
        memo_str = "\n".join(last_memories)
        prompt = f"{memo_str}。仅根据上述信息，我们能回答关于陈述中主题的3个最突出的高层次问题是什么？"
        questions = chat_glm.processing(prompt)
        questions = "解析问题"  # TODO 解析时需要关联相关记忆，注意还需要保存
        return questions

    def _insighting(self, topic, now: MyDateTime) -> List[str]:
        """
        感悟
        """
        name = "张三"
        related_statements = ""
        prompt = f"关于{name}的相关描述如下：\n{related_statements}\n从以上陈述中可以推断出哪5个与回答以下问题相关的高层次小说见解？\n\
        不要包含与问题无关的任何见解；不要重复任何已经做出的见解；\n\n\
        问题:{topic}\n\n\
        （示例格式：感悟（因为 1, 5, 3））"

        related_memories = self._retrieval(topic, now=now)
        memo_str = "\n".join(
            [f"{i}. {str(memo)}" for i, memo in enumerate(related_memories)]
        )
        prompt = f"问题:{topic};\n" \
                 f"关于问题的相关陈述如下：\n" \
                 f"{memo_str}\n" \
                 f"从以上陈述中可以推断出哪5个与回答问题相关的高层次见解？\n\
                不要包含与问题无关的任何见解；不要重复任何已经做出的见解；\n\n\
                （示例格式：感悟（因为 1, 5, 3））\n"
        insights_str = chat_glm.processing(prompt)
        insights = "解析感悟"

        return insights

    def _reflecting(self, now: Optional[MyDateTime] = None) -> List[str]:
        """反思"""

        new_insights = []
        # 1. 确定需要反思的问题
        topics = self._get_topics_of_reflection()
        for topic in topics:
            # 2. 进行反思
            insights = self._insighting(topic, now)
            for insight in insights:
                # 3. 保存反思
                self.memorizing(insight, now=now)
            new_insights += insights
        return new_insights

    def _evaluate_importance(self, desc) -> float:
        """
        计算重要性
        目的：为智能体觉得重要的记忆对象赋予更高的得分
        实现：使用LLM输出一个[1,10]之间的分数
        """
        """
        On the scale of 1 to 10, where 1 is purely mundane
        (e.g., brushing teeth, making bed) and 10 is
        extremely poignant (e.g., a break up, college
        acceptance), rate the likely poignancy of the
        following piece of memory.
        Memory: buying groceries at The Willows Market
        and Pharmacy
        Rating: <fill in>
        """
        prompt = f"在 1 到 10 的范围内，其中 1 是完全普通的（例如刷牙和整理床铺），10 是非常深刻的（例如分手和大学录取），请评估以下记忆片段可能的深刻程度：\n\
        记忆：{desc}；\n等级: <填写>"
        score = chat_glm.processing(prompt)
        # TODO 请求模型。解析分数
        score = 0
        return score

    def _evaluate_importance_batch(self, desc_list) -> List[float]:
        """批量计算重要性"""
        raise NotImplementedError

    def memorizing(self, desc: str, now: Optional[MyDateTime] = None,
                   place: Optional[MyLocation] = None, pointers=[]):
        """
        进行记忆
        """
        # 1. 计算重要性
        impt_score = self._evaluate_importance(desc)
        # 2. 累计重要性
        self.cumul_score += impt_score

        # 3. 获取词向量
        embed = self.get_embedding(desc)
        obv = Observation(create_time=now,
                          view_time=now,
                          loc=place,
                          desc=desc,
                          embed=embed,
                          importance=impt_score,
                          pointers=pointers
                          )
        # 4. 保存记忆
        result = self.memories.add_observations([obv], now=now)

        # 5. 判断是否需要反思
        if self.cumul_score > self.reflection_threshold and not self.reflecting:
            self.reflecting = True
            self._reflecting(now=now)
            # 反思结束将分数归零
            self.cumul_score = 0
            self.reflecting = False

        return result

    def memorizing_batch(self, desc_list, now: Optional[MyDateTime] = None):
        raise NotImplementedError

    def remembering(self, query: str, now: Optional[MyDateTime] = None) -> List[Observation]:
        """
        检索并取回记忆
        """
        if now is not None:
            with mock_now(now):
                return self.memories.get_relevant_memories(query)
        else:
            return self.memories.get_relevant_memories(query)

    def _get_memories_until_limit(self, consumed_tokens):
        """获取满足长度限制的最近记忆"""
        result = []
        for mem in self.memories[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += "获取token数量"
            if consumed_tokens < self.max_tokens_limit:
                result.append(mem)
        raise NotImplementedError

    def _evaluate_relevance(self):
        """
        计算相关性
        目的：为与当前情况紧密相关的记忆对象分配一个更高的得分
        实现：使用语言模型为每个记忆的文本描述生成一个 embedding，然后计算记忆 embedding 和 query embedding 的余弦距离作为相关性

        """
        score = 0
        return score

    def get_embedding(self, desc):
        """获取词向量"""

        return []

    def cos_sim(self, src_embed, compare_embeds, return_list=True):
        src_embed = src_embed.to(self.device)
        compare_embeds = compare_embeds.to(self.device)
        sim = F.cosine_similarity(src_embed, compare_embeds)
        if return_list:
            return sim.cpu().numpy()

    def next_day(self):
        self._cur = len(self.memories)

    def plan_first(self):
        """
        生成计划: 计划包含地点，开始时间和持续时间
        """
        # 获取摘要性描述(姓名、性格、最近经历概况)以及前一天的摘要
        memory_string = f"{[mem.desc for mem in self.memories]}"
