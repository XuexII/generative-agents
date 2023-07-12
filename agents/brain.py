import logging
from typing import Optional, List

import regex as re

from blocks.location_block import MyLocation
from blocks.retriever import Observation, Retriever
from blocks.time_block import MyDateTime
from processors.chatglm import llms_dict
from pydantic import BaseModel

llm = llms_dict["chat_gpt"]


class Brain(BaseModel):
    """
    1. 保存记忆流
    2. 实现记忆的检索
        2.1 时近性
        2.2 重要性
        2.3 相关性
    """
    memories: Retriever  # 记忆流， 按时间存放
    reflection_threshold: float = 8.0  # 反思阈值
    plans: List[str] = []  # 当前计划
    aggregate_importance: float = 0.0  # 跟踪最近记忆的重要性总和
    topk: int = 5

    max_tokens_limit: int = 1000
    reflecting: bool = False  # 是否正在反思

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _get_topics_of_reflection(self, last_k=50) -> List[str]:
        """
        确定反思的问题
        """
        # 1. 根据智能体记忆流中最近的100条记录，构造prompt来询问LLM
        """{记忆流中的最近 100 条记录} Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?”"""

        last_memories = [str(mem) for mem in self.memories[-last_k:]]
        memo_str = "\n".join(last_memories)
        prompt = f"{memo_str}\n\n仅根据上述信息，我们能回答关于陈述中主题的3个最突出的高层次问题是什么？(输出格式：1.)"
        questions_str = llm.processing(prompt)
        questions = []
        try:
            questions = questions_str.strip().split("\n")
            questions = [re.sub(r"^\d+\.\s+", "", i.strip()) for i in questions]
        except:
            logging.error(f"解析反思问题:{questions_str}的时候出错。。。")

        return questions

    def _insighting(self, topic, now: MyDateTime) -> List[str]:
        """
        感悟
        """
        # 获取与问题相关的记忆
        related_memories: List[Observation] = self.remembering(topic, now=now)
        memo_str = "\n".join(
            [f"{i}. {str(memo)}" for i, memo in enumerate(related_memories)]
        )
        prompt = f"问题:{topic}\n" \
                 f"关于问题的相关陈述如下：\n" \
                 f"{memo_str}\n" \
                 f"从以上陈述中可以推断出哪5个与回答问题相关的高层次见解？\n" \
                 f"见解尽可能简洁, 不要包含与问题无关的任何见解；不要重复任何已经做出的见解；\n\n" \
                 f"（示例格式：感悟^因为^[1,5,3]）\n"
        insights_str = llm.processing(prompt)
        insights = []
        try:
            all_insights = insights_str.strip().split("\n")
            for insight in all_insights:
                try:
                    insight = re.sub(r"^\d+\.\s+", "", insight.strip())
                    insight, _, cause_str = insight.split("^")
                    insights.append(insight)
                    # cause = []
                    # if re.match(r"\[[\d,]+\]", cause_str):
                    #     cause = eval(cause_str)
                    # insights.append((insight, cause))
                except:
                    continue
        except:
            logging.error(f"解析感悟:{insights_str}的时候出错。。。")

        return insights

    def _reflecting(self, now: Optional[MyDateTime] = None) -> List[str]:
        """反思"""

        new_insights = []
        # 1. 确定需要反思的问题
        topics = self._get_topics_of_reflection(last_k=50)
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
        prompt = f"在 1 到 10 的范围内，其中 1 是完全普通的（例如刷牙和整理床铺），10 是非常深刻的（例如分手和大学录取），请评估以下记忆片段可能的深刻程度：\n" \
                 f"记忆：{desc};等级: <填写>"
        score_str = llm.processing(prompt)
        score = 1
        try:
            score = re.search(r"(?<=[^\d]|^)(10|\d{1})(?:[^\d]|$)", score_str).group()
            score = int(score)
        except:
            logging.error(f"解析分数：{score_str}的过程中出错。。。")
        return score

    def _evaluate_importance_batch(self, desc_list) -> List[float]:
        """批量计算重要性"""
        raise NotImplementedError

    def memorizing(self, desc: str, now: MyDateTime,
                   place: Optional[MyLocation] = None, pointers=[]):
        """
        进行记忆
        """
        # 1. 计算重要性
        impt_score = self._evaluate_importance(desc)
        # 2. 累计重要性
        self.cumul_score += impt_score

        obv = Observation(create_time=now,
                          view_time=now,
                          loc=place,
                          desc=desc,
                          importance=impt_score,
                          pointers=pointers
                          )
        # 4. 保存记忆
        result = self.memories.add_observations([obv], now=now)

        # 5. 判断是否需要反思
        if self.cumul_score > self.reflection_threshold and not self.reflecting:
            logging.info(f"开始感悟。。。")
            self.reflecting = True
            self._reflecting(now=now)
            # 反思结束将分数归零
            self.cumul_score = 0
            self.reflecting = False

        return result

    def memorizing_batch(self, desc_list, now: MyDateTime,
                         place: Optional[MyLocation] = None, pointers=[]):
        """批量添加记忆"""
        for desc in desc_list:
            self.memorizing(desc, now, place)

    def remembering(self, query: str, now: Optional[MyDateTime] = None) -> List[Observation]:
        """
        检索并取回记忆
        """
        return self.memories.get_relevant_memories(query, self.topk, now)

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
