from typing import List, Dict, Optional, Any, Tuple

from pydantic import BaseModel, Field

from blocks.location_block import MyLocation
from blocks.time_block import MyDateTime
from blocks.vectorstores.my_faiss import FAISSVectorStore


class Observation(BaseModel):
    """
    记忆流最基本的元素，观察
    """
    id: int
    create_time: MyDateTime  # 发生时间
    view_time: MyDateTime  # 最近访问时间
    loc: Optional[MyLocation] = None  # 发生地点
    desc: str  # 发生的事情
    recency: float = 10.  # 时近性，默认为10分
    importance: float = 0.  # 重要性，默认为0分
    pointers: List = []  # 作为反思时指向的记忆流
    date_prefix: str = "现在是"  # 时间前缀
    loc_prefix: str = "现在在"  # 地点前缀

    def get_recency(self, now: MyDateTime, decay_rate=0.99):
        """
        获取时近性: 自上次检索记忆以来根据沙盒游戏内小时数呈指数衰减的函数，衰减因子为 0.99
        """
        interval_seconds = now - self.view_time
        hours_passed = interval_seconds / 3600
        # self.recency *= (self.decay_factor ** hour)
        recency = decay_rate ** hours_passed
        self.view_time = now
        return recency

    def get_importance(self):
        return self.importance

    def __str__(self):
        return f"{str(self.create_time)}: {self.desc}"

    def __getattr__(self, name):
        if name == 'x':
            self.x += 1
            return self.x


def min_max_scaling(x, _min, _max):
    y = (x - _min) / (_max - _min + 1e-9)
    return y


class MaxHeap:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.items = []
        self._min = -float("inf")

    def add(self, item, score):
        if len(self.items) < self.max_size:
            self.items.append((item, score))
        elif score > self._min:
            self.items.pop()
            self.items.append((item, score))
        self.items = sorted(self.items, key=lambda x: x[1], reverse=True)
        self._min = self.items[-1][1]

    def get_max(self, k=10):
        return self.items[:k]


class Retriever(BaseModel):
    vector_store: FAISSVectorStore  # 向量库
    memory_stream: List[Observation] = List  # 记忆流存储位置
    impt_max_heap: MaxHeap = MaxHeap() # 存储最重要的记忆
    search_kwargs: Dict = Field(default_factory=lambda: dict(k=100))  # 用于检索时的参数
    decay_rate: float = 0.99  # 时近性衰减因子
    k: int = 4  # 搜索的最大文档数
    other_score_keys: List[Tuple] = []  # (key, weight)
    default_salience: Optional[float] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True  # 允许任意类型作为字段类型

    def _get_combined_score(self, recency, importance, relevance):
        """获取综合分数"""
        alpha = 1.0
        beta = 1.0
        gamma = 1.0
        score = alpha * min_max_scaling(recency, 0, 1) + \
                beta * min_max_scaling(importance, 0, 10) + \
                gamma * min_max_scaling(relevance, 0, 1)
        return score

    def _get_relevance_docs(self, query: str) -> List[Tuple[Observation, float]]:
        """获取重要的文档"""
        obv_and_scores: List[Tuple[int, float]]
        obv_and_scores = self.vector_store.cosine_search(
            query, **self.search_kwargs
        )
        results = []
        for obv_id, relevance in obv_and_scores:
            if 0 <= obv_id <= len(self.memory_stream):
                obv = self.memory_stream[obv_id]
                results.append((obv, relevance))

        return results

    def _get_importance_docs(self, k):
        id_and_score = self.impt_max_heap.get_max(k)
        results = []
        for obv_id, importance in id_and_score:
            if 0 <= obv_id <= len(self.memory_stream):
                obv = self.memory_stream[obv_id]
                results.append((obv, self.default_salience))
        return results

    def get_relevant_memories(self, query: str, k, now) -> List[Observation]:
        """
        获取记忆
        1. 分别获取最新，最重要，最相似的前10条记忆
        2. 去重后排序，返回指定数量的记忆
        """

        latest_memories = [(obv, self.default_salience) for obv in self.memory_stream[-self.k:]]
        important_memories = self._get_importance_docs(self.k)
        relevant_memories = self._get_relevance_docs(query)

        recall_memories: List = []
        visited = set()
        for obv, relevance in relevant_memories + important_memories + latest_memories:
            if obv.id in visited:
                continue
            visited.add(obv.id)
            # 获取时近性得分
            recency = obv.get_recency(now, self.decay_rate)
            importance = obv.get_importance()
            score = self._get_combined_score(recency, importance, relevance)
            recall_memories.append((obv, score))
        recall_memories = sorted(recall_memories, key=lambda x: x[1], reverse=True)

        recall_memories = [i[0] for i in recall_memories[:k]]
        return recall_memories

    def add_observations(self, observations: List[Observation], **kwargs: Any) -> None:
        """保存记忆"""
        now = kwargs.get("now")
        if now is None:
            now = "获取时间"
        for i, obv in enumerate(observations):
            if not obv.create_time:
                obv.create_time = now
            if not obv.view_time:
                obv.view_time = now
            obv.id = len(self.memory_stream)
            # 添加到向量库
            self.vector_store.add_text(obv.desc, obv.id)
            # 维护大顶堆
            self.impt_max_heap.add(obv.id, obv.importance)

        self.memory_stream.extend(observations)

    def __getitem__(self, index):
        return self.memory_stream[index]
