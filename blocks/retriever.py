from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple
from blocks.time_block import MyDateTime
from blocks.location_block import MyLocation
import torch
from copy import deepcopy


class Observation(BaseModel):
    """
    记忆流最基本的元素，观察
    """
    create_time: MyDateTime  # 发生时间
    view_time: MyDateTime  # 最近访问时间
    loc: MyLocation  # 发生地点
    desc: str  # 发生的事情
    embed: Optional[torch.Tensor] = None  # 词向量
    recency: float = 10.  # 时近性，默认为10分
    importance: float = 0.  # 重要性，默认为0分
    decay_rate: float = 0.99
    pointers: List = []  # 作为反思时指向的记忆流
    date_prefix: str = "现在是"  # 时间前缀
    loc_prefix: str = "现在在"  # 地点前缀

    def get_recency(self, now: MyDateTime):
        """
        获取时近性: 自上次检索记忆以来根据沙盒游戏内小时数呈指数衰减的函数，衰减因子为 0.99
        """
        interval_seconds = now - self.view_time
        hours_passed = interval_seconds / 3600
        # self.recency *= (self.decay_factor ** hour)
        recency = self.decay_rate ** hours_passed
        return recency

    def update_recency(self, date: MyDateTime):
        self.view_time = date
        self.recency = 10.

    def get_importance(self):
        return self.importance

    def view(self):
        """
        查看记忆， 返回记忆内容，并更新查看时间
        TODO 返回哪些记忆内容
        """
        # 获取现在的时间
        date = ""
        self.view_time = date
        info = {}
        return info

    def __str__(self):
        return f"{self.date_prefix}{str(self.create_time)},{self.loc_prefix},{str(self.loc)}"


    def __getattr__(self, name):
        if name == 'x':
            self.x += 1
            return self.x

    # def get_score(self, max_recency, min_recency, max_imp, min_imp, alpha=1, beta=1):
    #     recency =


def get_time():
    pass


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)


class Retriever(BaseModel):
    vectorstore: Any  # 向量库
    search_kwargs: Dict = Field(default_factory=lambda: dict(k=100))  # 用于检索时的参数
    memory_stream: List[Observation]  # 记忆流存储位置
    decay_rate: float = 0.01  # 时近性衰减因子
    k: int = 4  # 搜索的最大文档数
    other_score_keys: List[Tuple] = []  # (key, weight)
    default_salience: Optional[float] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True  # 允许任意类型作为字段类型

    def _get_combined_score(self, observation: Observation,
                            relevance: Optional[float],
                            now: MyDateTime, ):
        """获取综合分数"""
        recency = observation.get_recency(now)
        score = recency
        for key, weight in self.other_score_keys:
            if key in observation:
                score += observation[key]
        if relevance is not None:
            score += relevance
        return score

    def get_salient_docs(self, query: str) -> Dict[int, Tuple[Observation, float]]:
        """获取重要的文档"""
        obv_and_scores: List[Tuple[Observation, float]]
        obv_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, **self.search_kwargs
        )
        results = {}
        for fetched_obv, relevance in obv_and_scores:
            if "buffer_idx" in fetched_obv:
                buffer_idx = fetched_obv["buffer_idx"]
                doc = self.memory_stream[buffer_idx]
                results[buffer_idx] = (doc, relevance)
        return results

    def get_relevant_memories(self, query: str) -> List[Observation]:
        """TODO 获取相关的文档"""
        now = get_time()
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self.k:]
        }
        # If a doc is considered salient, update the salience score
        docs_and_scores.update(self.get_salient_docs(query))
        rescored_docs = [
            (obv, self._get_combined_score(obv, relevance, now))
            for obv, relevance in docs_and_scores.values()
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        # Ensure frequently accessed memories aren't forgotten
        for doc, _ in rescored_docs[: self.k]:
            # TODO: Update vector store doc once `update` method is exposed.
            buffered_doc = self.memory_stream[doc.metadata["buffer_idx"]]
            buffered_doc.metadata["view_time"] = now
            result.append(buffered_doc)
        return result

    def add_observations(self, observations: List[Observation], **kwargs: Any) -> List[str]:
        """保存记忆"""
        now = kwargs.get("now")
        if now is None:
            now = get_time()
        # 防止改变数据
        dup_obvs = [deepcopy(obv) for obv in observations]
        for i, obv in enumerate(dup_obvs):
            obv["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_obvs)
        return self.vectorstore.add_documents(dup_obvs, **kwargs)

    def __getitem__(self, index):
        return self.memory_stream[index]
