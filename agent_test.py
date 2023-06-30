from agents.agent import Agent
from blocks.location_block import MyMap
from agents.brain import Brain
from blocks.retriever import Observation, Retriever
from blocks.vectorstores.my_faiss import FAISSVectorStore
from blocks.embedding.embedding_ch import Text2VecBase
from blocks.time_block import MyClock

my_clock = MyClock("2023-6-20 08:00:00")

# 获取词向量的函数
model_path = r"C:\Users\maiyuan\Desktop\开源模型\shibing624-text2vec-base-chinese".replace("\\", "/")
text_vector = Text2VecBase(model_path)
# 实例化faiss
faiss_store = FAISSVectorStore(embed_dim=768, embed_function=text_vector)
# 实例化记忆库
retriever = Retriever(vector_store=faiss_store)
brain = Brain(memories=retriever)

info = {"name": "张三",
        "age": 30,
        "traits": "勤劳踏实，努力上进",
        "status": "早上刚刚醒来",
        "profession": ["煎饼摊摊主"],
        "brain": brain}

zhang_san = Agent(**info)

print()

