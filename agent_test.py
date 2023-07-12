from blocks.vectorstores.my_faiss import FAISSVectorStore
from blocks.embedding.embedding_ch import Text2VecBase
from blocks.time_block import MyClock
from agents.agent import Agent

my_clock = MyClock("2023-6-20 08:00:00")

# 获取词向量的函数
model_path = r"C:\Users\maiyuan\Desktop\开源模型\shibing624-text2vec-base-chinese".replace("\\", "/")
text_vector = Text2VecBase.init_model(model_path)
# 实例化faiss
faiss_store = FAISSVectorStore(embed_dim=768, embed_function=text_vector)

zs = {"name": "张三",
      "age": 30,
      "traits": "勤劳踏实，努力上进",
      "status": "早上刚刚醒来",
      "profession": ["煎饼摊摊主"],
      "clock": my_clock,
      "loc": "家",
      "faiss_store": faiss_store,
      "memories": []
      }

ls = {"name": "李四",
      "age": 28,
      "traits": "勤劳踏实，努力上进",
      "status": "早上刚刚醒来",
      "profession": ["白领"]}

zhang_san = Agent(**zs)
zhang_san.start()

print()
