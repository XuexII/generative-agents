import math
import os
from uuid import uuid1

import faiss
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental.generative_agents import GenerativeAgentMemory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from blocks.location_block import my_map
from langchain_agents.langchain_agent import LangChainAgent
import logging
import threading
import time

# log_path = "./log.txt"
# logging.basicConfig(format='%(asctime)s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     handlers=[logging.FileHandler(log_path, encoding='utf-8')],
#                     level="INFO")

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

USER_NAME = "Person A"  # The name you want to use when interviewing the agent.
LLM = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"],
                 max_tokens=1500)  # Can be any LLM you want.
"""
1. 不用添加搜索历史
2. 路径死循环的问题
3. 地点解析的问题
4. 解析计划的问题
"""


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    #:the distance / similarity metric used by the VectorStore
    #:the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)


maps = [
    {
        "name": "Tommie's house",
        "type": 1,
        "coord": None,
        "layout": [
            {"name": "sofa"}
        ],
        "subspace": [
            {
                "name": "Tommie's bedroom",
                "type": "1",
                "coord": None,
                "subspace": []
            },
            {
                "name": "Tommie's kitchen",
                "type": "1",
                "coord": None,
                "subspace": []
            },
            {
                "name": "Amy's bedroom",
                "type": "1",
                "coord": None,
                "subspace": []
            },
            {
                "name": "Tommie's bathroom",
                "type": "1",
                "coord": None,
                "subspace": []
            }
        ]
    },
    {
        "name": "John's house",
        "type": "1",
        "coord": None,
        "subspace": [
            {
                "name": "John's bedroom",
                "type": "1",
                "coord": None,
                "subspace": []
            },
            {
                "name": "John's kitchen",
                "type": "1",
                "coord": None,
                "subspace": []
            },
            {
                "name": "John's bathroom",
                "type": "1",
                "coord": None,
                "subspace": []
            }
        ]
    },
    {
        "name": "CBD",
        "type": "1",
        "coord": None,
        "desc": "office building",
        "subspace": [
            {
                "name": "SpaceX Company",
                "type": "1",
                "coord": None,
                "subspace": []
            },
            {
                "name": "FaceBook Company",
                "type": "1",
                "coord": None,
                "subspace": []
            }
        ]
    },
    {
        "name": "Phandalin Town Square",
        "type": "1",
        "coord": None,
        "desc": "Leisure Square",
        "subspace": [
        ]
    },
    {
        "name": "Barthen's Provisions",
        "type": "1",
        "coord": None,
        "desc": "the biggest trading post",
        "subspace": [

        ]
    },
    {
        "name": "Edermath Hospital",
        "type": "1",
        "coord": None,
        "desc": "the Hospital",
        "subspace": [

        ]
    }
]

my_map.init_map(maps)

# 1. 生成角色
tommies_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8  # we will give this a relatively low number to show how reflection works
)

tommie = LangChainAgent(name="Tommie",
                        age=25,
                        traits="anxious, likes design, talkative",  # You can add more persistent traits here
                        status="Tired of work, ready to take a break",  # When connected to a virtual world, we can have the characters update their status
                        memory_retriever=create_new_memory_retriever(),
                        llm=LLM,
                        memory=tommies_memory,
                        id=str(uuid1()),
                        loc="SpaceX Company",
                        known_areas=["Tommie's house", "CBD",
                                     "Barthen's Provisions",
                                     "Edermath Hospital",
                                     "Phandalin Town Square"]
                        )

# 1.1 添加记忆
tommie_observations = [
    "He is a programmer who works in the SpaceX Company of the CBD."
    " The working hours are from 9:30 am to 6:30 pm",
    "Jhon is Tommie's best friend",
    "Jhon's wife is sick and hospitalized",
    "The company is very busy recently, often need to work overtime until 21:00",
    "He is at work now"
]

# for observation in tommie_observations:
#     tommie.memory.add_memory(observation)
tommie.init_agent("./", tommie_observations)

john = LangChainAgent(name="John",
                      age=24,
                      traits="wise, resourceful, humorous",  # You can add more persistent traits here
                      status="Greet Tommie warmly",  # When connected to a virtual world, we can have the characters update their status
                      memory_retriever=create_new_memory_retriever(),
                      llm=LLM,
                      memory=tommies_memory,
                      id=str(uuid1()),
                      loc="SpaceX Company",
                      known_areas=["John's house", "CBD",
                                   "Barthen's Provisions",
                                   "Edermath Hospital",
                                   "Phandalin Town Square"]
                      )
john_observations = [
    "He is a programmer who works in the SpaceX Company of the CBD.",
    "Tommie is Jhon's best friend",
    "His wife was released from the hospital today",
    "The company is very busy recently, often need to work overtime until 21:00",
    "He is at work now"
]

# for observation in john_observations:
#     john.memory.add_memory(observation)

john.init_agent("./", john_observations)

my_map.add_guest([tommie, john])


def my_excepthook(args):
    print(f'多线程报错 {args.thread}: {args.exc_type.__name__}: {args.exc_value}')


threading.excepthook = my_excepthook
tommie.start()

# for agent in [tommie, john]:
#     thread = threading.Thread(target=agent.start, daemon=True)
#     thread.start()
#     # thread.join()
#
# while True:
#     time.sleep(1)
