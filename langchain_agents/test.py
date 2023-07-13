from datetime import datetime, timedelta
from typing import List
from termcolor import colored

from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
import math
import faiss
import os
from langchain.prompts import PromptTemplate

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["OPENAI_API_KEY"] = ""

USER_NAME = "Person A"  # The name you want to use when interviewing the agent.
LLM = ChatOpenAI(openai_api_key="",
                 max_tokens=1500)  # Can be any LLM you want.


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


# 1. 生成角色
tommies_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8  # we will give this a relatively low number to show how reflection works
)

tommie = GenerativeAgent(name="Tommie",
                         age=25,
                         traits="anxious, likes design, talkative",  # You can add more persistent traits here
                         status="looking for a job",  # When connected to a virtual world, we can have the characters update their status
                         memory_retriever=create_new_memory_retriever(),
                         llm=LLM,
                         memory=tommies_memory
                         )

# 1.1 添加记忆
tommie_observations = [
    "Tommie remembers his dog, Bruno, from when he was a kid",
    "Tommie feels tired from driving so far",
    "Tommie sees the new home",
    "The new neighbors have a cat",
    "The road is noisy at night",
    "Tommie is hungry",
    "Tommie tries to get some rest.",
]
for observation in tommie_observations:
    tommie.memory.add_memory(observation)


# 1.2 采访角色
def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]

obv = "Tommie wakes up to the sound of a noisy construction site outside his window."

_, reaction = tommie.generate_reaction(obv)


weekday_map = {"1": "Monday", "2": "Tuesday", "3": "Wednesday",
               "4": "Thursday", "5": "Friday", "6": "Saturday", "7": "Sunday"}


def _get_yesterday(now=None, offset: int = 0):
    now = now if now else datetime.now()
    now = now + timedelta(offset)
    _, week, weekday = now.isocalendar()
    date_str = now.strftime("%B %d,%H:%M")
    date, hour = date_str.split(",")
    return date, hour, weekday_map.get(str(weekday), "Monday")


now = datetime.now()
summary_description = tommie.get_summary(now=now)

schedule = ['7:00:Wake up, wash up',
            '7:30:Morning exercise',
            '8:00:Breakfast',
            '9:00:Work or personal projects',
            '12:00:Lunch',
            '13:00:Rest, relaxation',
            '14:00:Continue work or personal projects',
            '18:00:Finish work, free time',
            '19:00:Dinner',
            '20:00:Leisure activities (reading, watching TV, etc.)',
            '22:00:Prepare for bed',
            '23:00:Sleep']
schedule_str = ";".join(schedule)
prompt = PromptTemplate.from_template(
    "{name}'s schedule for yesterday:\n"
    "{schedule_str}\n\n"
    "Generate 3-5 brief summaries of {name}'s schedule for yesterday (output format: Time Range: Main Event):"
)
summaries = tommie.chain(prompt).run(name=tommie.name, schedule_str=schedule_str).strip()

yestd, hour, yestw = _get_yesterday(offset=-1)

schedule_summary = f"On {yestw} {yestd}, {tommie.name}" + summaries

# 2. 生成粗略规划
date, hour, weekd = _get_yesterday(now=now)

prompt = PromptTemplate.from_template(
    "{summary_description}\n"
    "{schedule_summary}\n"
    "plan example format:"
    "[7:14-7:45]:  Wake up and complete the morining routine\n"
    "[7:45-8:35]: Eat breakfirst\n"
    "[8:35-17:10]: Go to school and study\n"
    "[17:10-22:30]: Play CSGO\n"
    "[22:30-7:30]: Go to sleep\n\n"
    "Today is {weekd} {date}. Here is {name}’s plan today in broad strokes from {hour} today:"
)

plans = tommie.chain(prompt).run(summary_description=summary_description,
                                 schedule_summary=schedule_summary,
                                 weekd=weekd,
                                 date=date,
                                 name=tommie.name,
                                 hour=hour).strip()

print()
