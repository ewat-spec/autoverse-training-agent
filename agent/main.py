# agent/main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import DuckDuckGoSearchRun
import os

app = FastAPI()

llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
search = DuckDuckGoSearchRun()
tools = [Tool(name="Search", func=search.run, description="Search the web")]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(query: Query):
    response = agent.run(query.question)
    return {"response": response}

