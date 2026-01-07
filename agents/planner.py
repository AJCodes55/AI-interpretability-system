from langchain_community.llms import Ollama
from typing import List

LLM_MODEL = "olmo-3:7b"
TEMPERATURE = 0.2

llm = Ollama(
    model = LLM_MODEL,
    temperature = TEMPERATURE
)

def plan_report(knowledge_chunks: List[dict]) -> str:
     # prompt to train your agent to write a report about the rental agreement

     prompt = f""" you are a planner for a rental agreement. you follow legal clauses.
     you will be given exrtacted knowledge about rental agreement and your job is to plan the report
     structure. The structure should include the following sections:

     - Introduction
     - Parties inolved
     - type of agreement(student, bachelors, married, corporate, etc.)
     - duration of agreement (monthly, yearly, etc.)
     - rent amount (monthly, yearly, etc.)
     - deposit amount (monthly, yearly, etc.)
     - rent due date
     - important clauses (security deposit, late rent, rent increase, etc.)
     - restrcitios on the tenant (no pets, no smoking, no parties, etc.)

     the input is as follows: 

     TEXT: {knowledge_chunks} """

     report_plan = llm.invoke(prompt)
     return report_plan

     