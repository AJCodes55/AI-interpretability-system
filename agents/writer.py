from langchain_community.llms import Ollama
from typing import List

from agents.planner import plan_report

LLM_MODEL = "olmo-3:7b"
TEMPERATURE = 0.2

llm = Ollama(
    model = LLM_MODEL,
    temperature = TEMPERATURE
)

def write_report(report_plan: str , knowledge_chunks: List[dict] ) -> str:
    # prompt to train your agent to write a report about the rental agreement

    prompt = f""" You are a professional legal writing agent.
    **INPUT**:
     1) You are given
      a) REPORT PLAN
      b) EXTRACTED KNOWLEDGE ABOUT RENTAL AGREEMENT

    **RULES**:
      1) Follow the report plan strictly.
      2) Do not hallucinate strictly from the extracted knowledge.
      3) Every section must have reference and citation from the extracted knowledge.
      4) The report must be in a professional and legal tone.

    **REPORT PLAN**:   
      PLAN = {report_plan}
      KNOWLEDGE = {knowledge_chunks} """

    report = llm.invoke(prompt)
    return report
    