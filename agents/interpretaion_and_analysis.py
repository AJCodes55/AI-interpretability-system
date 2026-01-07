from langchain_community.llms import Ollama
from typing import List

LLM_MODEL = "olmo-3:7b"
TEMPERATURE = 0.2

llm = Ollama(
    model = LLM_MODEL,
    temperature = TEMPERATURE
)

def interpret_and_analyze(report: str , knowledge_chunks: List[dict] ) -> str:
    # prompt to train your agent to interpret and analyze the report

    prompt = f""" You are a professional legal interpreter and analyst.

    **INPUT**:
        1) You are given a report and knowledge chunks about a rental agreement.
        2) You are given extracted knowledge about the rental agreement. Each clause has
        a) chunk ID(s)
        b) page number(s)
        c) knowledge
        
    **TASKS**:
        PART A — INTERPRETABILITY
            For EACH report section:
            - Identify which clause(s) support it
            - Provide clause ID(s) and page number(s)
            - Explain the relationship briefly
            - If grounding is weak or missing, state that explicitly

        PART B — CRITIQUE
            Based ONLY on the extracted clauses:
            - Identify tenant-unfriendly clauses
            - Identify ambiguous or vague clauses
            - Identify legal or financial risks
            - Mention page numbers for each issue

        The output should be in the following format:
            OUTPUT FORMAT (JSON ONLY):
{{
  "interpretability": [
    {{
      "report_section": "<section title>",
      "supported_by": [
        {{
          "chunk_id": "<id>",
          "page": <page>,
          "reason": "<why this clause supports the section>"
        }}
      ],
      "grounding_strength": "strong | partial | weak"
    }}
  ],
  "critique": {{
    "tenant_unfriendly": [
      {{
        "issue": "<description>",
        "page": <page>
      }}
    ],
    "ambiguities": [
      {{
        "issue": "<description>",
        "page": <page>
      }}
    ],
    "risks": [
      {{
        "issue": "<description>",
        "page": <page>
      }}
    ]
  }}
}}


        **REPORT**:
        REPORT: {report}
        KNOWLEDGE CHUNKS: {knowledge_chunks} """



    analysis = llm.invoke(prompt)
    return analysis




