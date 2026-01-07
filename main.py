import os
import json
from datetime import datetime
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from agents.read_pdf import read_pdf
from agents.planner import plan_report
from agents.writer import write_report
from agents.interpretaion_and_analysis import interpret_and_analyze


# -----------------------------
# CONFIG
# -----------------------------
PDF_PATH = "data/input.pdf"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# GRAPH STATE
# -----------------------------
class GraphState(TypedDict):
    knowledge_chunks: List[Dict]
    plan: str
    report: str
    analysis: str


# -----------------------------
# NODE FUNCTIONS
# -----------------------------
def pdf_node(state: GraphState) -> GraphState:
    print("📄 [PDF_AGENT] Extracting legal knowledge...")
    knowledge_chunks = read_pdf(PDF_PATH)

    with open(f"{OUTPUT_DIR}/knowledge_chunks.json", "w") as f:
        json.dump(knowledge_chunks, f, indent=2)

    return {
        "knowledge_chunks": knowledge_chunks
    }


def planner_node(state: GraphState) -> GraphState:
    print("🧠 [PLANNER_AGENT] Planning report structure...")
    plan = plan_report(state["knowledge_chunks"])

    with open(f"{OUTPUT_DIR}/plan.txt", "w") as f:
        f.write(plan)

    return {
        "plan": plan
    }


def writer_node(state: GraphState) -> GraphState:
    print("✍️ [WRITER_AGENT] Writing grounded report...")
    report = write_report(
        state["plan"],
        state["knowledge_chunks"]
    )

    with open(f"{OUTPUT_DIR}/report.md", "w") as f:
        f.write(report)

    return {
        "report": report
    }


def analysis_node(state: GraphState) -> GraphState:
    print("🔍 [ANALYSIS_AGENT] Running interpretability + critique...")
    analysis = interpret_and_analyze(
        state["report"],
        state["knowledge_chunks"]
    )

    with open(f"{OUTPUT_DIR}/analysis.json", "w") as f:
        f.write(analysis)

    return {
        "analysis": analysis
    }


# -----------------------------
# BUILD LANGGRAPH
# -----------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("pdf", pdf_node)
    graph.add_node("planner", planner_node)
    graph.add_node("writer", writer_node)
    graph.add_node("analysis", analysis_node)

    graph.set_entry_point("pdf")

    graph.add_edge("pdf", "planner")
    graph.add_edge("planner", "writer")
    graph.add_edge("writer", "analysis")
    graph.add_edge("analysis", END)

    return graph.compile()


# -----------------------------
# RUN
# -----------------------------
def main():
    print("\n🚀 Starting LangGraph-based Rental Agreement Analysis\n")

    app = build_graph()
    final_state = app.invoke({})

    metadata = {
        "model": "olmo:7b (ollama)",
        "document_type": "rental_agreement",
        "timestamp": datetime.utcnow().isoformat(),
        "num_pages": len(
            set(k["page"] for k in final_state["knowledge_chunks"])
        ),
        "num_clauses": len(final_state["knowledge_chunks"]),
        "agents": [
            "pdf_agent",
            "planner_agent",
            "writer_agent",
            "analysis_agent"
        ]
    }

    with open(f"{OUTPUT_DIR}/run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n🎉 LangGraph pipeline completed successfully!")
    print(f"📁 Outputs available in `{OUTPUT_DIR}/`\n")


if __name__ == "__main__":
    main()