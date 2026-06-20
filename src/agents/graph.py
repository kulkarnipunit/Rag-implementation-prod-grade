"""LangGraph stateful workflow for the research RAG agent.

Graph topology:
  START
    └─► route_query
          ├─► [vectorstore] retrieve_documents
          │       └─► grade_documents
          │               ├─► [has context] generate_with_context ─► format_report ─► END
          │               └─► [no context, retry < max] retrieve_documents  (loop)
          │               └─► [no context, retry >= max] generate_direct ─► format_report ─► END
          └─► [direct_answer] generate_direct ─► format_report ─► END
"""
from functools import partial
from typing import Any

from langgraph.graph import StateGraph, END

from .state import ResearchState
from .nodes import (
    route_query,
    retrieve_documents,
    grade_documents,
    generate_with_context,
    generate_direct,
    format_report,
    MAX_RETRIES,
)


def _decide_after_route(state: ResearchState) -> str:
    return state["route"]


def _decide_after_grade(state: ResearchState) -> str:
    if state.get("no_context_fallback") and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry_retrieve"
    if state.get("no_context_fallback"):
        return "direct_answer"
    return "generate"


def build_graph(retriever: Any) -> Any:
    """Construct and compile the LangGraph research agent."""

    # Bind the retriever into the retrieve node
    retrieve_fn = partial(retrieve_documents, retriever=retriever)

    workflow = StateGraph(ResearchState)

    # Register nodes
    workflow.add_node("route_query", route_query)
    workflow.add_node("retrieve_documents", retrieve_fn)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_with_context", generate_with_context)
    workflow.add_node("generate_direct", generate_direct)
    workflow.add_node("format_report", format_report)

    # Entry point
    workflow.set_entry_point("route_query")

    # Route decision
    workflow.add_conditional_edges(
        "route_query",
        _decide_after_route,
        {
            "vectorstore": "retrieve_documents",
            "direct_answer": "generate_direct",
        },
    )

    # Retrieval → grading
    workflow.add_edge("retrieve_documents", "grade_documents")

    # Grading decision: retry, fallback to direct, or generate
    workflow.add_conditional_edges(
        "grade_documents",
        _decide_after_grade,
        {
            "retry_retrieve": "retrieve_documents",
            "direct_answer": "generate_direct",
            "generate": "generate_with_context",
        },
    )

    # Generation → report
    workflow.add_edge("generate_with_context", "format_report")
    workflow.add_edge("generate_direct", "format_report")
    workflow.add_edge("format_report", END)

    return workflow.compile()


def run_research_agent(
    query: str,
    topic: str,
    retriever: Any,
) -> ResearchState:
    """Run the full research agent pipeline and return the final state."""
    graph = build_graph(retriever)

    initial_state: ResearchState = {
        "query": query,
        "topic": topic,
        "route": "vectorstore",
        "retrieved_docs": [],
        "graded_docs": [],
        "generation": "",
        "citations": [],
        "report": "",
        "retry_count": 0,
        "no_context_fallback": False,
    }

    final_state = graph.invoke(initial_state)
    return final_state
