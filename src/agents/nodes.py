"""LangGraph node functions for the research RAG agent."""
import os
import json
from typing import Any
from pprint import pprint

from ..llm.client import chat as _chat
from .state import ResearchState


MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


def _debug(node_name: str, label: str, data: Any):
    print(f"\n{'='*60}")
    print(f"  [{node_name}] {label}")
    print(f"{'='*60}")
    pprint(data, width=80, depth=3)
    print()


# ---------------------------------------------------------------------------
# Node: route_query
# ---------------------------------------------------------------------------
def route_query(state: ResearchState) -> ResearchState:
    """Decide whether to retrieve from the vector store or answer directly."""
    _debug("route_query", "STATE IN", {k: v for k, v in state.items() if k in ("query", "topic", "route", "retry_count")})

    query = state["query"]

    prompt = f"""You are a document routing assistant. Your default is ALWAYS to use the vectorstore.

Only choose "direct_answer" if the query is CLEARLY a general knowledge question with NO connection to personal data, resumes, companies, documents, or any specific entity.

Examples of direct_answer: "What is the capital of France?", "Explain what machine learning is"
Examples of vectorstore: EVERYTHING ELSE — including questions about people, companies, work experience, skills, documents, reports, data

Query: {query}

Respond ONLY with one of these JSON objects — no other text:
{{"route": "vectorstore"}}
{{"route": "direct_answer"}}"""

    text = _chat(prompt, max_tokens=64)
    try:
        decision = json.loads(text.strip())
        route = decision.get("route", "vectorstore")
    except (json.JSONDecodeError, KeyError):
        route = "vectorstore"

    new_state = {**state, "route": route, "retry_count": 0}
    _debug("route_query", "STATE OUT — route decided", {"route": new_state["route"], "retry_count": new_state["retry_count"]})
    return new_state


# ---------------------------------------------------------------------------
# Node: retrieve_documents
# ---------------------------------------------------------------------------
def retrieve_documents(state: ResearchState, retriever: Any) -> ResearchState:
    """Retrieve relevant document chunks from the vector store."""
    _debug("retrieve_documents", "STATE IN", {"query": state["query"], "retry_count": state.get("retry_count", 0)})

    query = state["query"]
    docs = retriever.retrieve(query)

    _debug("retrieve_documents", "STATE OUT — chunks fetched", {
        "total_retrieved": len(docs),
        "top_3_chunks": [
            {"filename": d.get("filename"), "page": d.get("page"), "score": round(d.get("relevance_score", 0), 3), "preview": d["content"][:80]}
            for d in docs[:3]
        ],
    })
    return {**state, "retrieved_docs": docs}


# ---------------------------------------------------------------------------
# Node: grade_documents
# ---------------------------------------------------------------------------
def grade_documents(state: ResearchState) -> ResearchState:
    """Filter retrieved chunks — keep only those relevant to the query."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])

    _debug("grade_documents", "STATE IN", {
        "query": query,
        "docs_to_grade": len(docs),
    })

    graded = []
    for i, doc in enumerate(docs):
        prompt = f"""You are a relevance grader. Be GENEROUS — if the document chunk contains ANY information that could help answer the query, mark it as relevant.

Query: {query}
Document: {doc['content'][:500]}

Respond ONLY with JSON: {{"relevant": true}} or {{"relevant": false}}
When in doubt, respond {{"relevant": true}}"""

        text = _chat(prompt, max_tokens=32)
        try:
            result = json.loads(text.strip())
            relevant = result.get("relevant", True)
        except (json.JSONDecodeError, KeyError):
            relevant = True

        print(f"  grading doc {i+1}/{len(docs)} → relevant={relevant} | {doc.get('filename','?')} p.{doc.get('page','?')} | \"{doc['content'][:60]}...\"")
        if relevant:
            graded.append(doc)

    no_context = len(graded) == 0
    retry_count = state.get("retry_count", 0)

    _debug("grade_documents", "STATE OUT", {
        "docs_in":            len(docs),
        "docs_kept":          len(graded),
        "docs_dropped":       len(docs) - len(graded),
        "no_context_fallback": no_context,
        "retry_count":        retry_count + (1 if no_context else 0),
    })

    return {
        **state,
        "graded_docs": graded,
        "no_context_fallback": no_context,
        "retry_count": retry_count + (1 if no_context else 0),
    }


# ---------------------------------------------------------------------------
# Node: generate_with_context
# ---------------------------------------------------------------------------
def generate_with_context(state: ResearchState) -> ResearchState:
    """Generate a comprehensive answer grounded in retrieved document context."""
    docs = state.get("graded_docs") or state.get("retrieved_docs", [])

    _debug("generate_with_context", "STATE IN", {
        "query":       state["query"],
        "graded_docs": len(docs),
        "sources":     [f"{d.get('filename','?')} p.{d.get('page','?')}" for d in docs],
    })

    query = state["query"]
    context_blocks = []
    citations = []
    for i, doc in enumerate(docs, 1):
        label = f"[{i}] {doc.get('filename', doc.get('source', 'Unknown'))}, p.{doc.get('page', '?')}"
        context_blocks.append(f"Source {label}:\n{doc['content']}")
        citations.append(label)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are an expert research analyst. You MUST answer using ONLY the information from the source documents below.
Do NOT use any outside knowledge or make up any information.
If a fact is not in the documents, do not include it.
Every factual claim must cite its source number, e.g. [1], [2].

Research Query: {query}

Source Documents:
{context}

Write a detailed answer using ONLY the information above. Do not invent or assume anything not stated in the sources:"""

    full_response = _chat(prompt)

    _debug("generate_with_context", "STATE OUT", {
        "citations":         citations,
        "generation_preview": full_response[:200],
    })
    return {**state, "generation": full_response, "citations": citations}


# ---------------------------------------------------------------------------
# Node: generate_direct
# ---------------------------------------------------------------------------
def generate_direct(state: ResearchState) -> ResearchState:
    """Answer directly from the model's knowledge when no retrieval is needed."""
    _debug("generate_direct", "STATE IN — no docs found, answering from model knowledge", {
        "query":       state["query"],
        "retry_count": state.get("retry_count", 0),
    })

    query = state["query"]

    prompt = f"""You are an expert research analyst.
Answer the following question using your expert knowledge.
Be detailed, accurate, and structured.

Question: {query}"""

    text = _chat(prompt)

    _debug("generate_direct", "STATE OUT", {"generation_preview": text[:200]})
    return {**state, "generation": text, "citations": [], "graded_docs": []}


# ---------------------------------------------------------------------------
# Node: format_report
# ---------------------------------------------------------------------------
def format_report(state: ResearchState) -> ResearchState:
    """
    Format the generation into a final report using a template — no second LLM call.
    A second LLM pass over the generation (without access to source docs) is the
    primary source of hallucinated summaries and self-contradictions.
    """
    _debug("format_report", "STATE IN", {
        "topic":              state.get("topic", state["query"]),
        "citations":          state.get("citations", []),
        "generation_preview": state.get("generation", "")[:150],
    })

    topic = state.get("topic", state["query"])
    generation = state.get("generation", "")
    citations = state.get("citations", [])
    graded_docs = state.get("graded_docs", [])

    if citations:
        sources_lines = "\n".join(f"- {c}" for c in citations)
    elif graded_docs:
        sources_lines = "\n".join(
            f"- [{i+1}] {d.get('filename', d.get('source', 'Unknown'))}, p.{d.get('page', '?')}"
            for i, d in enumerate(graded_docs)
        )
    else:
        sources_lines = "_General knowledge — no documents retrieved._"

    report = (
        f"# {topic}\n\n"
        f"{generation}\n\n"
        f"---\n\n"
        f"**Sources**\n\n{sources_lines}\n"
    )

    _debug("format_report", "STATE OUT — FINAL REPORT READY", {
        "report_length_chars": len(report),
        "report_preview":      report[:300],
    })
    return {**state, "report": report}
