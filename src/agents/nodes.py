"""LangGraph node functions for the research RAG agent."""
import os
import json
from typing import Any

import ollama

from .state import ResearchState


MODEL = os.getenv("OLLAMA_MODEL", "llama3")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))


def _chat(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens},
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Node: route_query
# ---------------------------------------------------------------------------
def route_query(state: ResearchState) -> ResearchState:
    """Decide whether to retrieve from the vector store or answer directly."""
    query = state["query"]

    prompt = f"""You are a research assistant router.
Given a user query, decide if it requires retrieving information from a document knowledge base
or if it can be answered directly from general knowledge.

Query: {query}

Respond with a JSON object:
{{"route": "vectorstore"}} — if the query needs specific documents, facts, data, or research
{{"route": "direct_answer"}} — if the query is a general question answerable without documents

Respond ONLY with the JSON object, no other text."""

    text = _chat(prompt, max_tokens=64)
    try:
        decision = json.loads(text.strip())
        route = decision.get("route", "vectorstore")
    except (json.JSONDecodeError, KeyError):
        route = "vectorstore"

    return {**state, "route": route, "retry_count": 0}


# ---------------------------------------------------------------------------
# Node: retrieve_documents
# ---------------------------------------------------------------------------
def retrieve_documents(state: ResearchState, retriever: Any) -> ResearchState:
    """Retrieve relevant document chunks from the vector store."""
    query = state["query"]
    docs = retriever.retrieve(query)
    return {**state, "retrieved_docs": docs}


# ---------------------------------------------------------------------------
# Node: grade_documents
# ---------------------------------------------------------------------------
def grade_documents(state: ResearchState) -> ResearchState:
    """Filter retrieved chunks — keep only those relevant to the query."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])

    graded = []
    for doc in docs:
        prompt = f"""You are a relevance grader.
Assess whether the following document chunk is useful for answering the query.

Query: {query}
Document: {doc['content'][:500]}

Respond ONLY with JSON: {{"relevant": true}} or {{"relevant": false}}"""

        text = _chat(prompt, max_tokens=32)
        try:
            result = json.loads(text.strip())
            if result.get("relevant", False):
                graded.append(doc)
        except (json.JSONDecodeError, KeyError):
            graded.append(doc)

    no_context = len(graded) == 0
    retry_count = state.get("retry_count", 0)

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
    query = state["query"]
    docs = state.get("graded_docs") or state.get("retrieved_docs", [])

    context_blocks = []
    citations = []
    for i, doc in enumerate(docs, 1):
        label = f"[{i}] {doc.get('filename', doc.get('source', 'Unknown'))}, p.{doc.get('page', '?')}"
        context_blocks.append(f"Source {label}:\n{doc['content']}")
        citations.append(label)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are an expert research analyst. Using ONLY the provided source documents,
write a comprehensive, well-structured answer to the research query.

Every factual claim must be supported by citing the source number, e.g. [1], [2].
Be precise and analytical. If sources conflict, note the discrepancy.

Research Query: {query}

Source Documents:
{context}

Write a detailed, cited answer:"""

    full_response = _chat(prompt)
    return {**state, "generation": full_response, "citations": citations}


# ---------------------------------------------------------------------------
# Node: generate_direct
# ---------------------------------------------------------------------------
def generate_direct(state: ResearchState) -> ResearchState:
    """Answer directly from the model's knowledge when no retrieval is needed."""
    query = state["query"]

    prompt = f"""You are an expert research analyst.
Answer the following question using your expert knowledge.
Be detailed, accurate, and structured.

Question: {query}"""

    text = _chat(prompt)
    return {**state, "generation": text, "citations": [], "graded_docs": []}


# ---------------------------------------------------------------------------
# Node: format_report
# ---------------------------------------------------------------------------
def format_report(state: ResearchState) -> ResearchState:
    """Synthesize the generation and citations into a structured research report."""
    query = state["query"]
    topic = state.get("topic", query)
    generation = state.get("generation", "")
    citations = state.get("citations", [])
    graded_docs = state.get("graded_docs", [])

    sources_section = ""
    if citations:
        sources_section = "\n".join(f"- {c}" for c in citations)
    elif graded_docs:
        sources_section = "\n".join(
            f"- [{i+1}] {d.get('filename', d.get('source', 'Unknown'))}, p.{d.get('page', '?')}"
            for i, d in enumerate(graded_docs)
        )

    prompt = f"""You are a professional research report writer.
Transform the following raw analysis into a polished, structured research report in Markdown.

Structure the report with:
1. **Executive Summary** — 2-3 sentence overview of key findings
2. **Key Findings** — bullet points of the most important insights
3. **Detailed Analysis** — the full analysis, well-organized with subheadings
4. **Conclusions & Recommendations** — actionable takeaways
5. **Sources** — reference list (provided below)

Research Topic: {topic}
Original Query: {query}

Raw Analysis:
{generation}

Sources:
{sources_section if sources_section else "General knowledge (no documents retrieved)"}

Write the complete formatted research report in Markdown:"""

    report = _chat(prompt)
    return {**state, "report": report}
