"""LangGraph node functions for the research RAG agent."""
import os
import json
from typing import Any

import ollama

from .state import ResearchState


MODEL = os.getenv("OLLAMA_MODEL", "llama3")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


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
        prompt = f"""You are a relevance grader. Be GENEROUS — if the document chunk contains ANY information that could help answer the query, mark it as relevant.

Query: {query}
Document: {doc['content'][:500]}

Respond ONLY with JSON: {{"relevant": true}} or {{"relevant": false}}
When in doubt, respond {{"relevant": true}}"""

        text = _chat(prompt, max_tokens=32)
        try:
            result = json.loads(text.strip())
            if result.get("relevant", True):
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

    prompt = f"""You are an expert research analyst. You MUST answer using ONLY the information from the source documents below.
Do NOT use any outside knowledge or make up any information.
If a fact is not in the documents, do not include it.
Every factual claim must cite its source number, e.g. [1], [2].

Research Query: {query}

Source Documents:
{context}

Write a detailed answer using ONLY the information above. Do not invent or assume anything not stated in the sources:"""

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
