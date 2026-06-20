"""FastAPI enterprise endpoint for the research RAG agent."""
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from ..ingestion.loader import load_documents
from ..ingestion.chunker import chunk_documents
from ..ingestion.embedder import embed_chunks
from ..retrieval.vectorstore import VectorStore
from ..retrieval.retriever import DocumentRetriever
from ..agents.graph import run_research_agent

app = FastAPI(
    title="Enterprise RAG Research Agent",
    description="Production-grade document Q&A and research report generation using LangGraph + Claude",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(_static_dir / "index.html")

# Module-level singletons (initialized at startup)
_store: Optional[VectorStore] = None
_retriever: Optional[DocumentRetriever] = None


@app.on_event("startup")
def startup():
    global _store, _retriever
    _store = VectorStore()
    _retriever = DocumentRetriever(_store)


class IngestRequest(BaseModel):
    data_dir: str = Field(..., description="Directory containing documents to ingest")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Research question or query")
    topic: str = Field(default="", description="High-level research topic label")


class QueryResponse(BaseModel):
    query: str
    topic: str
    route: str
    report: str
    citations: List[str]
    sources_used: int


@app.get("/health")
def health():
    doc_count = _store.count() if _store else 0
    return {"status": "ok", "documents_indexed": doc_count}


@app.post("/ingest")
def ingest_documents(req: IngestRequest):
    """Ingest all documents from a directory into the vector store."""
    data_dir = Path(req.data_dir)
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {data_dir}")

    raw_docs = load_documents(data_dir)
    if not raw_docs:
        raise HTTPException(status_code=400, detail="No supported documents found in directory")

    chunks = chunk_documents(raw_docs)
    embeddings = embed_chunks(chunks)
    _store.add_chunks(chunks, embeddings)

    return {
        "status": "success",
        "documents_loaded": len(raw_docs),
        "chunks_indexed": len(chunks),
        "total_indexed": _store.count(),
    }


@app.post("/query", response_model=QueryResponse)
def query_agent(req: QueryRequest):
    """Run the full research agent pipeline and return a structured report."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    topic = req.topic or req.query
    final_state = run_research_agent(
        query=req.query,
        topic=topic,
        retriever=_retriever,
    )

    return QueryResponse(
        query=req.query,
        topic=topic,
        route=final_state.get("route", "unknown"),
        report=final_state.get("report", ""),
        citations=final_state.get("citations", []),
        sources_used=len(final_state.get("graded_docs", [])),
    )


@app.delete("/index")
def reset_index():
    """Clear the vector store index."""
    _store.reset()
    return {"status": "index cleared"}


@app.get("/chunks")
def get_chunks(limit: int = 100):
    """Return all stored chunks from ChromaDB for inspection."""
    if _store is None:
        raise HTTPException(status_code=503, detail="Store not initialized")
    collection = _store._collection
    total = collection.count()
    if total == 0:
        return {"total": 0, "chunks": []}
    results = collection.get(
        limit=min(limit, total),
        include=["documents", "metadatas"],
    )
    chunks = [
        {
            "index": i + 1,
            "filename": meta.get("filename", "unknown"),
            "page": meta.get("page", 0),
            "chunk_index": meta.get("chunk_index", i),
            "content": doc,
        }
        for i, (doc, meta) in enumerate(
            zip(results["documents"], results["metadatas"])
        )
    ]
    return {"total": total, "chunks": chunks}
