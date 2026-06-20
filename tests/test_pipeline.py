"""Integration tests for the RAG pipeline components."""
import os
import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------
class TestLoader:
    def test_load_txt_file(self, tmp_path):
        from src.ingestion.loader import load_documents
        doc_file = tmp_path / "test.txt"
        doc_file.write_text("This is a test document about AI systems.")
        docs = load_documents(tmp_path)
        assert len(docs) == 1
        assert "test document" in docs[0].content
        assert docs[0].source == str(doc_file)

    def test_load_empty_file(self, tmp_path):
        from src.ingestion.loader import load_documents
        (tmp_path / "empty.txt").write_text("")
        docs = load_documents(tmp_path)
        assert len(docs) == 0

    def test_load_multiple_files(self, tmp_path):
        from src.ingestion.loader import load_documents
        (tmp_path / "a.txt").write_text("Document A content.")
        (tmp_path / "b.md").write_text("Document B content.")
        docs = load_documents(tmp_path)
        assert len(docs) == 2


class TestChunker:
    def test_short_text_no_split(self):
        from src.ingestion.loader import RawDocument
        from src.ingestion.chunker import chunk_documents
        doc = RawDocument(content="Short text.", source="test.txt", page=1)
        chunks = chunk_documents([doc], chunk_size=800)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_long_text_splits(self):
        from src.ingestion.loader import RawDocument
        from src.ingestion.chunker import chunk_documents
        long_text = "This is a sentence. " * 100
        doc = RawDocument(content=long_text, source="test.txt", page=1)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_metadata(self):
        from src.ingestion.loader import RawDocument
        from src.ingestion.chunker import chunk_documents
        doc = RawDocument(content="Hello world.", source="file.txt", page=3, metadata={"filename": "file.txt"})
        chunks = chunk_documents([doc])
        assert chunks[0].source == "file.txt"
        assert chunks[0].page == 3
        assert chunks[0].chunk_index == 0


# ---------------------------------------------------------------------------
# Vector store tests
# ---------------------------------------------------------------------------
class TestVectorStore:
    def test_add_and_query(self, tmp_path):
        from src.ingestion.loader import RawDocument
        from src.ingestion.chunker import chunk_documents
        from src.retrieval.vectorstore import VectorStore
        from src.ingestion.embedder import embed_chunks, embed_query

        doc = RawDocument(
            content="Enterprise RAG systems improve document retrieval accuracy by 60%.",
            source="test.txt",
            page=1,
            metadata={"filename": "test.txt", "type": "txt"},
        )
        chunks = chunk_documents([doc])
        embeddings = embed_chunks(chunks)

        store = VectorStore(persist_dir=str(tmp_path / "chroma"))
        store.add_chunks(chunks, embeddings)
        assert store.count() == len(chunks)

        query_vec = embed_query("RAG retrieval accuracy")
        results = store.query(query_vec, top_k=1)
        assert len(results) == 1
        assert "RAG" in results[0]["content"]
        assert results[0]["relevance_score"] > 0.0

    def test_reset_clears_store(self, tmp_path):
        from src.ingestion.loader import RawDocument
        from src.ingestion.chunker import chunk_documents
        from src.retrieval.vectorstore import VectorStore
        from src.ingestion.embedder import embed_chunks

        doc = RawDocument(content="Some content here.", source="x.txt", page=1, metadata={"type": "txt", "filename": "x.txt"})
        chunks = chunk_documents([doc])
        embeddings = embed_chunks(chunks)

        store = VectorStore(persist_dir=str(tmp_path / "chroma2"))
        store.add_chunks(chunks, embeddings)
        assert store.count() > 0

        store.reset()
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Agent state tests
# ---------------------------------------------------------------------------
class TestAgentState:
    def test_state_schema(self):
        from src.agents.state import ResearchState
        state: ResearchState = {
            "query": "What is RAG?",
            "topic": "AI Systems",
            "route": "vectorstore",
            "retrieved_docs": [],
            "graded_docs": [],
            "generation": "",
            "citations": [],
            "report": "",
            "retry_count": 0,
            "no_context_fallback": False,
        }
        assert state["route"] == "vectorstore"
        assert state["retry_count"] == 0


# ---------------------------------------------------------------------------
# Run with: python -m pytest tests/ -v
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
