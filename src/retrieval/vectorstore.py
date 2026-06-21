import os
from typing import List, Optional
from pathlib import Path

from ..ingestion.chunker import Chunk

# ChromaDB only accepts these scalar types as metadata values
_CHROMA_SAFE = (str, int, float, bool)


class VectorStore:
    """ChromaDB-backed vector store with sentence-transformer embeddings."""

    COLLECTION_NAME = "research_docs"

    def __init__(self, persist_dir: Optional[str] = None):
        import chromadb
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Upsert chunks and their embeddings into the collection."""
        ids = [f"{c.source}::p{c.page}::c{c.chunk_index}" for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [_safe_metadata(c) for c in chunks]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[dict]:
        """Return top-k most similar chunks with metadata and distances."""
        top_k = min(top_k, self._collection.count())
        if top_k == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks_out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks_out.append({
                "content": doc,
                "source": meta.get("source", ""),
                "filename": meta.get("filename", ""),
                "page": meta.get("page", 0),
                "content_type": meta.get("content_type", "prose"),
                "relevance_score": 1.0 - dist,
                # pass through all structured fields (city, state, hospital…)
                **{k: v for k, v in meta.items() if k not in {"source", "filename", "page", "content_type"}},
            })
        return chunks_out

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def _safe_metadata(chunk: Chunk) -> dict:
    """Build a ChromaDB-safe metadata dict from a Chunk, preserving all structured fields."""
    base = {
        "source": chunk.source,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "filename": chunk.metadata.get("filename", ""),
        "type": chunk.metadata.get("type", ""),
        "content_type": chunk.metadata.get("content_type", "prose"),
    }
    # Forward any extra structured metadata from table rows (city, state, hospital, etc.)
    for key, val in chunk.metadata.items():
        if key not in base and isinstance(val, _CHROMA_SAFE):
            base[key] = val
    return base
