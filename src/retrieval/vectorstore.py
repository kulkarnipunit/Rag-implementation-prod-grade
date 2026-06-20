import os
from typing import List, Optional
from pathlib import Path

from ..ingestion.chunker import Chunk


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
        metadatas = [
            {
                "source": c.source,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "filename": c.metadata.get("filename", ""),
                "type": c.metadata.get("type", ""),
            }
            for c in chunks
        ]
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
                "relevance_score": 1.0 - dist,  # cosine distance → similarity
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
