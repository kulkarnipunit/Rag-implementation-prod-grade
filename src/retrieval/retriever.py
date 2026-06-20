import os
from typing import List, Optional

from ..ingestion.embedder import embed_query
from .vectorstore import VectorStore


class DocumentRetriever:
    """Retrieves and re-ranks relevant document chunks for a given query."""

    def __init__(self, vectorstore: VectorStore):
        self._store = vectorstore
        self._top_k = int(os.getenv("TOP_K_RESULTS", "5"))
        self._threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Embed query, retrieve top-k chunks, filter by relevance threshold."""
        k = top_k or self._top_k
        query_vec = embed_query(query)
        results = self._store.query(query_vec, top_k=k * 2)  # over-fetch for re-ranking

        # Filter by relevance threshold
        filtered = [r for r in results if r["relevance_score"] >= self._threshold]

        # MMR-style deduplication: remove near-duplicate content
        deduplicated = _deduplicate(filtered, similarity_cutoff=0.95)

        return deduplicated[:k]


def _deduplicate(docs: List[dict], similarity_cutoff: float = 0.95) -> List[dict]:
    """Remove chunks whose content overlaps heavily with a higher-ranked chunk."""
    kept = []
    for doc in docs:
        if not _is_near_duplicate(doc["content"], [k["content"] for k in kept], similarity_cutoff):
            kept.append(doc)
    return kept


def _is_near_duplicate(text: str, existing: List[str], cutoff: float) -> bool:
    """Jaccard similarity check on word sets."""
    words = set(text.lower().split())
    for other in existing:
        other_words = set(other.lower().split())
        if not words or not other_words:
            continue
        intersection = len(words & other_words)
        union = len(words | other_words)
        if union > 0 and intersection / union >= cutoff:
            return True
    return False
