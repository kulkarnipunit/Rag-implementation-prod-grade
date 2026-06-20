import os
from typing import List
from .chunker import Chunk


def get_embedding_model():
    """Lazy-load the sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def embed_chunks(chunks: List[Chunk], batch_size: int = 64) -> List[List[float]]:
    """Generate embeddings for a list of chunks."""
    model = get_embedding_model()
    texts = [c.content for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string for retrieval."""
    model = get_embedding_model()
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0].tolist()
