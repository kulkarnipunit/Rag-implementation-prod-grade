from typing import List
from dataclasses import dataclass
from .loader import RawDocument


@dataclass
class Chunk:
    content: str
    source: str
    page: int
    chunk_index: int
    metadata: dict


def chunk_documents(
    docs: List[RawDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """Split documents into overlapping chunks for embedding."""
    chunks: List[Chunk] = []
    for doc in docs:
        doc_chunks = _split_text(doc.content, chunk_size, chunk_overlap)
        for i, text in enumerate(doc_chunks):
            chunks.append(Chunk(
                content=text,
                source=doc.source,
                page=doc.page,
                chunk_index=i,
                metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(doc_chunks)},
            ))
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-level sliding window split on sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to split at a sentence boundary
        boundary = _find_sentence_boundary(text, end)
        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)
        start = boundary - overlap
        if start >= len(text) - overlap:
            break

    return [c for c in chunks if c]


def _find_sentence_boundary(text: str, pos: int) -> int:
    """Walk backwards from pos to find the last sentence-ending punctuation."""
    for i in range(pos, max(pos - 200, 0), -1):
        if text[i] in ".!?\n":
            return i + 1
    return pos
