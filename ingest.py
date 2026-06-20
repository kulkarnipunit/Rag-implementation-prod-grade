#!/usr/bin/env python3
"""Standalone script to ingest documents into the vector store."""
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_chunks
from src.retrieval.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/sample_docs",
        help="Directory containing documents to ingest (default: data/sample_docs)",
    )
    parser.add_argument("--reset", action="store_true", help="Clear the index before ingesting")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    store = VectorStore()

    if args.reset:
        print("Clearing existing index...")
        store.reset()

    print(f"Loading documents from: {data_dir}")
    raw_docs = load_documents(data_dir)
    if not raw_docs:
        print("No supported documents found (PDF, TXT, MD).")
        sys.exit(1)
    print(f"  Loaded {len(raw_docs)} document pages")

    print("Chunking documents...")
    chunks = chunk_documents(raw_docs)
    print(f"  Created {len(chunks)} chunks")

    print("Generating embeddings (this may take a moment on first run)...")
    embeddings = embed_chunks(chunks)
    print(f"  Generated {len(embeddings)} embeddings")

    print("Indexing into ChromaDB...")
    store.add_chunks(chunks, embeddings)
    print(f"  Total documents in index: {store.count()}")

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
