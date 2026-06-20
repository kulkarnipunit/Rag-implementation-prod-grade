#!/usr/bin/env python3
"""
Enterprise RAG Research Agent — Main Entry Point

Demonstrates the full pipeline:
  1. Ingest documents (if index is empty)
  2. Run the LangGraph research agent on a query
  3. Print the formatted research report
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich import print as rprint

load_dotenv()

console = Console()


def require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        console.print(f"[red]Error: environment variable {key} is not set.[/red]")
        console.print("Copy .env.example to .env and fill in your ANTHROPIC_API_KEY.")
        sys.exit(1)
    return val


def ensure_ingested(store) -> None:
    if store.count() == 0:
        console.print("[yellow]Vector store is empty — ingesting sample documents...[/yellow]")
        from src.ingestion.loader import load_documents
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.embedder import embed_chunks

        docs = load_documents("data/sample_docs")
        chunks = chunk_documents(docs)
        embeddings = embed_chunks(chunks)
        store.add_chunks(chunks, embeddings)
        console.print(f"[green]Indexed {store.count()} chunks.[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Enterprise RAG Research Agent powered by LangGraph + Claude"
    )
    parser.add_argument(
        "--query",
        default="What are the key benefits and metrics of deploying RAG systems in enterprise environments?",
        help="Research question to answer",
    )
    parser.add_argument(
        "--topic",
        default="Enterprise AI and RAG Systems",
        help="Research topic label for the report",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save report to this file path (optional)",
    )
    args = parser.parse_args()

    from src.retrieval.vectorstore import VectorStore
    from src.retrieval.retriever import DocumentRetriever
    from src.agents.graph import run_research_agent

    # Initialize infrastructure
    store = VectorStore()
    ensure_ingested(store)
    retriever = DocumentRetriever(store)

    console.print(Panel(
        f"[bold cyan]Query:[/bold cyan] {args.query}\n"
        f"[bold cyan]Topic:[/bold cyan] {args.topic}",
        title="Enterprise RAG Research Agent",
        border_style="cyan",
    ))

    console.print("\n[yellow]Running LangGraph research agent...[/yellow]\n")

    final_state = run_research_agent(
        query=args.query,
        topic=args.topic,
        retriever=retriever,
    )

    # Print stats
    console.print(f"[green]Route:[/green] {final_state.get('route', 'unknown')}")
    console.print(f"[green]Sources graded as relevant:[/green] {len(final_state.get('graded_docs', []))}")
    console.print(f"[green]Citations:[/green] {len(final_state.get('citations', []))}")
    console.print()

    report = final_state.get("report", "No report generated.")
    console.print(Markdown(report))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        console.print(f"\n[green]Report saved to: {output_path}[/green]")


if __name__ == "__main__":
    main()
