"""Standalone report formatting utilities (used outside the agent graph)."""
from datetime import datetime
from typing import List


def build_report_header(topic: str, query: str) -> str:
    date = datetime.now().strftime("%B %d, %Y")
    return f"# Research Report: {topic}\n\n**Date:** {date}  \n**Query:** {query}\n\n---\n\n"


def build_sources_appendix(citations: List[str]) -> str:
    if not citations:
        return ""
    lines = ["\n\n---\n\n## References\n"]
    for i, cite in enumerate(citations, 1):
        lines.append(f"{i}. {cite}")
    return "\n".join(lines)


def format_full_report(topic: str, query: str, body: str, citations: List[str]) -> str:
    header = build_report_header(topic, query)
    appendix = build_sources_appendix(citations)
    return header + body + appendix
