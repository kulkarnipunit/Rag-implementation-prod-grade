from typing import TypedDict, List, Literal, Optional


class ResearchState(TypedDict):
    # Input
    query: str
    topic: str

    # Routing decision
    route: Literal["vectorstore", "direct_answer"]

    # Retrieval
    retrieved_docs: List[dict]

    # After grading — only relevant chunks
    graded_docs: List[dict]

    # Generated answer text
    generation: str

    # Formatted citations list
    citations: List[str]

    # Final report markdown
    report: str

    # Retry counter for the retrieve→grade loop
    retry_count: int

    # Set to True if grading found no usable context
    no_context_fallback: bool
