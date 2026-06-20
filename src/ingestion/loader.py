from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field


@dataclass
class RawDocument:
    content: str
    source: str
    page: int = 0
    metadata: dict = field(default_factory=dict)


def load_documents(data_dir: Union[str, Path]) -> List[RawDocument]:
    """Load all supported documents from a directory."""
    data_dir = Path(data_dir)
    docs: List[RawDocument] = []

    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            if path.suffix.lower() == ".pdf":
                docs.extend(_load_pdf(path))
            elif path.suffix.lower() in {".txt", ".md"}:
                docs.extend(_load_text(path))

    return docs


def _load_pdf(path: Path) -> List[RawDocument]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf required: pip install pypdf")

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(RawDocument(
                content=text.strip(),
                source=str(path),
                page=i + 1,
                metadata={"type": "pdf", "filename": path.name},
            ))
    return docs


def _load_text(path: Path) -> List[RawDocument]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [RawDocument(
        content=text,
        source=str(path),
        page=1,
        metadata={"type": path.suffix.lstrip("."), "filename": path.name},
    )]
