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
    """Load a PDF using pdfplumber (table-aware) with pypdf fallback."""
    try:
        import pdfplumber
        return _load_pdf_pdfplumber(path, pdfplumber)
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pdfplumber or pypdf: pip install pdfplumber")

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(RawDocument(
                content=text.strip(),
                source=str(path),
                page=i + 1,
                metadata={"type": "pdf", "filename": path.name, "content_type": "prose"},
            ))
    return docs


def _load_pdf_pdfplumber(path: Path, pdfplumber) -> List[RawDocument]:
    """
    Extract PDFs with content-type awareness:
      - Tables  → one RawDocument per row, structured key:value content
      - Prose   → text outside table bounding boxes, one RawDocument per page
    """
    docs = []

    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            tables = page.find_tables()
            table_bboxes = []

            for table in tables:
                bbox = table.bbox  # (x0, top, x1, bottom)
                table_bboxes.append(bbox)
                data = table.extract()

                if not data or len(data) < 2:
                    continue

                # Treat the first row as the header
                header = [
                    str(c).strip() if c else f"col_{j}"
                    for j, c in enumerate(data[0])
                ]

                for row in data[1:]:
                    if not any(c for c in row if c and str(c).strip()):
                        continue  # skip blank rows

                    parts = []
                    row_meta: dict = {"content_type": "table_row"}
                    for j, cell in enumerate(row):
                        val = str(cell).strip() if cell else ""
                        if not val or j >= len(header):
                            continue
                        col = header[j]
                        parts.append(f"{col}: {val}")
                        # Normalise key for metadata storage
                        meta_key = col.lower().replace(" ", "_").replace(".", "").replace("/", "_")
                        row_meta[meta_key] = val

                    content = " | ".join(parts)
                    if not content.strip():
                        continue

                    docs.append(RawDocument(
                        content=f"[TABLE ROW] {content}",
                        source=str(path),
                        page=page_num,
                        metadata={"type": "pdf", "filename": path.name, **row_meta},
                    ))

            # Extract prose text from regions outside table bounding boxes
            words = page.extract_words()
            if table_bboxes:
                prose_words = [
                    w["text"] for w in words
                    if not _word_in_any_bbox(w, table_bboxes)
                ]
            else:
                prose_words = [w["text"] for w in words]

            prose_text = " ".join(prose_words).strip()
            if prose_text:
                docs.append(RawDocument(
                    content=prose_text,
                    source=str(path),
                    page=page_num,
                    metadata={"type": "pdf", "filename": path.name, "content_type": "prose"},
                ))

    return docs


def _word_in_any_bbox(word: dict, bboxes: list) -> bool:
    """Return True if a word's position falls inside any bounding box."""
    x0, top = word["x0"], word["top"]
    for b in bboxes:
        if b[0] <= x0 <= b[2] and b[1] <= top <= b[3]:
            return True
    return False


def _load_text(path: Path) -> List[RawDocument]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [RawDocument(
        content=text,
        source=str(path),
        page=1,
        metadata={"type": path.suffix.lstrip("."), "filename": path.name, "content_type": "prose"},
    )]
