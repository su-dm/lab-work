"""PDF and text file extraction utility.

Usage:
    from pdf_utils import extract_text
    text = extract_text("document.pdf")
    text = extract_text("document.txt")
"""

from pathlib import Path


def extract_text(path: str) -> str:
    """Extract text from a PDF or plain text file.

    Transparently handles .pdf (via pymupdf) and .txt files.
    Raises FileNotFoundError if the file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".pdf":
        return _extract_pdf(p)
    else:
        return p.read_text(encoding="utf-8")


def _extract_pdf(path: Path) -> str:
    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF extraction. Install with: pip install pymupdf"
        )

    doc = pymupdf.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)
