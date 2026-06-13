"""PyMuPDF-based PDF parser."""

import logging
from pathlib import Path


try:
    import fitz  # PyMuPDF  # noqa: F401
except ImportError:
    pass  # skipped by __init__ loop if dependency not installed

logger = logging.getLogger(__name__)


class PDFParser:
    """Extract text from PDFs using PyMuPDF.

    Handles: plain text, scanned images (basic), and multi-column layouts.
    """

    def __init__(self, *, pages=None, max_pages=0):
        self._pages = pages  # list[int] | None — specific page numbers to extract
        self._max_pages = max_pages  # 0 means no limit

    def parse(self, filepath: str) -> str:
        path = Path(filepath).resolve()
        doc = fitz.open(str(path))

        if self._pages:
            pages_to_read = sorted(set(p for p in self._pages if 0 <= p < len(doc)))
        else:
            pages_to_read = list(range(min(len(doc), self._max_pages) if self._max_pages else len(doc)))

        texts: list[str] = []
        for i, page_num in enumerate(pages_to_read):
            page_text = doc[page_num].get_text("text")
            logger.debug(
                "  Page %d — %.0f chars",
                page_num + 1, len(page_text),
            )
            texts.append(page_text)

        doc.close()
        full_text = "\n--- PAGE BREAK ---\n".join(texts) if len(pages_to_read) > 1 else texts[0] if texts else ""
        return full_text


# Register after class definition
from .dispatcher import register_parser, TextParser  # noqa: E402
register_parser("pdf", PDFParser)
