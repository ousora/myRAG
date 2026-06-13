"""python-docx-based DOCX parser."""

import logging
from pathlib import Path


try:
    from docx import Document as DocxDocument  # noqa: F401
except ImportError:
    pass  # skipped by __init__ loop if dependency not installed

logger = logging.getLogger(__name__)


class DOCXParser:
    """Extract text from Microsoft Word documents."""

    def parse(self, filepath: str) -> str:
        path = Path(filepath).resolve()
        doc = DocxDocument(str(path))

        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n".join(paragraphs)

        logger.info("  Extracted %d paragraphs from %s", len(paragraphs), path.name)
        return full_text


# Register after class definition (decorator can't reference itself)
from .dispatcher import register_parser, TextParser  # noqa: E402
register_parser("docx", DOCXParser)
