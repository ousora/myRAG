"""Markdown parser."""

import logging
from pathlib import Path


try:
    import markdown as md_lib  # noqa: F401
except ImportError:
    pass  # skipped by __init__ loop if dependency not installed

logger = logging.getLogger(__name__)


class MarkdownParser:
    """Convert Markdown to plain text.

    Strips HTML tags produced by the MD→HTML conversion for clean output.
    """

    def parse(self, filepath: str) -> str:
        path = Path(filepath).resolve()
        raw_text = path.read_text(encoding="utf-8", errors="replace")

        html_output = md_lib.markdown(raw_text, extensions=["tables"])
        cleaned = self._strip_html(html_output)
        logger.info("  Parsed %s — %.0f chars", path.name, len(cleaned))
        return cleaned

    @staticmethod
    def _strip_html(text: str) -> str:
        import re
        # Remove <br> / <hr /> tags
        text = re.sub(r"<(br|HR)\b[^>]*/?>", "\n", text, flags=re.I)
        # Strip remaining HTML tags
        text = re.sub(r"<.*?>", "", text, flags=re.S)
        return text


# Register after class definition
from .dispatcher import register_parser, TextParser  # noqa: E402
register_parser("markdown", MarkdownParser)