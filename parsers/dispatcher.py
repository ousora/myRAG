"""Parser dispatcher — route file types to the right parser.

Unified backend: MarkItDown (pdf/docx/md/txt) + Trafilatura (html).
Old individual parsers removed; dispatch registered at module load time.
"""

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal protocol — not public API, just for type-checking dispatcher internals
# ---------------------------------------------------------------------------
@runtime_checkable
class TextParser(Protocol):
    """Interface for text extractors from documents."""

    def parse(self, filepath: str) -> str: ...


PARSERS: dict[str, type[TextParser]] = {}


def register_parser(extension: str, parser_cls: type[TextParser]) -> None:
    """Register a parser for the given file extension."""
    PARSERS[extension.lower()] = parser_cls
    # Alias common extensions
    aliases: dict[str, list[str]] = {
        "markdown": ["md", "mkd"],
        "html": ["htm"],
    }
    if extension.lower() in aliases:
        for alias in aliases[extension.lower()]:
            PARSERS[alias] = parser_cls


# ---------------------------------------------------------------------------
# MarkItDown Parser (pdf, docx, markdown, txt)
# ---------------------------------------------------------------------------

try:
    from markitdown import MarkItDown  # noqa: F401
except ImportError:
    raise RuntimeError(
        "markitdown is required for PDF/DOCX/MD/TXT parsing. "
        "Install with: pip install markitdown"
    ) from None


class MarkItDownParser(TextParser):
    """Extract text using MarkItDown — handles pdf, docx, markdown, txt."""

    def __init__(self) -> None:
        self._converter = MarkItDown()

    def parse(self, filepath: str) -> str:  # type: ignore[override]
        result = self._converter.convert(filepath)
        return result.text_content


# ---------------------------------------------------------------------------
# Trafilatura Parser (html, htm)
# ---------------------------------------------------------------------------

try:
    import trafilatura  # noqa: F401
except ImportError:
    raise RuntimeError(
        "trafilatura is required for HTML parsing. "
        "Install with: pip install trafilatura"
    ) from None


class TrafilaturaParser(TextParser):
    """Extract text using Trafilatura — handles html, htm."""

    def parse(self, filepath: str) -> str:  # type: ignore[override]
        content = trafilatura.extract(
            filepath,
            include_comments=False,
            include_tables=True,
            prefer_full_output=True,
        ) or ""
        return content.strip()


# ---------------------------------------------------------------------------
# Register built-in parsers at module load time
# ---------------------------------------------------------------------------

register_parser("pdf", MarkItDownParser)
register_parser("docx", MarkItDownParser)
register_parser("markdown", MarkItDownParser)  # also aliases md/mkd
register_parser("txt", MarkItDownParser)        # txt maps to MarkItDown via generic handler
register_parser("html", TrafilaturaParser)      # also aliases htm

logger.info(
    "Registered %d parsers: pdf, docx, markdown(.md/.mkd), txt, html(.htm)",
    len(PARSERS),
)


# ---------------------------------------------------------------------------
# Public API — resolve_parser is the only external-facing function
# ---------------------------------------------------------------------------

def resolve_parser(filepath: str | Path) -> TextParser | None:
    """Look up and return a parser instance, or None if unsupported."""
    path = Path(filepath)
    ext = path.suffix.lstrip(".")
    cls = PARSERS.get(ext.lower())
    if cls is not None:
        instance = cls()  # type: ignore[call-arg]
        logger.info("Using %s to parse %s (.%s)", cls.__name__, path.name, ext)
        return instance

    # Fallback: try MarkItDown for unknown extensions it can handle
    if ext.lower() in ("pptx", "xls", "xlsx", "epub"):
        logger.info("Falling back to MarkItDownParser for %s (.%s)", path.name, ext)
        return MarkItDownParser()

    logger.warning("No parser registered for .%s — skipping %s", ext, filepath)
    return None
