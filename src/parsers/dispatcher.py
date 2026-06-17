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
# Thread safety: reads (resolve_parser) are safe from multiple threads.
# Writes (register_parser) require external synchronization.

# Common extension aliases — map primary name to its variants
_ALIASES: dict[str, list[str]] = {
    "markdown": ["md", "mkd"],
    "html": ["htm"],
}


def register_parser(extension: str, parser_cls: type[TextParser]) -> None:
    """Register a parser for the given file extension.

    Not thread-safe for concurrent registration; safe to read from multiple threads.
    """
    PARSERS[extension.lower()] = parser_cls
    if extension.lower() in _ALIASES:
        for alias in _ALIASES[extension.lower()]:
            PARSERS[alias] = parser_cls


# ---------------------------------------------------------------------------
# MarkItDown Parser (pdf, docx, markdown, txt)
# ---------------------------------------------------------------------------

class MarkItDownParser(TextParser):
    """Extract text using MarkItDown — handles pdf, docx, markdown, txt.

    MarkItDown is imported lazily so the module can be loaded even if the
    package isn't installed. Dependency is checked upon initialization.
    """

    def __init__(self) -> None:
        try:
            from markitdown import MarkItDown
            self._converter = MarkItDown()
        except ImportError:
            raise RuntimeError(
                "markitdown is required for this parser. "
                "Install with: pip install markitdown"
            ) from None

    def parse(self, filepath: str) -> str:
        result = self._converter.convert(filepath)
        return result.text_content or ""


# ---------------------------------------------------------------------------
# Trafilatura Parser (html, htm)
# ---------------------------------------------------------------------------

class TrafilaturaParser(TextParser):
    """Extract text using Trafilatura — handles html, htm.

    Trafilatura is imported lazily. Dependency is checked upon initialization.
    Uses a pragmatic utf-8/gbk fallback for encoding handling.
    """

    def __init__(self) -> None:
        try:
            import trafilatura
            # Store function reference only, lighter than storing the whole module.
            self._extract = trafilatura.extract
        except ImportError:
            raise RuntimeError(
                "trafilatura is required for this parser. "
                "Install with: pip install trafilatura"
            ) from None

    def parse(self, filepath: str) -> str:
        path = Path(filepath)

        # Pragmatic encoding handling — try UTF-8 first, fallback to GBK for
        # most Chinese web pages.
        try:
            html_content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            html_content = path.read_text(encoding="gbk", errors="replace")

        # trafilatura.extract() only accepts HTML strings (not file paths).
        content = self._extract(
            html_content,
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

logger.debug(
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

    # Fallback: MarkItDown also supports these formats but they're not explicitly
    # registered (to keep PARSERS dict clean — only primary types are registered).
    if ext.lower() in ("pptx", "xls", "xlsx", "epub"):
        logger.info("Falling back to MarkItDownParser for %s (.%s)", path.name, ext)
        return MarkItDownParser()

    logger.warning("No parser registered for .%s — skipping %s", ext, filepath)
    return None
