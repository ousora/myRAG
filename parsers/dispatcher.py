"""Parser dispatcher — route file types to the right parser."""

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TextParser(Protocol):
    """Interface for text extractors from documents."""

    def parse(self, filepath: str) -> str: ...


PARSERS: dict[str, type[TextParser]] = {}


def register_parser(extension: str, parser_cls: type[TextParser]) -> None:
    """Register a parser for the given file extension."""
    PARSERS[extension.lower()] = parser_cls
    # Also alias common extensions (e.g., .md → markdown)
    aliases = {
        "markdown": ["md", "mkd"],
        "html": ["htm"],
    }
    if extension.lower() in aliases:
        for alias in aliases[extension.lower()]:
            PARSERS[alias] = parser_cls
    logger.debug("Registered parser '%s' for extension .%s", parser_cls.__name__, extension)


def resolve_parser(filepath: str | Path) -> TextParser | None:
    """Look up and return a parser instance, or None if unsupported."""
    path = Path(filepath)
    ext = path.suffix.lstrip(".")
    # Try exact match first, then try with dot prefix for extensions like .md → markdown
    cls = PARSERS.get(ext.lower())
    if cls is not None:
        instance = cls()
        logger.info("Using %s to parse %s (.%s)", cls.__name__, path.name, ext)
        return instance
    
    # Try matching the extension against all registered parsers' aliases
    for key, parser_cls in PARSERS.items():
        if key.lower().startswith(ext.lower()):
            instance = parser_cls()
            logger.info("Using %s to parse %s (.%s → .%s)", parser_cls.__name__, path.name, ext, key)
            return instance

    logger.warning("No parser registered for .%s — skipping %s", ext, filepath)
    return None
