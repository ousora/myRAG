"""Utility functions for the RAG pipeline.

This module contains helper functions used by the main pipeline:
- Building document summaries for coarse-grained indexing
- Resolving parsers for file types
- Mapping file extensions to source types
"""

from __future__ import annotations

from pathlib import Path


def build_doc_summary(title: str, tags: list[str], body: str, *, head: int = 800, tail: int = 400) -> str:
    """Build a document-level (B index) summary from head + tail of the body.

    The first ``head`` chars capture the document's lead/abstract; for longer
    documents the last ``tail`` chars are appended so the closing section also
    informs the coarse-grained embedding (otherwise the B index only ever sees
    the opening, which harms retrieval for documents whose answer is near the end).
    """
    body = body or ""
    if len(body) <= head:
        snippet = body
    else:
        snippet = body[:head]
        if tail and len(body) > head + tail:
            snippet += "\n...\n" + body[-tail:]
    tag_str = " ".join(tags) if tags else ""
    return f"Title: {title}\nTags: {tag_str}\n{snippet}".strip()


def resolve_parser(filepath: str):
    """Resolve a parser for the given file path.
    
    Args:
        filepath: Path to the file to parse.
        
    Returns:
        Parser instance or None if no parser is available.

    """
    from parsers.dispatcher import resolve_parser as rp
    return rp(filepath)


def source_type_for(filepath: str) -> str:
    """Map a file extension to the formatter's source_type hint.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        Source type string for the formatter.

    """
    ext = Path(filepath).suffix.lstrip(".").lower()
    return {
        "pdf": "pdf",
        "docx": "pdf",
        "html": "web",
        "htm": "web",
        "md": "markdown",
        "mkd": "markdown",
        "txt": "web",
    }.get(ext, "web")
