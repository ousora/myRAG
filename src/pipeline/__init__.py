"""RAG data cleanup pipeline — parse → clean → format → chunk → embed (Hybrid A+B).

This package re-exports the public API from submodules for backward compatibility.
All imports should work as before: `from pipeline import process_file, ...`
"""

# Re-export core functions for backward compat
from pipeline.core import (
    Chunker,
    TextCleaner,
    _resolve_parser,
    process_directory,
    process_file,
    process_file_hybrid,
    process_file_with_md,
    rag_query,
)

# Re-export ingest for backward compat (used by docs and skills)
from pipeline.ingest import _ingest_markdown

__all__ = [
    "Chunker",
    "TextCleaner",
    "_resolve_parser",
    "process_directory",
    "process_file",
    "process_file_hybrid",
    "process_file_with_md",
    "rag_query",
    "_ingest_markdown",
]
