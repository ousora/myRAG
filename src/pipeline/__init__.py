"""RAG data cleanup pipeline — parse → clean → format → chunk → embed (Hybrid A+B).

This package re-exports the public API from submodules for backward compatibility.
All imports should work as before: `from pipeline import process_file, ...`
"""

# Re-export core functions for backward compat
from pipeline.core import (
    Chunker,
    TextCleaner,
    process_directory,
    process_file,
    process_file_hybrid,
    process_file_with_md,
    rag_query,
)

# Re-export utilities that were previously in core
from pipeline.utils import resolve_parser, source_type_for

# Re-export ingest for backward compat (used by docs and skills)
from pipeline.ingest import _ingest_markdown

__all__ = [
    "Chunker",
    "TextCleaner",
    "resolve_parser",
    "source_type_for",
    "process_directory",
    "process_file",
    "process_file_hybrid",
    "process_file_with_md",
    "rag_query",
    "_ingest_markdown",
]
