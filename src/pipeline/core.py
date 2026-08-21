"""RAG data cleanup pipeline — parse → clean → format → chunk → embed (Hybrid A+B).

Pipeline flow:
    Raw file (.pdf/.docx/.txt)
        ↓ parser.parse_file()     # Unified backend: MarkItDown + Trafilatura
        ↓ cleaner.clean_text()    # Deterministic regex-based noise removal
        ↓ formatter.format_text()  # LLM semantic structuring → title/tags/chunks
            → write_to_md(result, output_dir/)   [human-readable .md]
        ↓ chunker.chunk(section_path=...)     # Physical splitting with section headers
        ↓ embedder.store_chunks() / store_document()  # Hybrid A+B indexing

Hybrid Retrieval (A + B):
     A: Chunk-level index — fine-grained search, direct answer generation
        [chunk] → bge-m3 embedding → sqlite-vec vector DB

     B: Document-level index — coarse-grained context fallback
       [doc_summary] → bge-m3 embedding → sqlite-vec vector DB

Usage (traditional RAG):
    from pipeline.core import process_file, process_directory

    chunks = process_file("path/to/report.pdf")  # traditional chunking only

Usage (LLM-formatted + Hybrid A+B):
    from pipeline.core import process_file_hybrid

    result = process_file_hybrid(
        filepath="path/to/document.pdf",
        doc_id="doc_001"
    )
    # Returns: {
    #   "chunks": list[dict  (A - fine-grained, ready for sqlite-vec)
    #   "document": dict      (B - coarse-grained, ready for sqlite-vec)
    # }

Usage (LLM-formatted + Markdown output):
    from pipeline.core import process_file_with_md

    md_path = process_file_with_md(
        filepath="path/to/document.pdf",
        output_dir="./output/",
    )
"""

from __future__ import annotations

import logging
from typing import Any

# Trigger parser registration at module load time
import parsers  # noqa: F401 — loads dispatcher (MarkItDown + Trafilatura)
from config import CLEAN_RULES_PATH
from storage.sqlite_vec import SQLiteVecStore  # noqa: F401,TC001 — re-exported for callers

from . import (
    markdown_utils,  # noqa: F401,TC001 — re-exported so tests can patch pipeline.core.markdown_utils
    utils,
)
from .hybrid import Chunker, TextCleaner, _get_config  # noqa: F401,TC001 — re-exported for test patching
from .utils import build_doc_summary as _build_doc_summary  # noqa: F401,TC001 — re-exported for test import

logger = logging.getLogger(__name__)


def process_file(filepath: str, *, remove_page_breaks: bool = True, collapse_whitespace: bool = True,
                 rules_config: str | None = None, chunk_size: int = 1024) -> list[dict[str, Any]]:
    """Parse a single file and return structured chunks (traditional RAG).

    Pipeline: parser → cleaner → chunker → output dict list.

    For LLM-formatted output with hybrid A+B indexing, use process_file_hybrid().

    Returns list of dicts: [{"text": ..., "metadata": {...}}, ...]
    """
    parser = utils.resolve_parser(filepath)
    if parser is None:
        logger.warning("Skipped %s — no parser found", filepath)
        return []

    raw_text = parser.parse(filepath)
    if rules_config is None:
        rules_config = str(CLEAN_RULES_PATH)
    cleaned = TextCleaner(remove_page_breaks=remove_page_breaks, collapse_whitespace=collapse_whitespace, rules_config=rules_config).clean(raw_text)
    chunks = Chunker(chunk_size=chunk_size).chunk(cleaned)

    result = [
        {"text": chunk["text"], "metadata": {"source": filepath}}
        for chunk in chunks
    ]
    logger.info("  → %d chunks from %s", len(result), filepath)
    return result


def process_directory(dirpath: str, *, extensions: set[str] | list[str] | None = None,
                      chunk_size: int = 1024, **kwargs: Any) -> list[dict[str, Any]]:
    """Walk a directory and process all supported files (traditional RAG)."""
    from pathlib import Path

    from parsers.dispatcher import PARSERS

    path = Path(dirpath)

    if extensions is None:
        extensions = set(PARSERS.keys())

    results: list[dict] = []
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        ext = file.suffix.lstrip(".")
        if ext.lower() in {e.lower() for e in extensions}:
            chunks = process_file(str(file), **kwargs)
            results.extend(chunks)

    logger.info("Processed %d files from %s → %d total chunks", len(results), dirpath, len(results))
    return results


# Re-export hybrid functions from the dedicated module for backward compatibility.
from .hybrid import (  # noqa: E402, F401
    process_directory_hybrid,
    process_file_hybrid,
    process_file_with_md,
    rag_query,
)
