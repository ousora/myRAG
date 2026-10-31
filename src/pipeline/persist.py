"""SQLite persistence helpers for the hybrid pipeline.

Extracted from ``pipeline.hybrid`` to keep that module under 500 lines.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import httpx

from myrag.exceptions import MyRagException

logger = logging.getLogger(__name__)


def persist_to_db(
    all_chunks: list[dict[str, Any]],
    store_path: str,
    doc_id: str,
    title: str,
    tags: list[str],
    summary_text: str,
    filepath: str,
    db_write_lock: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Embed chunks and persist to sqlite-vec.

    Returns ``(stored_chunks, stored_doc, store_path)`` on success, or
    ``(stored_chunks, stored_doc, None)`` when embedding/storage failed and
    the result degraded to in-memory, vector-less chunks.

    The lock serializes DB writes across batch threads; the context manager
    closes the connection even when an upsert fails mid-way.
    """
    from embedders import maybe_create_embedder
    from storage.sqlite_vec import SQLiteVecStore

    e: Any = None
    try:
        e = maybe_create_embedder()
        if e is None:
            # Embedding disabled (mode="none"): persist without vectors;
            # retrieval runs keyword-only. upsert_chunks defaults to a
            # zero-vector placeholder when "embedding" is missing.
            stored_chunks, stored_doc = _build_stored_result(
                all_chunks, doc_id, title, tags, summary_text, filepath,
            )
            doc_embedding: list[float] | None = None
        else:
            stored_chunks = e.store_chunks(all_chunks, doc_id=doc_id)

            # Document-level index (B) — only embed when we persist it.
            stored_doc = e.store_document(
                title=title,
                tags=tags,
                text_summary=summary_text,
                source_file=filepath,
                total_chunks=len(stored_chunks),
            )
            doc_embedding = stored_doc.get("embedding")

        with db_write_lock, SQLiteVecStore(store_path) as db:
            db.upsert_chunks(stored_chunks, doc_id=doc_id)
            db.upsert_document(
                title=title,
                tags=tags,
                text_summary=summary_text,
                source_file=filepath,
                total_chunks=len(stored_chunks),
                embedding=doc_embedding,
            )
    except (httpx.HTTPError, RuntimeError, sqlite3.Error, MyRagException) as exc:
        logger.warning("Embedding/storage failed (%s): %s", type(exc).__name__, exc)
        stored_chunks, stored_doc = _build_stored_result(
            all_chunks, doc_id, title, tags, summary_text, filepath, truncate_summary=True,
        )
        # Nothing was persisted (or only partial writes landed before the
        # failure), so report no DB path — matching the pre-refactor contract.
        return stored_chunks, stored_doc, None
    finally:
        if e is not None:
            e.close()

    logger.info("  → Persisted %d chunks + 1 doc to %s", len(stored_chunks), store_path)
    return stored_chunks, stored_doc, store_path


def _build_stored_result(
    chunks: list[dict[str, Any]],
    doc_id: str,
    title: str,
    tags: list[str],
    summary_text: str,
    filepath: str,
    *,
    truncate_summary: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the ``(stored_chunks, stored_doc)`` pair for persistence or fallback.

    Used by ``process_file_hybrid`` in three places: normal embedding path,
    exception fallback path, and the ``store_path is None`` path.
    """
    stored_chunks = [
        {**c, "source_doc_id": doc_id, "chunk_index": i}
        for i, c in enumerate(chunks)
    ]
    stored_doc: dict[str, Any] = {
        "title": title,
        "tags": tags,
        "text_summary": summary_text[:1000] if truncate_summary else summary_text,
        "source_file": filepath,
        "total_chunks": len(stored_chunks),
    }
    return stored_chunks, stored_doc
