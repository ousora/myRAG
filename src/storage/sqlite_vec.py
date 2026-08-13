"""SQLite-based vector store using sqlite-vec extension.

Splits into submodules for each responsibility:
  - schema.py     — table creation, schema definitions, sqlite-vec loader
  - inserts.py    — upsert_chunk, upsert_chunks, upsert_document
  - search.py     — hybrid_search, search_documents, get_embeddings_by_ids

The public API (SQLiteVecStore class) remains unchanged; methods delegate
to the appropriate submodule.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .inserts import _InsertOps
from .schema import _SQLITE_VEC
from .search import _SearchOps, _build_fts_query, _deserialize_embedding  # noqa: F401 — re-exported for tests

logger = logging.getLogger(__name__)


class SQLiteVecStore(_InsertOps, _SearchOps):
    """SQLite-backed vector store using sqlite-vec extension.

    Inherits insert and search operations from mixin classes; the class
    itself only manages connection lifecycle and shared state.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        import sqlite3
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.enable_load_extension(True)
        _load_sqlite_vec().load(conn)  # type: ignore[attr-defined]

        self.conn = conn
        self._schema_ready = False
        self.conn.execute("PRAGMA journal_mode=WAL")

    def get_chunks_by_doc(self, doc_id: str) -> list[dict]:
        """Retrieve all chunks for a specific document."""
        self._setup_schema()

        results = self.conn.execute(
            "SELECT * FROM chunks WHERE source_doc_id = ? ORDER BY chunk_index",
            (doc_id,)
        ).fetchall()

        return [{
            "id": row[0],
            "text": row[1],
            "embedding": _deserialize_embedding(row[2]),
            "source_doc_id": row[3],
            "chunk_index": row[4],
            "section_path": self._parse_section_path(row[5]) if row[5] else ["General"],
            "word_count": row[6],
        } for row in results]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def close(self):
        """Close the database connection safely."""
        try:
            self.conn.commit()
        except Exception as exc:  # noqa: BLE001 — best-effort commit on close
            logger.debug("Commit during close failed: %s", exc)
        finally:
            try:
                self.conn.close()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Close failed: %s", exc)


def _load_sqlite_vec() -> object:
    """Re-export the loader so callers can access it if needed."""
    from .schema import _load_sqlite_vec as _inner
    return _inner()
