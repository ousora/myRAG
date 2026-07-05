"""SQLite-based vector store using sqlite-vec extension."""

from __future__ import annotations

import importlib
import json
import logging
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic loader for the third-party ``sqlite-vec`` package.
#
# Strategy 1: direct ``import sqlite_vec`` — works when installed normally.
# Strategy 2: filesystem path via
#   ``importlib.metadata.distribution("sqlite-vec").files`` — robust across
#   editable installs, wheels, and different Python versions.
# ---------------------------------------------------------------------------
_sqlite_vec: Optional[object] = None


def _load_sqlite_vec() -> object:
    """Return the third-party ``sqlite_vec`` module (loaded once).

    Tries two strategies in order:
      1. ``importlib.import_module("sqlite_vec")`` — works for pip/uv-installed
         packages (the canonical case).
      2. Locate via distribution metadata → file-based loading — robust across
         editable installs, wheels, and different Python versions.

    Raises RuntimeError if neither strategy succeeds.
    """
    global _sqlite_vec
    if _sqlite_vec is not None:
        return _sqlite_vec

    # Strategy 1: Direct import (canonical install path).
    try:
        _sqlite_vec = importlib.import_module("sqlite_vec")
        return _sqlite_vec
    except ImportError:
        pass

    from importlib.metadata import PackageNotFoundError, distribution
    import importlib.util as _util

    # Strategy 2: Locate __init__.py via the distribution's file list — robust
    # across editable installs, wheels, and different Python versions.
    try:
        dist = distribution("sqlite-vec")
    except PackageNotFoundError as exc:  # type: ignore[attr-defined]
        raise RuntimeError(
            "The 'sqlite-vec' package is required but not installed.\n"
            "Install it with: pip install sqlite-vec\n"
            "(or: uv add --dev sqlite-vec)"
        ) from exc

    init_py = next(
        (f for f in dist.files or [] if str(f) == "sqlite_vec/__init__.py"),
        None,
    )
    if init_py is None:
        raise RuntimeError(
            "Could not locate sqlite_vec.__init__ inside the 'sqlite-vec' distribution.\n"
            "The installed version may be corrupted or incompatible."
        )

    spec = _util.spec_from_file_location(
        "_third_party_sqlite_vec", str(dist.locate_file(init_py)),
    )
    mod = _util.module_from_spec(spec)  # type: ignore[union-attr]
    sys.modules["_third_party_sqlite_vec"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _sqlite_vec = mod
    return mod


# Cached reference for convenience — callers use this instead of calling
# ``_load_sqlite_vec()`` every time.
_SQLITE_VEC = _load_sqlite_vec()


class SQLiteVecStore:
    """SQLite-backed vector store using sqlite-vec extension."""

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

    def _setup_schema(self):
        """Create tables and FTS sync triggers if they don't exist.

        Idempotent — runs at most once per connection. Avoids the cost of
        ``executescript`` (which implicitly commits) on every query.
        """
        if self._schema_ready:
            return
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source_doc_id TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                section_path TEXT,
                word_count INTEGER,
                entity_names TEXT DEFAULT '[]',
                UNIQUE(source_doc_id, chunk_index)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_doc_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, content='chunks', content_rowid='id'
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                text_summary TEXT NOT NULL,
                source_file TEXT NOT NULL,
                total_chunks INTEGER DEFAULT 0,
                created_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_file);

            -- Keep FTS index in sync with chunks table on INSERT/UPDATE/DELETE.
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES('delete', old.id, old.text);
            END;
        """)
        self._schema_ready = True

    def upsert_chunk(self, chunk_data: dict, *, doc_id: str,
                     embedding: list[float], chunk_index: int = 0) -> dict:
        """Insert or replace a chunk by (source_doc_id, chunk_index).

        Uses INSERT OR REPLACE so re-ingesting the same document overwrites
        existing chunks instead of creating duplicates. The AFTER UPDATE trigger
        on chunks also keeps FTS in sync.
        """
        self._setup_schema()

        section_path = chunk_data.get("section_path", ["General"])
        entity_names = chunk_data.get("entity_names", [])
        word_count = len(chunk_data.get("text", "").split())

        cursor = self.conn.execute(
            """INSERT OR REPLACE INTO chunks (text, embedding, source_doc_id, chunk_index, section_path, word_count, entity_names)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chunk_data["text"], _SQLITE_VEC.serialize_float32(embedding), doc_id,
             chunk_index, json.dumps(section_path), word_count, json.dumps(entity_names))
        )

        return {
            "id": cursor.lastrowid or self.conn.execute(
                "SELECT id FROM chunks WHERE source_doc_id=? AND chunk_index=?", (doc_id, chunk_index)
            ).fetchone()[0],
            "text": chunk_data["text"],
            "section_path": section_path,
            "source_doc_id": doc_id,
            "chunk_index": chunk_index,
            "word_count": word_count,
        }

    def upsert_chunks(self, chunks: list[dict], *, doc_id: str) -> list[dict]:
        """Batch insert or replace multiple chunks in a single transaction."""
        self._setup_schema()

        results = []
        for i, chunk in enumerate(chunks):
            section_path = chunk.get("section_path", ["General"])
            entity_names = chunk.get("entity_names", [])
            word_count = len(chunk.get("text", "").split())

            cursor = self.conn.execute(
                """INSERT OR REPLACE INTO chunks (text, embedding, source_doc_id, chunk_index, section_path, word_count, entity_names)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chunk["text"], _SQLITE_VEC.serialize_float32(chunk.get("embedding", [0.0])),
                 doc_id, i, json.dumps(section_path), word_count, json.dumps(entity_names))
            )
            results.append({
                "id": cursor.lastrowid or self.conn.execute(
                    "SELECT id FROM chunks WHERE source_doc_id=? AND chunk_index=?", (doc_id, i)
                ).fetchone()[0],
                "text": chunk["text"],
                "section_path": section_path,
                "source_doc_id": doc_id,
                "chunk_index": i,
                "word_count": word_count,
            })

        self.conn.commit()
        return results

    def _parse_section_path(self, raw: str) -> list[str]:
        """Parse section path from JSON or empty string."""
        if not raw or len(raw.strip()) < 1:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            elif isinstance(data, str):
                return [data]
            else:
                return [str(data)]
        except json.JSONDecodeError:
            logger.debug("Failed to parse section_path JSON: %r", raw)
            return []

    def search_chunks(self, query_vector: list[float], *, k: int = 10,
                      source_doc_id: Optional[str] = None, section_filter: Optional[list[str]] = None) -> list[dict]:
        """Search chunks by vector similarity (cosine distance)."""
        self._setup_schema()

        emb_str = _SQLITE_VEC.serialize_float32(query_vector)

        conditions = []
        params = []

        if source_doc_id:
            conditions.append("source_doc_id = ?")
            params.append(source_doc_id)

        if section_filter:
            # Use exact JSON array membership via json_each — no LIKE, so no
            # wildcard/escape issues with special characters in section names.
            for s in section_filter:
                conditions.append(
                    "c.id IN (SELECT c2.id FROM chunks c2 JOIN json_each(c2.section_path) ON json_each.value = ? WHERE c2.source_doc_id = c.source_doc_id)"
                )
                params.append(s)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = (f"""SELECT c.id, c.text, json_each.value as section_path, 
                         c.source_doc_id, c.chunk_index, c.word_count
                 FROM chunks c, json_each(c.section_path)
                 {where}
                 ORDER BY vec_distance_cosine(c.embedding, ?) ASC
                 LIMIT ?""")

        results = self.conn.execute(sql, params + [emb_str, k]).fetchall()

        return [{
            "id": row[0],
            "text": row[1],
            "section_path": self._parse_section_path(row[2]),
            "source_doc_id": row[3],
            "chunk_index": row[4],
            "word_count": row[5],
        } for row in results]

    def upsert_document(self, title: str, tags: list[str], text_summary: str,
                        source_file: str, total_chunks: int,
                        embedding: Optional[list[float]] = None) -> dict:
        """Insert or update a document-level record (B index)."""
        self._setup_schema()

        tags_json = json.dumps(tags)
        created_at = datetime.now(timezone.utc).isoformat()

        # Upsert by source_file — re-ingesting the same file overwrites.
        existing = self.conn.execute(
            "SELECT id FROM documents WHERE source_file=?", (source_file,)
        ).fetchone()

        if existing:
            cursor = self.conn.execute(
                """UPDATE documents SET title=?, tags=?, text_summary=?, 
                  total_chunks=?, created_at=? WHERE id=?""",
                (title, tags_json, text_summary[:1000], total_chunks, created_at, existing[0])
            )
        else:
            cursor = self.conn.execute(
                """INSERT INTO documents (title, tags, text_summary, source_file, 
                                          total_chunks, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (title, tags_json, text_summary[:1000], source_file, total_chunks, created_at)
            )

        self.conn.commit()

        doc_id = existing[0] if existing else cursor.lastrowid or 1
        return {
            "id": doc_id,
            "title": title,
            "tags": json.loads(tags_json),
            "text_summary": text_summary,
            "source_file": source_file,
            "total_chunks": total_chunks,
            "created_at": created_at,
        }

    def search_documents(self, query_vector: Optional[list[float]] = None,
                         tags: Optional[list[str]] = None) -> list[dict]:
        """Search documents by vector similarity or tag filter."""
        self._setup_schema()

        where_clauses = []
        params = []

        if tags:
            # Exact match via json_each — avoids LIKE wildcard issues entirely.
            for i, tag in enumerate(tags):
                where_clauses.append(
                    "EXISTS (SELECT 1 FROM json_each(documents.tags) WHERE json_each.value = ?)"
                )
                params.append(tag)

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        results = self.conn.execute(
            f"""SELECT id, title, tags as raw_tags, 
                         text_summary, source_file, total_chunks, created_at
                 FROM documents {where}""",
            params
        ).fetchall()

        return [{
            "id": row[0],
            "title": row[1],
            "tags": json.loads(row[2]) if isinstance(row[2], str) else [],
            "text_summary": row[3],
            "source_file": row[4],
            "total_chunks": row[5],
            "created_at": row[6],
        } for row in results]

    def hybrid_search(self, query_text: str, query_vector: Optional[list[float]] = None,
                      k: int = 10) -> list[dict]:
        """Hybrid search: vector similarity + full-text (FTS5)."""
        self._setup_schema()

        # Empty queries fall back to pure vector search if a vector is provided.
        if not query_text.strip():
            if query_vector:
                emb_str = _SQLITE_VEC.serialize_float32(query_vector)
                results = self.conn.execute(
                    """SELECT c.id, c.text, json_each.value as section_path, 
                                 c.source_doc_id, c.chunk_index, c.word_count
                         FROM chunks c, json_each(c.section_path)
                         ORDER BY vec_distance_cosine(embedding, ?) ASC
                         LIMIT ?""",
                    [emb_str, k]
                ).fetchall()

                return [{
                    "id": row[0],
                    "text": row[1],
                    "section_path": self._parse_section_path(row[2]),
                    "source_doc_id": row[3],
                    "chunk_index": row[4],
                    "word_count": row[5],
                } for row in results]
            return []

        fts_results = self.conn.execute(
            "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            (query_text, k)
        ).fetchall()

        vec_results = []
        if query_vector:
            emb_str = _SQLITE_VEC.serialize_float32(query_vector)
            results = self.conn.execute(
                """SELECT c.id, c.text, json_each.value as section_path, 
                             c.source_doc_id, c.chunk_index, c.word_count
                     FROM chunks c, json_each(c.section_path)
                     ORDER BY vec_distance_cosine(embedding, ?) ASC
                     LIMIT ?""",
                [emb_str, k]
            ).fetchall()

            for row in results:
                vec_results.append({
                    "id": row[0],
                    "text": row[1],
                    "section_path": self._parse_section_path(row[2]),
                    "source_doc_id": row[3],
                    "chunk_index": row[4],
                    "word_count": row[5],
                })

        combined = {}

        # Build lookup maps for FTS results by rowid
        fts_map = {r[0]: r[1] for r in fts_results}

        if vec_results:
            for v in vec_results:
                if v["id"] not in combined:
                    combined[v["id"]] = dict(v)

            # Compute cosine distances once per combined ID (batched).
            ids_to_score = [v["id"] for v in vec_results]
            score_map = {}
            if ids_to_score:
                placeholders = ",".join("?" * len(ids_to_score))
                emb_str = _SQLITE_VEC.serialize_float32(query_vector)
                for row in self.conn.execute(
                    f"SELECT id, vec_distance_cosine(embedding, ?) FROM chunks WHERE id IN ({placeholders})",
                    [emb_str] + ids_to_score,
                ).fetchall():
                    score_map[row[0]] = row[1]

            # Reciprocal Rank Fusion with proper rank assignment.
            rrf_k = 60
            total_results = max(len(fts_map), len(vec_results), 1)

            result_list = []
            for doc_id, data in combined.items():
                fts_rank = fts_map.get(doc_id, total_results + 1)
                # Proper rank: sort all vec results by distance to assign ranks.
                sorted_by_dist = sorted(vec_results, key=lambda x: score_map.get(x["id"], 2.0))
                vec_rank = next(
                    (i + 1 for i, v in enumerate(sorted_by_dist) if v["id"] == data["id"]),
                    total_results + 1,
                )
                rrf_score = (1.0 / (fts_rank + rrf_k)) + (1.0 / (vec_rank + rrf_k))
                result_list.append({k: v for k, v in data.items() if not k.startswith("_")} | {"_rrf_score": rrf_score})

            return sorted(result_list, key=lambda x: -x["_rrf_score"])[:k]
        else:
            # Text-only query (no vector): fetch all chunk details in one JOIN.
            if fts_results:
                ids = [r[0] for r in fts_results]
                placeholders = ",".join("?" * len(ids))
                rows = self.conn.execute(
                    f"""SELECT id, text, section_path, source_doc_id, chunk_index, word_count 
                        FROM chunks WHERE id IN ({placeholders})""",
                    ids,
                ).fetchall()
                return [{
                    "id": row[0],
                    "text": row[1],
                    "section_path": self._parse_section_path(row[2]),
                    "source_doc_id": row[3],
                    "chunk_index": row[4],
                    "word_count": row[5],
                } for row in rows][:k]
            return []

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

    def close(self):
        """Close the database connection safely."""
        try:
            self.conn.commit()
        except Exception as exc:  # noqa: BLE001 — best-effort commit on close
            logger.debug("Commit during close failed: %s", exc)
        finally:
            if not self.conn.in_transaction:
                pass  # already committed or no transaction open
            try:
                self.conn.close()
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Close failed: %s", exc)


def _deserialize_embedding(raw) -> list[float]:
    """Deserialize embedding from BLOB (sqlite_vec format) or legacy string.

    Module-level so it isn't re-created on every ``get_chunks_by_doc`` call.
    """
    if isinstance(raw, bytes):
        n = len(raw) // 4
        return list(struct.unpack(f"{n}f", raw))
    # Legacy comma-separated string format
    if isinstance(raw, str):
        return [float(v) for v in raw.split(",")]
    return list(raw)
