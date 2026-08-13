"""Insert operations: upsert_chunk, upsert_chunks, upsert_document."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from .schema import _CJK_RANGE, _SQLITE_VEC

logger = logging.getLogger(__name__)

# Pre-compile CJK regex once at module level for performance.
_CJK_RE = re.compile("|".join(_CJK_RANGE))


def _count_words(text: str) -> int:
    """Count words, treating CJK characters as individual tokens.

    Uses ``re.sub`` to remove CJK in one pass, then counts Latin tokens
    via ``\\S+`` splitting. Each CJK character counts individually since
    there are no word boundaries.
    """
    if not text:
        return 0
    cjk_count = len(_CJK_RE.findall(text))
    non_cjk = re.sub(_CJK_RE, '', text)
    return cjk_count + len(re.findall(r"\S+", non_cjk))


class _InsertOps:
    """Handles all insert/upsert operations for chunks and documents."""

    def _setup_schema(self) -> None:
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
                source_file TEXT NOT NULL UNIQUE,
                total_chunks INTEGER DEFAULT 0,
                embedding BLOB,
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
        # Backward-compat migration: existing DBs created before the document-level
        # embedding column existed won't have it. Add it without dropping data.
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(documents)")}
        if "embedding" not in cols:
            self.conn.execute("ALTER TABLE documents ADD COLUMN embedding BLOB")
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
        word_count = _count_words(chunk_data.get("text", ""))

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
        """Batch insert or replace multiple chunks in a single transaction.

        Uses ``executemany`` so a whole document's chunks are written in one
        round-trip, then the generated row ids are fetched back by
        (source_doc_id, chunk_index) to build the result list.
        """
        self._setup_schema()

        if not chunks:
            return []

        params = [
            (
                chunk["text"],
                _SQLITE_VEC.serialize_float32(chunk.get("embedding", [0.0])),
                doc_id,
                i,
                json.dumps(chunk.get("section_path", ["General"])),
                _count_words(chunk.get("text", "")),
                json.dumps(chunk.get("entity_names", [])),
            )
            for i, chunk in enumerate(chunks)
        ]

        self.conn.executemany(
            """INSERT OR REPLACE INTO chunks (text, embedding, source_doc_id, chunk_index, section_path, word_count, entity_names)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            params,
        )
        self.conn.commit()

        id_by_index = {
            idx: cid for cid, idx in self.conn.execute(
                "SELECT id, chunk_index FROM chunks WHERE source_doc_id=?", (doc_id,)
            ).fetchall()
        }

        return [
            {
                "id": id_by_index.get(i),
                "text": chunk["text"],
                "section_path": chunk.get("section_path", ["General"]),
                "source_doc_id": doc_id,
                "chunk_index": i,
                "word_count": _count_words(chunk.get("text", "")),
            }
            for i, chunk in enumerate(chunks)
        ]

    def upsert_document(self, title: str, tags: list[str], text_summary: str,
                        source_file: str, total_chunks: int,
                        embedding: Optional[list[float]] = None) -> dict:
        """Insert or update a document-level record (B index).

        Persists the document-level embedding so the coarse-grained B index
        can be searched by vector similarity, not just by tags.
        """
        self._setup_schema()

        tags_json = json.dumps(tags)
        created_at = datetime.now(timezone.utc).isoformat()
        emb_blob = _SQLITE_VEC.serialize_float32(embedding) if embedding else None

        # Upsert by source_file using ON CONFLICT for atomicity.
        self.conn.execute(
            """INSERT INTO documents (title, tags, text_summary, source_file, 
                                      total_chunks, embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_file) DO UPDATE SET
                   title=excluded.title,
                   tags=excluded.tags,
                   text_summary=excluded.text_summary,
                   total_chunks=excluded.total_chunks,
                   embedding=excluded.embedding,
                   created_at=excluded.created_at""",
            (title, tags_json, text_summary[:1000], source_file, total_chunks, emb_blob, created_at)
        )
        self.conn.commit()

        doc_id = self.conn.execute(
            "SELECT id FROM documents WHERE source_file=?", (source_file,)
        ).fetchone()[0]
        return {
            "id": doc_id,
            "title": title,
            "tags": json.loads(tags_json),
            "text_summary": text_summary,
            "source_file": source_file,
            "total_chunks": total_chunks,
            "embedding": embedding,
            "created_at": created_at,
        }
