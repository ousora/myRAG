"""SQLite-based vector store using sqlite-vec extension."""


import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class SQLiteVecStore:
    """SQLite-backed vector store using sqlite-vec extension."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        import sqlite3
        import sqlite_vec
        
        conn = sqlite3.connect(self.db_path, timeout=10)
        sqlite_vec.load(conn)
        
        self.conn = conn
        self.conn.execute("PRAGMA journal_mode=WAL")

    def _setup_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source_doc_id TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                section_path TEXT,
                word_count INTEGER
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
        """)

    def upsert_chunk(self, chunk_data: dict, *, doc_id: str, 
                     embedding: list[float], chunk_index: int = 0) -> dict:
        """Insert or update a chunk with its embedding."""
        import sqlite_vec
        
        self._setup_schema()
        
        section_json = json.dumps(chunk_data.get("section_path", ["General"]))
        word_count = len(chunk_data.get("text", "").split())

        cursor = self.conn.execute(
            """INSERT INTO chunks (text, embedding, source_doc_id, chunk_index, section_path, word_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (chunk_data["text"], sqlite_vec.serialize_float32(embedding), doc_id, 
             chunk_index, json.dumps(chunk_data.get("section_path", ["General"])), word_count)
        )

        self.conn.execute(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
            (cursor.lastrowid, chunk_data["text"])
        )

        return {
            "id": cursor.lastrowid,
            "text": chunk_data["text"],
            "section_path": json.loads(section_json),
            "source_doc_id": doc_id,
            "chunk_index": chunk_index,
            "word_count": word_count,
        }

    def upsert_chunks(self, chunks: list[dict], *, doc_id: str) -> list[dict]:
        """Batch insert multiple chunks."""
        results = []
        for i, chunk in enumerate(chunks):
            result = self.upsert_chunk(chunk, doc_id=doc_id, embedding=chunk["embedding"], 
                                       chunk_index=i)
            results.append(result)
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
        except (json.JSONDecodeError, Exception):
            return []

    def search_chunks(self, query_vector: list[float], *, k: int = 10,
                      source_doc_id: Optional[str] = None, section_filter: Optional[list[str]] = None) -> list[dict]:
        """Search chunks by vector similarity (cosine distance)."""
        import sqlite_vec
        
        self._setup_schema()

        emb_str = sqlite_vec.serialize_float32(query_vector)
        
        conditions = []
        params = []
        
        if source_doc_id:
            conditions.append("source_doc_id = ?")
            params.append(source_doc_id)
            
        if section_filter:
            for s in section_filter:
                conditions.append(
                    "json_extract(section_path, '$') LIKE ?"
                )
                params.append(json.dumps(s))

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

        cursor = self.conn.execute(
            """INSERT INTO documents (title, tags, text_summary, source_file, 
                                      total_chunks, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (title, tags_json, text_summary[:1000], source_file, total_chunks, created_at)
        )
        self.conn.commit()

        return {
            "id": cursor.lastrowid or 1,
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
            tag_conditions = " OR ".join([f'json_extract(tags, "$[{i}]") LIKE ?' for i in range(len(tags))])
            where_clauses.append(f"({tag_conditions})")
            params.extend(tags)

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        results = self.conn.execute(
            f"""SELECT id, title, json_extract(tags, '$') as tags, 
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
        import sqlite_vec
        
        self._setup_schema()

        fts_results = self.conn.execute(
            "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            (query_text, k)
        ).fetchall()
        
        vec_results = []
        if query_vector:
            emb_str = sqlite_vec.serialize_float32(query_vector)
            results = self.conn.execute(
                f"""SELECT c.id, c.text, json_each.value as section_path, 
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
        
        for r in fts_results:
            if r[0] not in combined:
                combined[r[0]] = {"text": None, "section_path": [], "source_doc_id": "", 
                                   "chunk_index": 0, "word_count": 0, "_fts_rank": r[1]}

        for v in vec_results:
            if v["id"] not in combined:
                combined[v["id"]] = dict(v)
                emb_str = sqlite_vec.serialize_float32(query_vector)
                combined[v["id"]]["_vec_score"] = self.conn.execute(
                    f"SELECT vec_distance_cosine(embedding, ?) FROM chunks WHERE id=?",
                    [emb_str, v["id"]]
                ).fetchone()[0]

        result_list = list(combined.values())
        
        return sorted(result_list, key=lambda x: (x.get("_fts_rank", 999), x.get("_vec_score", 1.0)))[:k]

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
            "embedding": [float(v) for v in row[2].split(",")] if isinstance(row[2], str) else list(row[2]),
            "source_doc_id": row[3],
            "chunk_index": row[4],
            "section_path": self._parse_section_path(row[5]) if row[5] else ["General"],
            "word_count": row[6],
        } for row in results]

    def close(self):
        """Close the database connection."""
        self.conn.commit()
        self.conn.close()
