"""Search operations: search_chunks, search_documents, hybrid_search, get_embeddings_by_ids."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .schema import _CJK_RANGE, _SQLITE_VEC

logger = logging.getLogger(__name__)

# Characters with special meaning in FTS5 MATCH query syntax. Left in the
# query string they are parsed as operators (e.g. "-" → AND NOT) and can raise
# "no such column" errors. We strip them before querying.
# Characters with special meaning in FTS5 MATCH query syntax that must be stripped:
#   "..." phrase match, ()  grouping, - AND NOT, [] {} <> / ~ ? ! .
_FTS_SPECIAL = re.compile(r"""["*^:()\\[\]{}<>/~?!.]+""")


def _deserialize_embedding(raw) -> list[float]:
    """Deserialize embedding from BLOB (sqlite_vec format) or legacy string.

    Module-level so it isn't re-created on every ``get_chunks_by_doc`` call.
    """
    import struct
    if isinstance(raw, bytes):
        n = len(raw) // 4
        return list(struct.unpack(f"{n}f", raw))
    # Legacy comma-separated string format
    if isinstance(raw, str):
        return [float(v) for v in raw.split(",")]
    return list(raw)


# Pre-compile CJK pattern once at module level.
_CJK_PAT = re.compile("|".join(_CJK_RANGE))


def _build_fts_query(text: str) -> str | None:
    """Turn free-text into an FTS5-safe MATCH query.

    Strips all recognized FTS5 operator characters, then OR-joins the surviving tokens so any
    query term can contribute to the fused score (recall-friendly). The CJK character class covers Blocks A–D; the basic range alone misses ~1% of modern Chinese text in Extensions E/F/G which are rarely used outside specialized domains. Returns
    None when no usable token remains, so callers can fall back to vector-only.
    """
    cleaned = _FTS_SPECIAL.sub(" ", text)
    cjk_re_str = "|".join(_CJK_RANGE)
    tokens = re.findall(r"[A-Za-z0-9]+|" + cjk_re_str, cleaned)
    tokens = [t for t in tokens if len(t) > 1 or bool(_CJK_PAT.match(t))]
    if not tokens:
        return None
    return " OR ".join(tokens)


def _parse_section_path(raw: str) -> list[str]:
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


class _SearchOps:
    """Handles all search/query operations."""

    def _parse_section_path(self, raw: str) -> list[str]:
        """Parse section path from JSON or empty string."""
        return _parse_section_path(raw)

    def search_chunks(self, query_vector: list[float], *, k: int = 10,
                      source_doc_id: str | None = None, section_filter: list[str] | None = None) -> list[dict]:
        """Search chunks by vector similarity (cosine distance)."""
        self._setup_schema()

        emb_str = _SQLITE_VEC.serialize_float32(query_vector)

        conditions = []
        params = []

        if source_doc_id:
            conditions.append("source_doc_id = ?")
            params.append(source_doc_id)

        if section_filter:
            # Exact JSON array membership via json_each (EXISTS, no JOIN) — no
            # row multiplication, and no LIKE wildcard/escape issues with
            # special characters in section names.
            for s in section_filter:
                conditions.append(
                    "EXISTS (SELECT 1 FROM json_each(c.section_path) WHERE json_each.value = ?)"
                )
                params.append(s)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = (f"""SELECT c.id, c.text, c.section_path,
                          c.source_doc_id, c.chunk_index, c.word_count
                  FROM chunks c
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

    def search_documents(self, query_vector: list[float] | None = None,
                         tags: list[str] | None = None, k: int = 5) -> list[dict]:
        """Search documents by vector similarity or tag filter.

        When *query_vector* is provided, ranks documents by cosine distance
        (coarse-grained B index). *tags* filters by exact tag membership.
        Both can be combined (vector ranking within the tag-filtered set).
        """
        self._setup_schema()

        where_clauses = []
        params = []

        if tags:
            # Exact match via json_each — avoids LIKE wildcard issues entirely.
            for tag in tags:
                where_clauses.append(
                    "EXISTS (SELECT 1 FROM json_each(documents.tags) WHERE json_each.value = ?)"
                )
                params.append(tag)

        # Vector search requires a non-null embedding column.
        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        if query_vector is not None:
            where_clauses.append("embedding IS NOT NULL")
            where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            emb_str = _SQLITE_VEC.serialize_float32(query_vector)
            sql = (
                f"""SELECT id, title, tags as raw_tags, text_summary,
                              source_file, total_chunks, created_at,
                              vec_distance_cosine(embedding, ?) AS distance
                       FROM documents {where}
                       ORDER BY distance ASC
                       LIMIT ?"""
            )
            rows = self.conn.execute(sql, [emb_str, *params, k]).fetchall()
        else:
            sql = (
                f"""SELECT id, title, tags as raw_tags, text_summary,
                              source_file, total_chunks, created_at
                       FROM documents {where}"""
            )
            rows = self.conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "title": row[1],
                "tags": json.loads(row[2]) if isinstance(row[2], str) else [],
                "text_summary": row[3],
                "source_file": row[4],
                "total_chunks": row[5],
                "created_at": row[6],
                **({"distance": row[7]} if query_vector is not None else {}),
            })
        return results

    def hybrid_search(self, query_text: str, query_vector: list[float] | None = None,
                      k: int = 10) -> list[dict]:
        """Hybrid search: vector similarity + full-text (FTS5)."""
        self._setup_schema()

        # Empty queries fall back to pure vector search if a vector is provided.
        if not query_text.strip():
            if query_vector:
                emb_str = _SQLITE_VEC.serialize_float32(query_vector)
                results = self.conn.execute(
                    """SELECT c.id, c.text, c.section_path,
                                  c.source_doc_id, c.chunk_index, c.word_count
                           FROM chunks c
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

        fts_query = _build_fts_query(query_text)
        fts_results: list = []
        if fts_query is not None:
            fts_results = self.conn.execute(
                "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
                (fts_query, k)
            ).fetchall()

        vec_results = []
        if query_vector:
            emb_str = _SQLITE_VEC.serialize_float32(query_vector)
            results = self.conn.execute(
                """SELECT c.id, c.text, c.section_path,
                                 c.source_doc_id, c.chunk_index, c.word_count
                          FROM chunks c
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

        # Build FTS rank map by 1-based result order (bm25 score is NOT a rank).
        fts_rank_map = {rowid: i + 1 for i, (rowid, _rank) in enumerate(fts_results)}

        if vec_results:
            for v in vec_results:
                if v["id"] not in combined:
                    combined[v["id"]] = dict(v)

            # Compute cosine distances once per combined ID (batched).
            # Cap at 1000 to stay under SQLite's SQLITE_MAX_VARIABLE_NUMBER.
            _MAX_IN_CLAUSE = 1000
            ids_to_score = [v["id"] for v in vec_results[:_MAX_IN_CLAUSE]]
            score_map = {}
            if ids_to_score:
                placeholders = ",".join("?" * len(ids_to_score))
                emb_str = _SQLITE_VEC.serialize_float32(query_vector)
                for row in self.conn.execute(
                    f"SELECT id, vec_distance_cosine(embedding, ?) FROM chunks WHERE id IN ({placeholders})",
                    [emb_str] + ids_to_score,
                ).fetchall():
                    score_map[row[0]] = row[1]

            # Reciprocal Rank Fusion with proper integer rank assignment.
            # Sort all vector results by distance ONCE (rank = 1 for nearest).
            sorted_by_dist = sorted(vec_results, key=lambda x: score_map.get(x["id"], 2.0))
            vec_rank_map = {v["id"]: i + 1 for i, v in enumerate(sorted_by_dist)}

            rrf_k = 60
            total_results = max(len(fts_rank_map), len(vec_results), 1)

            result_list = []
            for doc_id, data in combined.items():
                fts_rank = fts_rank_map.get(doc_id, total_results + 1)
                vec_rank = vec_rank_map.get(doc_id, total_results + 1)
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

    def get_embeddings_by_ids(self, ids: list[int]) -> list[list[float]]:
        """Return chunk embeddings aligned to *ids* (document-side vectors).

        Used for MMR re-ranking so we don't re-embed chunks that are already
        stored in the index. Unknown ids yield an empty list (kept in order);
        callers may fall back to embedding those texts on demand.
        """
        self._setup_schema()
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, embedding FROM chunks WHERE id IN ({placeholders})", ids
        ).fetchall()
        by_id = {row[0]: _deserialize_embedding(row[1]) for row in rows}
        return [by_id.get(i, []) for i in ids]
