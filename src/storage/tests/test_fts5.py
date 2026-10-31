"""Tests for FTS5 full-text search integration."""

from storage.tests.conftest import _make_embedding


class TestFTS5Search:
    """Functional tests for SQLiteVecStore FTS5 integration."""

    def test_fts5_sync_on_insert(self, store):
        """Chunks inserted via upsert_chunk should appear in FTS index."""
        emb = _make_embedding()
        store.upsert_chunk(
            {"text": "Full text search indexing", "section_path": ["FTS"]},
            doc_id="doc_fts", embedding=emb, chunk_index=0,
        )

        # Verify FTS5 table has the entry via MATCH query (rowid not id).
        rows = store.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'indexing'"
        ).fetchall()
        assert len(rows) >= 1

    def test_fts5_sync_on_delete(self, store):
        """Deleting a chunk should remove its FTS entry."""
        emb = _make_embedding()
        upsert_result = store.upsert_chunk(
            {"text": "FTS deletion test content", "section_path": ["Del"]},
            doc_id="doc_del", embedding=emb, chunk_index=0,
        )

        # Delete the chunk.
        store.conn.execute("DELETE FROM chunks WHERE id = ?", (upsert_result["id"],))

        rows = store.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'deletion'"
        ).fetchall()
        assert len(rows) == 0, "FTS entry should be removed after chunk deletion"

    def test_fts5_full_text_search_function(self, store):
        """hybrid_search with text-only query (no vector) uses FTS."""
        emb = _make_embedding()
        store.upsert_chunk(
            {"text": "Python programming language features", "section_path": ["Py"]},
            doc_id="doc_py_fts", embedding=emb, chunk_index=0,
        )

        results = store.hybrid_search("python programming")
        assert len(results) >= 1
        assert any("Python" in r["text"] for r in results)
