"""Tests for hybrid search and RRF fusion."""

from storage.tests.conftest import _make_embedding


class TestHybridSearch:
    """Test hybrid vector + FTS5 search."""

    def test_hybrid_search_returns_results(self, store):
        """Hybrid search should return results when vector is provided."""
        emb = _make_embedding()
        store.upsert_chunk(
            {"text": "Machine learning is fascinating", "section_path": ["ML"]},
            doc_id="doc_hybrid", embedding=emb, chunk_index=0,
        )
        results = store.hybrid_search("machine learning", query_vector=emb, k=10)
        assert len(results) >= 1


class TestHybridRRF:
    """Verify hybrid search uses Reciprocal Rank Fusion (RRF) correctly."""

    def test_hybrid_ranks_by_combined_score(self, store):
        """Chunks appearing in both vector and FTS results should rank higher."""
        emb = _make_embedding()
        # Chunk that matches BOTH text and is "close" to the query embedding.
        store.upsert_chunk(
            {"text": "Hybrid RRF ranking algorithm implementation", "section_path": ["RRF"]},
            doc_id="doc_rrf", embedding=emb, chunk_index=0,
        )

        results = store.hybrid_search("hybrid ranking")
        assert len(results) >= 1
        top_text = results[0]["text"]
        # The text containing the query terms should rank first.
        assert "Hybrid" in top_text or "ranking" in top_text.lower()

    def test_hybrid_empty_query_fallback_to_vector(self, store):
        """hybrid_search with empty query falls back to pure vector search."""
        emb = _make_embedding()
        store.upsert_chunk(
            {"text": "Empty query fallback", "section_path": ["EQ"]},
            doc_id="doc_eq", embedding=emb, chunk_index=0,
        )

        # Empty text + provided vector → pure vector search.
        results = store.hybrid_search("", query_vector=emb)
        assert len(results) >= 1

    def test_hybrid_search_sanitizes_fts_special_chars(self, store):
        """Hyphens / FTS5 operators in the query must not crash the search.

        Regression test: a query like 'retrieval-augmented generation' used to
        raise sqlite3.OperationalError: no such column: augmented because FTS5
        parsed the hyphen as an operator.
        """
        emb = _make_embedding()
        store.upsert_chunk(
            {"text": "Retrieval-augmented generation combines retrieval with generation",
             "section_path": ["RAG"]},
            doc_id="doc_hyphen", embedding=emb, chunk_index=0,
        )
        store.upsert_chunk(
            {"text": "Unrelated content about cooking pasta", "section_path": ["Food"]},
            doc_id="doc_hyphen", embedding=_make_embedding(), chunk_index=1,
        )

        # Should not raise; should return the matching chunk.
        results = store.hybrid_search("What is retrieval-augmented generation?")
        assert any("Retrieval-augmented" in r["text"] for r in results)

    def test_build_fts_query_strips_special_chars(self):
        """_build_fts_query returns FTS5-safe OR-joined tokens (or None)."""
        from storage.sqlite_vec import _build_fts_query

        assert _build_fts_query("retrieval-augmented generation") == "retrieval OR augmented OR generation"
        assert _build_fts_query("") is None
        assert _build_fts_query("   ") is None
        # Parentheses / quotes / colons are stripped, not treated as operators.
        fts_query = _build_fts_query('RAG: "what is this" (explained)')
        assert fts_query is not None
        assert "OR" in fts_query
