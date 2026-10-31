"""Tests for vector search, chunk retrieval, and text-only ranking."""

from storage.tests.conftest import _make_embedding


class TestSearchChunks:
    """Test vector similarity search on chunks."""

    def test_search_returns_results(self, store):
        """Vector search should return nearest neighbors."""
        emb = _make_embedding()
        store.upsert_chunk(
            {"text": "Relevant content about AI", "section_path": ["AI"]},
            doc_id="doc_search", embedding=emb, chunk_index=0,
        )
        # Same embedding → should find the chunk
        results = store.search_chunks(emb, k=10, source_doc_id="doc_search")
        assert len(results) >= 1
        assert results[0]["text"] == "Relevant content about AI"

    def test_search_with_source_filter(self, store):
        """Filter by source_doc_id."""
        emb = _make_embedding()
        store.upsert_chunk(
            {"text": "Doc A chunk", "section_path": ["A"]},
            doc_id="doc_a", embedding=emb, chunk_index=0,
        )
        store.upsert_chunk(
            {"text": "Doc B chunk", "section_path": ["B"]},
            doc_id="doc_b", embedding=emb, chunk_index=0,
        )

        results = store.search_chunks(emb, k=10, source_doc_id="doc_a")
        assert len(results) == 1
        assert results[0]["source_doc_id"] == "doc_a"


class TestGetChunksByDoc:
    """Test chunk retrieval and embedding deserialization."""

    def test_get_chunks_round_trip_embedding(self, store):
        """Embedding stored as BLOB must deserialize correctly via get_chunks_by_doc()."""
        emb = _make_embedding()
        store.upsert_chunk(
            {"text": "Round-trip test", "section_path": ["RT"]},
            doc_id="doc_rt", embedding=emb, chunk_index=0,
        )

        chunks = store.get_chunks_by_doc("doc_rt")
        assert len(chunks) == 1
        retrieved_emb = chunks[0]["embedding"]

        assert isinstance(retrieved_emb, list), f"Expected list, got {type(retrieved_emb)}"
        assert len(retrieved_emb) == 1024, f"Expected 1024 dims, got {len(retrieved_emb)}"
        # Values should be close (float32 precision loss is expected)
        for orig, retrieved in zip(emb, retrieved_emb, strict=True):
            assert abs(orig - retrieved) < 0.001, \
                f"Embedding mismatch: orig={orig}, retrieved={retrieved}"

    def test_get_chunks_by_doc_order(self, store):
        """Chunks should be returned ordered by chunk_index."""
        for i in range(3):
            emb = _make_embedding()
            store.upsert_chunk(
                {"text": f"Chunk {i}", "section_path": ["Order"]},
                doc_id="doc_order", embedding=emb, chunk_index=i,
            )

        chunks = store.get_chunks_by_doc("doc_order")
        indices = [c["chunk_index"] for c in chunks]
        assert indices == sorted(indices), "Chunks not ordered by chunk_index"

    def test_get_chunks_empty_doc(self, store):
        """Non-existent doc should return empty list."""
        chunks = store.get_chunks_by_doc("nonexistent")
        assert chunks == []


class TestTextOnlySearchRanking:
    """Text-only hybrid_search must return chunks in BM25 rank order."""

    def test_text_only_results_follow_fts_rank(self, store):
        """Regression test for FTS rank order in text-only search.

        The detail fetch (WHERE id IN ...) used to discard the FTS rank
        order, returning top-k in arbitrary scan order.
        """
        emb = _make_embedding()
        # Filler docs give query terms positive IDF; each shares at most one
        # term with the query.
        fillers = [
            "cooking pasta requires salt",
            "mountain hiking trails overview",
            "classical music composers list",
            "gardening tips for spring soil",
            "ocean wildlife documentary series",
        ]
        for i, text in enumerate(fillers):
            store.upsert_chunk(
                {"text": text, "section_path": ["S"]},
                doc_id=f"doc_fill{i}", embedding=emb, chunk_index=0,
            )
        # Weak match inserted second-to-last; strong match LAST so scan order
        # (rowid) is the reverse of relevance order.
        store.upsert_chunk(
            {"text": "alpha retrieval study", "section_path": ["S"]},
            doc_id="doc_weak", embedding=emb, chunk_index=0,
        )
        store.upsert_chunk(
            {"text": "alpha beta gamma delta retrieval pipeline evaluation",
             "section_path": ["S"]},
            doc_id="doc_strong", embedding=emb, chunk_index=0,
        )

        results = store.hybrid_search("alpha beta gamma delta retrieval")
        assert len(results) >= 2
        # The chunk containing all query terms must rank first — under the old
        # bug it came back near the end (highest rowid, scan order).
        assert "alpha beta gamma delta" in results[0]["text"]

    def test_text_only_results_truncated_to_k(self, store):
        for i in range(5):
            store.upsert_chunk(
                {"text": f"keyword alpha_{i}", "section_path": ["S"]},
                doc_id=f"doc_k{i}", embedding=_make_embedding(), chunk_index=0,
            )
        results = store.hybrid_search("keyword", k=3)
        assert len(results) == 3
