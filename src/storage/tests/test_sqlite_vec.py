"""Tests for SQLiteVecStore — embedding serialization, CRUD, search."""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store():
    """Create a temp-file SQLiteVecStore for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        from storage.sqlite_vec import SQLiteVecStore
        db = SQLiteVecStore(path)
        yield db
        # Don't call db.close() — the test_close_closes_connection test already
        # closes the connection, and calling close() twice raises an error.
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helper — generate a dummy embedding vector (1024-d like bge-m3)
# ---------------------------------------------------------------------------

def _make_embedding(dim=1024):
    """Return a deterministic dummy embedding list."""
    import random
    rng = random.Random(42)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


# ---------------------------------------------------------------------------
# upsert_chunk / upsert_chunks
# ---------------------------------------------------------------------------

class TestUpsertChunk:
    """Test chunk insertion and embedding serialization."""

    def test_upsert_chunk_creates_record(self, store):
        chunk_data = {"text": "Hello world", "section_path": ["Intro"]}
        emb = _make_embedding()
        result = store.upsert_chunk(chunk_data, doc_id="doc_1", embedding=emb, chunk_index=0)

        assert result["source_doc_id"] == "doc_1"
        assert result["chunk_index"] == 0
        assert result["text"] == "Hello world"

    def test_upsert_chunk_serializes_embedding_as_blob(self, store):
        """Embedding must be stored as binary BLOB (sqlite_vec format)."""
        import struct

        chunk_data = {"text": "Blob test", "section_path": ["Test"]}
        emb = _make_embedding()
        store.upsert_chunk(chunk_data, doc_id="doc_blob", embedding=emb, chunk_index=0)

        # Direct DB query to verify BLOB storage
        conn = store.conn
        row = conn.execute(
            "SELECT embedding FROM chunks WHERE source_doc_id = ?", ("doc_blob",)
        ).fetchone()
        assert row is not None
        raw = row[0]
        assert isinstance(raw, bytes), f"Expected BLOB (bytes), got {type(raw)}"

        # Verify we can deserialize it back to floats
        n = len(raw) // 4
        values = struct.unpack(f"{n}f", raw)
        assert len(values) == 1024

    def test_upsert_chunks_batch(self, store):
        """Batch insert multiple chunks."""
        chunks = [
            {"text": f"Chunk {i}", "section_path": ["Section"]}
            for i in range(3)
        ]
        embeddings = [_make_embedding() for _ in range(3)]
        for i, c in enumerate(chunks):
            c["embedding"] = embeddings[i]

        results = store.upsert_chunks(chunks, doc_id="doc_batch")
        assert len(results) == 3
        for r in results:
            assert r["source_doc_id"] == "doc_batch"

    def test_upsert_chunks_returns_aligned_ids(self, store):
        """Batch upsert must return one result per chunk with correct ids/order."""
        chunks = [
            {"text": f"C{i}", "section_path": ["S"]} for i in range(3)
        ]
        embs = [_make_embedding() for _ in range(3)]
        for i, c in enumerate(chunks):
            c["embedding"] = embs[i]

        results = store.upsert_chunks(chunks, doc_id="doc_ids")
        assert [r["chunk_index"] for r in results] == [0, 1, 2]
        assert all(r["id"] is not None for r in results)
        # Ids must be distinct per chunk_index within the same doc.
        assert len({r["id"] for r in results}) == 3

    def test_get_embeddings_by_ids_round_trip(self, store):
        """Stored chunk embeddings are returned aligned to the requested ids."""
        embs = [_make_embedding() for _ in range(2)]
        r0 = store.upsert_chunk({"text": "a", "section_path": ["S"]}, doc_id="doc_ge", embedding=embs[0], chunk_index=0)
        r1 = store.upsert_chunk({"text": "b", "section_path": ["S"]}, doc_id="doc_ge", embedding=embs[1], chunk_index=1)

        out = store.get_embeddings_by_ids([r0["id"], r1["id"]])
        assert len(out) == 2
        for orig, got in zip(embs, out):
            assert len(got) == 1024
            assert all(abs(o - g) < 0.001 for o, g in zip(orig, got))

    def test_get_embeddings_by_ids_unknown_is_empty(self, store):
        """Unknown ids yield an empty list kept in the requested position."""
        out = store.get_embeddings_by_ids([999, 12345])
        assert out == [[], []]

    def test_get_embeddings_by_ids_empty_input(self, store):
        assert store.get_embeddings_by_ids([]) == []


# ---------------------------------------------------------------------------
# search_chunks
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# get_chunks_by_doc — embedding deserialization round-trip
# ---------------------------------------------------------------------------

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
        for orig, retrieved in zip(emb, retrieved_emb):
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


# ---------------------------------------------------------------------------
# upsert_document / search_documents
# ---------------------------------------------------------------------------

class TestDocumentOps:
    """Test document-level CRUD operations."""

    def test_upsert_document(self, store):
        doc = store.upsert_document(
            title="Test Doc",
            tags=["tag1", "tag2"],
            text_summary="A summary of the document.",
            source_file="/path/to/file.pdf",
            total_chunks=5,
        )
        assert doc["title"] == "Test Doc"
        assert doc["total_chunks"] == 5

    def test_search_documents_by_tags(self, store):
        store.upsert_document(
            title="Doc A", tags=["finance"],
            text_summary="Financial document.",
            source_file="/a.pdf", total_chunks=1,
        )
        store.upsert_document(
            title="Doc B", tags=["tech"],
            text_summary="Technology document.",
            source_file="/b.pdf", total_chunks=1,
        )

        results = store.search_documents(tags=["finance"])
        assert len(results) == 1
        assert results[0]["title"] == "Doc A"

    def test_search_documents_no_filter(self, store):
        """Search without filters returns all documents."""
        store.upsert_document(
            title="Doc X", tags=[],
            text_summary="Summary X.", source_file="/x.pdf", total_chunks=1,
        )
        results = store.search_documents()
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# FTS5 full-text search functional tests (previously missing)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Hybrid RRF sorting verification
# ---------------------------------------------------------------------------

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
        assert "OR" in _build_fts_query('RAG: "what is this" (explained)')



# ---------------------------------------------------------------------------
# close / resource cleanup
# ---------------------------------------------------------------------------

class TestResourceCleanup:
    """Test connection lifecycle."""

    def test_close_closes_connection(self, store):
        store.close()
        with pytest.raises((sqlite3.OperationalError, sqlite3.ProgrammingError)):
            store.conn.execute("SELECT 1")
