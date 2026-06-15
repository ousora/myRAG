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
# close / resource cleanup
# ---------------------------------------------------------------------------

class TestResourceCleanup:
    """Test connection lifecycle."""

    def test_close_closes_connection(self, store):
        store.close()
        with pytest.raises((sqlite3.OperationalError, sqlite3.ProgrammingError)):
            store.conn.execute("SELECT 1")
