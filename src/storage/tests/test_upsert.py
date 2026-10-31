"""Tests for chunk upsert, batch insert, and FTS sync."""


from storage.tests.conftest import _make_embedding


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
        chunks: list[dict] = [
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
        chunks: list[dict] = [
            {"text": f"C{i}", "section_path": ["S"]} for i in range(3)
        ]
        embs = [_make_embedding() for _ in range(3)]
        for i, c in enumerate(chunks):
            c["embedding"] = embs[i]

        results = store.upsert_chunks(chunks, doc_id="doc_ids")
        ids = [r["id"] for r in results]
        assert all(i is not None for i in ids)
        assert [r["chunk_index"] for r in results] == [0, 1, 2]
        # Ids must be distinct per chunk_index within the same doc.
        assert len({r["id"] for r in results}) == 3

    def test_upsert_chunks_removes_stale_tail(self, store):
        """Re-ingesting a shrunken document must delete orphaned old chunks.

        Regression: ON CONFLICT upserts left rows beyond the new chunk count
        in the index forever, polluting retrieval with ghost content.
        """
        embs = [_make_embedding() for _ in range(3)]
        chunks3 = [{"text": f"old {i}", "section_path": ["S"], "embedding": embs[i]} for i in range(3)]
        store.upsert_chunks(chunks3, doc_id="doc_shrink")
        assert len(store.get_chunks_by_doc("doc_shrink")) == 3

        chunks1 = [{"text": "new shorter content", "section_path": ["S"], "embedding": embs[0]}]
        store.upsert_chunks(chunks1, doc_id="doc_shrink")

        remaining = store.get_chunks_by_doc("doc_shrink")
        assert len(remaining) == 1
        assert remaining[0]["text"] == "new shorter content"
        # Ghost text must be gone from the FTS index too.
        ghosts = store.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'old'"
        ).fetchall()
        assert len(ghosts) == 0

    def test_delete_doc_chunks(self, store):
        """delete_doc_chunks removes all rows for the doc and syncs FTS."""
        emb = _make_embedding()
        store.upsert_chunk({"text": "deletable content", "section_path": ["S"]},
                           doc_id="doc_del_api", embedding=emb, chunk_index=0)
        store.upsert_chunk({"text": "second chunk", "section_path": ["S"]},
                           doc_id="doc_del_api", embedding=emb, chunk_index=1)

        removed = store.delete_doc_chunks("doc_del_api")
        assert removed == 2
        assert store.get_chunks_by_doc("doc_del_api") == []
        ghosts = store.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'deletable'"
        ).fetchall()
        assert len(ghosts) == 0
        assert store.delete_doc_chunks("doc_del_api") == 0

    def test_delete_document(self, store):
        """delete_document removes the B-index row keyed by source_file."""
        store.upsert_document(
            title="Doomed", tags=[], text_summary="bye",
            source_file="/gone.pdf", total_chunks=0,
        )
        assert store.delete_document("/gone.pdf") is True
        assert store.delete_document("/gone.pdf") is False
        assert store.search_documents(tags=[]) == []

    def test_get_embeddings_by_ids_round_trip(self, store):
        """Stored chunk embeddings are returned aligned to the requested ids."""
        embs = [_make_embedding() for _ in range(2)]
        r0 = store.upsert_chunk({"text": "a", "section_path": ["S"]}, doc_id="doc_ge", embedding=embs[0], chunk_index=0)
        r1 = store.upsert_chunk({"text": "b", "section_path": ["S"]}, doc_id="doc_ge", embedding=embs[1], chunk_index=1)

        out = store.get_embeddings_by_ids([r0["id"], r1["id"]])
        assert len(out) == 2
        for orig, got in zip(embs, out, strict=True):
            assert len(got) == 1024
            assert all(abs(o - g) < 0.001 for o, g in zip(orig, got, strict=True))

    def test_get_embeddings_by_ids_unknown_is_empty(self, store):
        """Unknown ids yield an empty list kept in the requested position."""
        out = store.get_embeddings_by_ids([999, 12345])
        assert out == [[], []]

    def test_get_embeddings_by_ids_empty_input(self, store):
        assert store.get_embeddings_by_ids([]) == []


class TestUpsertFtsSync:
    """Re-ingesting a document must not leave ghost entries in chunks_fts."""

    def test_reingest_updates_fts_no_ghosts(self, store):
        """Regression test for FTS sync on re-ingest.

        INSERT OR REPLACE skipped the AFTER DELETE trigger
        (recursive_triggers=OFF), leaving stale FTS entries forever.
        """
        emb = _make_embedding()
        store.upsert_chunk(
            {"text": "obsolete fuzzy wobble content", "section_path": ["S"]},
            doc_id="doc_sync", embedding=emb, chunk_index=0,
        )
        # Re-ingest the same (doc_id, chunk_index) with new content.
        store.upsert_chunk(
            {"text": "brand new shiny content", "section_path": ["S"]},
            doc_id="doc_sync", embedding=emb, chunk_index=0,
        )

        # Old text must be gone from the FTS index...
        ghosts = store.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'wobble'"
        ).fetchall()
        assert len(ghosts) == 0, "stale FTS entry survived re-ingest"
        # ...and the new text must be searchable.
        results = store.hybrid_search("shiny content")
        assert any("brand new shiny" in r["text"] for r in results)

    def test_reingest_preserves_chunk_id(self, store):
        """Upsert must UPDATE in place, not churn AUTOINCREMENT ids."""
        emb = _make_embedding()
        first = store.upsert_chunk(
            {"text": "version one", "section_path": ["S"]},
            doc_id="doc_stable", embedding=emb, chunk_index=0,
        )
        second = store.upsert_chunk(
            {"text": "version two", "section_path": ["S"]},
            doc_id="doc_stable", embedding=emb, chunk_index=0,
        )
        assert first["id"] == second["id"]

    def test_fts_rebuild_migration_clears_preexisting_ghosts(self, tmp_path):
        """Opening a legacy DB (user_version=0) rebuilds the FTS index."""
        from storage.sqlite_vec import SQLiteVecStore

        db_path = tmp_path / "legacy.db"
        db = SQLiteVecStore(str(db_path))
        emb = _make_embedding()
        db.upsert_chunk(
            {"text": "ghost words here", "section_path": ["S"]},
            doc_id="doc_legacy", embedding=emb, chunk_index=0,
        )
        # Simulate legacy corruption directly in the FTS index, then roll the
        # schema version back so the next open sees an unmigrated database.
        # (Note: index-only ghost rows are visible via MATCH, not plain scans.)
        db.conn.execute(
            "INSERT INTO chunks_fts(rowid, text_seg) VALUES (999999, 'phantom orphan entry')"
        )
        db.conn.execute("PRAGMA user_version = 0")
        db.conn.commit()
        db.close()

        # Reopen: the one-time rebuild must drop the ghost row.
        db2 = SQLiteVecStore(str(db_path))
        results = db2.hybrid_search("ghost words")
        ghosts = db2.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'phantom'"
        ).fetchall()
        assert len(ghosts) == 0
        # v3 is current: legacy DBs are migrated through the CJK-segmented FTS
        # rebuild (which also covers the old v1 ghost-row fix) and the v2
        # chunks_vec ANN backfill in one open.
        assert db2.conn.execute("PRAGMA user_version").fetchone()[0] == 3
        # Real content survives the rebuild.
        assert any("ghost words here" in r["text"] for r in results)
        db2.close()
