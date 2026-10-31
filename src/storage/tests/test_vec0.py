"""Tests for vec0 ANN table (chunks_vec): sync, KNN, migration."""

import os
import tempfile

from storage.tests.conftest import _make_embedding


class TestVec0AnnTable:
    """The chunks_vec KNN mirror must stay in sync and serve ANN queries."""

    def test_upsert_chunk_syncs_ann_mirror(self, store):
        emb = _make_embedding()
        res = store.upsert_chunk({"text": "hello world", "section_path": ["S"]},
                                 doc_id="d1", embedding=emb, chunk_index=0)
        rows = store.conn.execute(
            "SELECT rowid, source_doc_id, chunk_index FROM chunks_vec"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0] == (res["id"], "d1", 0)
        # Re-upsert: mirror stays at exactly one row (delete-then-insert).
        store.upsert_chunk({"text": "updated", "section_path": ["S"]},
                           doc_id="d1", embedding=emb, chunk_index=0)
        assert store.conn.execute("SELECT count(*) FROM chunks_vec").fetchone()[0] == 1

    def test_upsert_chunks_and_tail_delete_sync_ann_mirror(self, store):
        embs = [_make_embedding() for _ in range(3)]
        chunks = [{"text": f"chunk {i}", "section_path": ["S"]} for i in range(3)]
        store.upsert_chunks([{"text": c["text"], "embedding": e} for c, e in zip(chunks, embs, strict=True)],
                            doc_id="d1")
        assert store.conn.execute(
            "SELECT count(*) FROM chunks_vec WHERE source_doc_id='d1'"
        ).fetchone()[0] == 3
        # Re-ingest with fewer chunks → tail cleanup removes mirror rows too.
        store.upsert_chunks([{"text": "only one", "embedding": embs[0]}], doc_id="d1")
        assert store.conn.execute(
            "SELECT count(*) FROM chunks_vec WHERE source_doc_id='d1'"
        ).fetchone()[0] == 1

    def test_delete_doc_chunks_clears_ann_mirror(self, store):
        embs = [_make_embedding() for _ in range(2)]
        store.upsert_chunks([{"text": "a", "embedding": embs[0]},
                             {"text": "b", "embedding": embs[1]}], doc_id="d1")
        store.delete_doc_chunks("d1")
        assert store.conn.execute(
            "SELECT count(*) FROM chunks_vec WHERE source_doc_id='d1'"
        ).fetchone()[0] == 0

    def test_search_chunks_uses_knn_not_brute_force(self, store):
        """Emptying the ANN mirror must yield empty results.

        Proves the KNN path is taken instead of the brute-force scan fallback.
        """
        emb = _make_embedding()
        store.upsert_chunk({"text": "find me", "section_path": ["S"]},
                           doc_id="d1", embedding=emb, chunk_index=0)
        assert len(store.search_chunks(emb, k=5)) == 1
        store.conn.execute("DELETE FROM chunks_vec")
        store.conn.commit()
        # Brute force would still find the row; KNN returns nothing.
        assert store.search_chunks(emb, k=5) == []

    def test_knn_respects_source_doc_filter(self, store):
        emb_a = _make_embedding()
        emb_b = [v * -1 for v in _make_embedding()]
        store.upsert_chunk({"text": "a", "section_path": ["S"]}, doc_id="a", embedding=emb_a, chunk_index=0)
        store.upsert_chunk({"text": "b", "section_path": ["S"]}, doc_id="b", embedding=emb_b, chunk_index=0)
        results = store.search_chunks(emb_a, k=5, source_doc_id="a")
        assert [r["source_doc_id"] for r in results] == ["a"]

    def test_migration_backfills_ann_table(self, tmp_path):
        """Opening a legacy DB (user_version=0 with existing rows) backfills chunks_vec."""
        fd, path = tempfile.mkstemp(suffix=".db", dir=tmp_path)
        os.close(fd)
        from storage.sqlite_vec import SQLiteVecStore
        db = SQLiteVecStore(path)
        emb = _make_embedding()
        db.upsert_chunk({"text": "legacy row", "section_path": ["S"]},
                        doc_id="doc_legacy", embedding=emb, chunk_index=0)
        # Roll back to a pre-vec0 version to simulate a legacy database file.
        db.conn.execute("PRAGMA user_version = 0")
        db.conn.commit()
        db.close()

        db2 = SQLiteVecStore(path)
        rows = db2.conn.execute(
            "SELECT rowid, source_doc_id FROM chunks_vec"
        ).fetchall()
        assert len(rows) == 1 and rows[0][1] == "doc_legacy"
        # And retrieval works through the ANN path.
        assert len(db2.search_chunks(emb, k=5)) == 1
        db2.close()
