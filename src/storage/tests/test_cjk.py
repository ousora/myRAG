"""Tests for CJK-segmented FTS (v3 schema)."""

import os
import tempfile


class TestCjkSegmentedFts:
    """unicode61 merges CJK runs into one token.

    The segmented text_seg column makes per-character matching work for
    Chinese queries.
    """

    def test_chinese_keyword_search_finds_chunk(self, store):
        store.upsert_chunks([
            {"text": "风险管控是项目管理的核心内容", "section_path": ["管理"]},
            {"text": "plain english body", "section_path": ["S"]},
        ], doc_id="d1")
        # Text-only path (no vector): FTS5 + BM25
        results = store.hybrid_search("管控")
        assert len(results) == 1
        assert results[0]["text"].startswith("风险管控")

    def test_single_char_query_matches(self, store):
        store.upsert_chunks([{"text": "风险管控", "section_path": ["S"]}],
                            doc_id="d1")
        assert len(store.hybrid_search("险")) == 1

    def test_mixed_cjk_latin_token(self, store):
        store.upsert_chunks([{"text": "部署AI系统提升效率", "section_path": ["S"]}],
                            doc_id="d1")
        assert len(store.hybrid_search("AI")) == 1

    def test_v2_to_v3_migration_preserves_chinese_search(self, tmp_path):
        """A v2-era DB (with a v2-era FTS table) gains CJK search on reopen."""
        fd, path = tempfile.mkstemp(suffix=".db", dir=tmp_path)
        os.close(fd)
        from storage.sqlite_vec import SQLiteVecStore

        db = SQLiteVecStore(path)
        db.upsert_chunks(
            [{"text": "风险管控是项目管理的核心内容", "section_path": ["S"]}],
            doc_id="doc_legacy",
        )
        db.conn.execute("PRAGMA user_version = 2")
        db.conn.commit()
        db.close()

        db2 = SQLiteVecStore(path)
        # Schema setup (and migration) runs lazily on first operation.
        assert len(db2.hybrid_search("核心")) == 1
        assert db2.conn.execute("PRAGMA user_version").fetchone()[0] == 3
        db2.close()
