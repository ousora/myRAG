"""Tests for the Embedder class."""


class TestEmbedderInit:
    """Test Embedder initialization and config loading."""

    def test_init_with_explicit_params(self):
        from embedders.bge_m3 import Embedder

        e = Embedder(base_url="http://example.com", model="test-model")
        assert e.model == "test-model"

    def test_init_uses_config_defaults(self, monkeypatch):
        """Embedder should fall back to config when params are empty."""
        from embedders.bge_m3 import Embedder

        class FakeConfig:
            embedding_base_url = "http://config-url"
            embedding_model = "bge-m3-from-config"
            embedding_timeout = 120

        monkeypatch.setattr("config.get_config", lambda: FakeConfig())

        e = Embedder()
        assert e.model == "bge-m3-from-config"


class TestStoreChunk:
    """Test store_chunk metadata construction."""

    def test_store_chunk_returns_expected_keys(self):
        from embedders.bge_m3 import Embedder

        class FakeEmbedder(Embedder):
            def embed(self, text):
                return [0.1] * 1024 if isinstance(text, str) else [[0.1] * 1024 for _ in text]

        e = FakeEmbedder()
        result = e.store_chunk("test content", section_path=["Section"], doc_id="doc_1", chunk_idx=5)

        assert "text" in result
        assert "section_path" in result
        assert "source_doc_id" in result
        assert "chunk_index" in result
        assert "word_count" in result
        assert "embedding" in result
        assert result["source_doc_id"] == "doc_1"
        assert result["chunk_index"] == 5

    def test_store_chunk_defaults_section_path(self):
        from embedders.bge_m3 import Embedder

        class FakeEmbedder(Embedder):
            def embed(self, text):
                return [0.1] * 1024 if isinstance(text, str) else [[0.1] * 1024 for _ in text]

        e = FakeEmbedder()
        result = e.store_chunk("test content")
        assert result["section_path"] == ["General"]


class TestStoreChunks:
    """Test store_chunks batch embedding."""

    def test_store_chunks_returns_correct_count(self):
        from embedders.bge_m3 import Embedder

        class FakeEmbedder(Embedder):
            def embed(self, text):
                return [0.1] * 1024 if isinstance(text, str) else [[0.1] * 1024 for _ in text]

        e = FakeEmbedder()
        chunks = [{"text": "chunk one"}, {"text": "chunk two"}]
        results = e.store_chunks(chunks, doc_id="doc_0")

        assert len(results) == 2
        assert all("embedding" in r for r in results)


class TestStoreDocument:
    """Test store_document metadata construction."""

    def test_store_document_truncates_summary(self):
        from embedders.bge_m3 import Embedder

        class FakeEmbedder(Embedder):
            def embed(self, text):
                return [0.1] * 1024 if isinstance(text, str) else [[0.1] * 1024 for _ in text]

        e = FakeEmbedder()
        long_text = "x" * 2000
        result = e.store_document(
            title="Test Doc",
            tags=["tag1"],
            text_summary=long_text,
            source_file="/path/to/file.txt",
            total_chunks=42,
        )

        assert len(result["text_summary"]) <= 1000


class TestEmbedTextsWrapper:
    """Test the convenience wrapper function."""

    def test_embed_texts_wrapper(self):
        from embedders.bge_m3 import Embedder, embed_texts

        class FakeEmbedder(Embedder):
            def embed(self, text):
                return [0.1] * 1024 if isinstance(text, str) else [[0.1] * 1024 for _ in text]

        # Patch the Embedder constructor to use our fake class
        original_embedder = Embedder
        try:
            import embedders.bge_m3 as mod
            mod.Embedder = FakeEmbedder
            results = embed_texts(["hello", "world"])
            assert len(results) == 2
            assert all(isinstance(r, list) for r in results)
        finally:
            mod.Embedder = original_embedder
