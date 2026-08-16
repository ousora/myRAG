"""Tests for LocalEmbedder class in embedders.local_bge."""

import sys
import types
from unittest.mock import Mock, patch

import pytest


class FakeSentenceTransformer:
    """A fake SentenceTransformer class for testing.
    
    encode() returns a list with a tolist() method (simulating numpy array).
    For single string input: returns [[emb]] where emb is a list of floats.
    For list input: returns [[emb1], [emb2], ...].
    tolist() returns the first embedding for str input, or full list for list input.
    """

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device

    def encode(self, text):
        if isinstance(text, str):
            data = [[0.1] * 1024]
        else:
            data = [[0.1] * 1024 for _ in text]
        return _NumpyLike(data, is_batch=not isinstance(text, str))


class _NumpyLike:
    """Minimal numpy array-like with tolist() method."""

    def __init__(self, data, is_batch=False):
        self._data = data
        self._is_batch = is_batch

    def tolist(self):
        if self._is_batch:
            return self._data  # Full list of embeddings for batch input
        return self._data[0]  # Single embedding for str input


@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    """Provide a fake sentence_transformers module for all tests in this file."""
    st_module = types.ModuleType("sentence_transformers")
    st_module.SentenceTransformer = FakeSentenceTransformer
    orig = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = st_module
    yield st_module
    if orig is not None:
        sys.modules["sentence_transformers"] = orig
    elif "sentence_transformers" in sys.modules:
        del sys.modules["sentence_transformers"]


class TestLocalEmbedderInit:
    """Test LocalEmbedder initialization."""

    def test_default_model_name(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        assert e._model.model_name == "BAAI/bge-m3"
        assert e._model.device == "cpu"

    def test_custom_model_name(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder(model_name="custom/model")
        assert e._model.model_name == "custom/model"

    def test_custom_device(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder(device="cuda")
        assert e._model.device == "cuda"

    def test_custom_batch_size(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder(batch_size=16)
        assert e.batch_size == 16

    def test_context_manager_returns_self(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        assert e.__enter__() is e

    def test_context_manager_exit_returns_false(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        assert e.__exit__(None, None, None) is False


class TestLocalEmbedderEmbed:
    """Test the embed() method."""

    def test_single_string_returns_list(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        result = e.embed("hello")
        assert isinstance(result, list)
        assert len(result) == 1024

    def test_list_of_strings_returns_list_of_lists(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        result = e.embed(["a", "b"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, list) and len(r) == 1024 for r in result)

    def test_empty_list_returns_empty_list(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        result = e.embed([])
        assert result == []

    def test_single_string_cached(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        mock_encode = Mock(return_value=_NumpyLike([[0.1] * 1024], is_batch=False))
        e._model.encode = mock_encode
        e.embed("cached text")
        e.embed("cached text")
        # Should only call encode once (cached)
        assert mock_encode.call_count == 1

    def test_batch_uses_adaptive_batch_size(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder(batch_size=2)
        texts = ["short"] * 5
        result = e.embed(texts)
        assert len(result) == 5

    def test_oom_fallback_single_item(self):
        from embedders.local_bge import LocalEmbedder

        class FakeModel:
            def __init__(self):
                self.calls = [0]

            def encode(self, text):
                self.calls[0] += 1
                if isinstance(text, list) and len(text) > 1:
                    raise RuntimeError("out of memory")
                if isinstance(text, str):
                    return _NumpyLike([[0.1] * 1024], is_batch=False)
                return _NumpyLike([[0.1] * 1024 for _ in text], is_batch=True)

        model = FakeModel()
        e = LocalEmbedder(batch_size=32)
        e._model = model
        result = e.embed(["a", "b", "c", "d", "e"])
        assert len(result) == 5


class TestLocalEmbedderEmbedQuery:
    """Test the embed_query() method."""

    def test_prepends_instruction(self):
        with patch("config.get_config") as cfg_mock:
            cfg_mock.return_value.embedding_query_instruction = "Represent this sentence for searching relevant passages: "

            from embedders.local_bge import LocalEmbedder
            e = LocalEmbedder()
            mock_encode = Mock(return_value=_NumpyLike([[0.1] * 1024], is_batch=False))
            e._model.encode = mock_encode
            e.embed_query("test query")

            call_arg = mock_encode.call_args[0][0]
            assert call_arg.startswith("Represent this sentence for searching relevant passages: test query")

    def test_no_instruction_when_empty(self):
        with patch("config.get_config") as cfg_mock:
            cfg_mock.return_value.embedding_query_instruction = ""

            from embedders.local_bge import LocalEmbedder
            e = LocalEmbedder()
            mock_encode = Mock(return_value=_NumpyLike([[0.1] * 1024], is_batch=False))
            e._model.encode = mock_encode
            e.embed_query("test query")

            call_arg = mock_encode.call_args[0][0]
            assert call_arg == "test query"


class TestLocalEmbedderStoreChunks:
    """Test store_chunks method."""

    def test_returns_enriched_chunks(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        chunks = [{"text": "chunk one", "section_path": ["S1"]}]
        result = e.store_chunks(chunks, doc_id="doc_1")

        assert len(result) == 1
        assert result[0]["source_doc_id"] == "doc_1"
        assert result[0]["chunk_index"] == 0
        assert result[0]["word_count"] == 2
        assert len(result[0]["embedding"]) == 1024

    def test_empty_chunks_returns_empty_list(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        result = e.store_chunks([])
        assert result == []

    def test_multiple_chunks_batched(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        chunks = [{"text": f"chunk {i}", "section_path": ["S"]} for i in range(3)]
        result = e.store_chunks(chunks, doc_id="doc_0")

        assert len(result) == 3
        for i, r in enumerate(result):
            assert r["chunk_index"] == i
            assert r["source_doc_id"] == "doc_0"


class TestLocalEmbedderStoreDocument:
    """Test store_document method."""

    def test_returns_document_metadata(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        result = e.store_document(
            title="Test Doc", tags=["tag1"],
            text_summary="A summary", source_file="/path/file.pdf",
            total_chunks=5,
        )

        assert result["title"] == "Test Doc"
        assert result["tags"] == ["tag1"]
        assert result["source_file"] == "/path/file.pdf"
        assert result["total_chunks"] == 5
        assert len(result["embedding"]) == 1024

    def test_summary_truncated_to_1000(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        long_summary = "x" * 5000
        result = e.store_document(
            title="Test", tags=[], text_summary=long_summary,
            source_file="/f.pdf", total_chunks=1,
        )

        assert len(result["text_summary"]) <= 1000


class TestLocalEmbedderTokenEstimation:
    """Test _estimate_tokens static method."""

    def test_english_text(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        tokens = e._estimate_tokens("Hello world this is a test sentence for token estimation")
        assert tokens > 0

    def test_cjk_text(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        tokens = e._estimate_tokens("你好世界测试")
        assert tokens >= 6

    def test_empty_string(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder()
        tokens = e._estimate_tokens("")
        assert tokens == 0


class TestLocalEmbedderAdaptiveBatchSize:
    """Test _adaptive_batch_size method."""

    def test_short_texts_use_default_batch_size(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder(batch_size=32)
        texts = ["short"] * 10
        bs = e._adaptive_batch_size(texts)
        assert bs == 32

    def test_long_texts_reduce_batch_size(self):
        from embedders.local_bge import LocalEmbedder
        e = LocalEmbedder(batch_size=32, max_tokens_per_batch=100)
        texts = ["x" * 50] * 10
        bs = e._adaptive_batch_size(texts)
        assert bs < 32
        assert bs >= 4
