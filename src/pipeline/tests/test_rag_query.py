"""Tests for rag_query() in pipeline.core."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestRagQuery:
    """Tests for rag_query — retrieves chunks and generates LLM answer."""

    def _mock_embedding(self, dim=1024):
        import random
        rng = random.Random(42)
        return [rng.uniform(-1.0, 1.0) for _ in range(dim)]

    @pytest.fixture()
    def tmp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            yield path
        finally:
            Path(path).unlink(missing_ok=True)

    def test_returns_answer_with_context(self, tmp_db):
        """Normal query should return answer, context, and question."""
        from storage.sqlite_vec import SQLiteVecStore

        store = SQLiteVecStore(tmp_db)
        emb = self._mock_embedding()
        store.upsert_chunk(
            {"text": "RAG is retrieval augmented generation", "section_path": ["Intro"]},
            doc_id="doc_1", embedding=emb, chunk_index=0,
        )

        mock_embedder = Mock()
        mock_embedder.embed_query = Mock(return_value=emb)
        mock_embedder.embed = Mock(return_value=[emb])
        mock_embedder.__enter__ = Mock(return_value=mock_embedder)
        mock_embedder.__exit__ = Mock(return_value=False)

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            embedder_cls.return_value = mock_embedder
            llm_mock.return_value = "RAG stands for Retrieval Augmented Generation."

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            result = mod.rag_query("What is RAG?", db_path=tmp_db, k=2,
                                   db=store, embedder=mock_embedder)

        assert "answer" in result
        assert "context" in result
        assert "question" in result
        assert result["question"] == "What is RAG?"
        assert len(result["context"]) >= 1

        store.close()

    def test_empty_results_no_match(self, tmp_db):
        """No matching docs should return a 'no match' answer."""
        from storage.sqlite_vec import SQLiteVecStore

        # Don't close the store — we need it for the query
        store = SQLiteVecStore(tmp_db)

        mock_embedder = Mock()
        mock_embedder.embed_query = Mock(return_value=self._mock_embedding())
        mock_embedder.__enter__ = Mock(return_value=mock_embedder)
        mock_embedder.__exit__ = Mock(return_value=False)

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            embedder_cls.return_value = mock_embedder
            llm_mock.return_value = "some answer"

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            result = mod.rag_query("unknown topic", db_path=tmp_db, k=2,
                                   db=store, embedder=mock_embedder)

        assert "No matching documents found" in result["answer"]
        assert result["context"] == []

    def test_creates_embedder_when_not_provided(self, tmp_db):
        """When embedder is None, a new one should be created."""
        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            mock_e = Mock()
            mock_e.embed_query = Mock(return_value=self._mock_embedding())
            mock_e.embed = Mock(return_value=[self._mock_embedding()])
            mock_e.__enter__ = Mock(return_value=mock_e)
            mock_e.__exit__ = Mock(return_value=False)
            embedder_cls.return_value = mock_e
            llm_mock.return_value = "answer"

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            mod.rag_query("test", db_path=tmp_db, k=2)

        embedder_cls.assert_called()

    def test_creates_store_when_not_provided(self, tmp_db):
        """When db is None, a new SQLiteVecStore should be created."""
        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("storage.sqlite_vec.SQLiteVecStore") as store_cls, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            mock_store = Mock()
            mock_store.hybrid_search = Mock(return_value=[])
            mock_store.search_documents = Mock(return_value=[])
            mock_store.get_embeddings_by_ids = Mock(return_value=[[]])
            mock_store.close = Mock()
            store_cls.return_value = mock_store

            mock_e = Mock()
            mock_e.embed_query = Mock(return_value=self._mock_embedding())
            mock_e.__enter__ = Mock(return_value=mock_e)
            mock_e.__exit__ = Mock(return_value=False)
            embedder_cls.return_value = mock_e
            llm_mock.return_value = "answer"

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            mod.rag_query("test", db_path=tmp_db, k=2)

        store_cls.assert_called_with(tmp_db)

    def test_closes_created_store(self, tmp_db):
        """A store created internally should be closed after use."""
        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("storage.sqlite_vec.SQLiteVecStore") as store_cls, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            mock_store = Mock()
            mock_store.hybrid_search = Mock(return_value=[])
            mock_store.search_documents = Mock(return_value=[])
            mock_store.get_embeddings_by_ids = Mock(return_value=[[]])
            mock_store.close = Mock()
            store_cls.return_value = mock_store

            mock_e = Mock()
            mock_e.embed_query = Mock(return_value=self._mock_embedding())
            mock_e.__enter__ = Mock(return_value=mock_e)
            mock_e.__exit__ = Mock(return_value=False)
            embedder_cls.return_value = mock_e
            llm_mock.return_value = "answer"

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            mod.rag_query("test", db_path=tmp_db, k=2)

        mock_store.close.assert_called_once()

    def test_does_not_close_provided_store(self, tmp_db):
        """A store passed in should not be closed by rag_query."""
        from storage.sqlite_vec import SQLiteVecStore

        # Don't close the store
        store = SQLiteVecStore(tmp_db)

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("storage.sqlite_vec.SQLiteVecStore") as store_cls, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            store_cls.return_value = Mock()
            store_cls.return_value.hybrid_search = Mock(return_value=[])
            store_cls.return_value.search_documents = Mock(return_value=[])
            store_cls.return_value.get_embeddings_by_ids = Mock(return_value=[[]])

            mock_e = Mock()
            mock_e.embed_query = Mock(return_value=self._mock_embedding())
            mock_e.__enter__ = Mock(return_value=mock_e)
            mock_e.__exit__ = Mock(return_value=False)
            embedder_cls.return_value = mock_e
            llm_mock.return_value = "answer"

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            mod.rag_query("test", db_path=tmp_db, k=2, db=store)

        store_cls.assert_not_called()

    def test_does_not_close_provided_embedder(self, tmp_db):
        """A provided embedder should not have __exit__ called."""
        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("storage.sqlite_vec.SQLiteVecStore") as store_cls, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            store_cls.return_value = Mock()
            store_cls.return_value.hybrid_search = Mock(return_value=[])
            store_cls.return_value.search_documents = Mock(return_value=[])
            store_cls.return_value.get_embeddings_by_ids = Mock(return_value=[[]])
            store_cls.return_value.close = Mock()

            mock_embedder = Mock()
            mock_embedder.embed_query = Mock(return_value=self._mock_embedding())
            mock_embedder.__exit__ = Mock(return_value=False)
            embedder_cls.return_value = mock_embedder
            llm_mock.return_value = "answer"

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            mod.rag_query("test", db_path=tmp_db, k=2, embedder=mock_embedder)

        mock_embedder.__exit__.assert_not_called()

    def test_context_assembled_from_chunks(self, tmp_db):
        """Context parts should include chunk metadata and text."""
        from storage.sqlite_vec import SQLiteVecStore

        # Don't close the store
        store = SQLiteVecStore(tmp_db)
        emb = self._mock_embedding()
        store.upsert_chunk(
            {"text": "RAG is cool", "section_path": ["Intro"]},
            doc_id="doc_1", embedding=emb, chunk_index=0,
        )

        mock_embedder = Mock()
        mock_embedder.embed_query = Mock(return_value=emb)
        mock_embedder.__enter__ = Mock(return_value=mock_embedder)
        mock_embedder.__exit__ = Mock(return_value=False)

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            embedder_cls.return_value = mock_embedder
            llm_mock.return_value = "RAG is cool."

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            mod.rag_query("What is RAG?", db_path=tmp_db, k=2,
                                   db=store, embedder=mock_embedder)

        # The LLM was called with a prompt that includes the assembled context
        llm_mock.assert_called_once()
        call_args = llm_mock.call_args
        user_prompt = call_args[0][1] if call_args[0] else call_args[1].get("user_prompt", "")
        assert "RAG is cool" in user_prompt
        assert "doc_1" in user_prompt

    def test_llm_error_propagates(self, tmp_db):
        """LLM API errors should propagate."""
        from storage.sqlite_vec import SQLiteVecStore

        # Don't close the store
        store = SQLiteVecStore(tmp_db)
        emb = self._mock_embedding()
        store.upsert_chunk(
            {"text": "relevant chunk", "section_path": ["S"]},
            doc_id="doc_1", embedding=emb, chunk_index=0,
        )

        mock_embedder = Mock()
        mock_embedder.embed_query = Mock(return_value=emb)
        mock_embedder.__enter__ = Mock(return_value=mock_embedder)
        mock_embedder.__exit__ = Mock(return_value=False)

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            embedder_cls.return_value = mock_embedder
            llm_mock.side_effect = RuntimeError("LLM API error")

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            with pytest.raises(RuntimeError, match="LLM API error"):
                mod.rag_query("test", db_path=tmp_db, k=2, db=store, embedder=mock_embedder)

    def test_k_parameter_respected(self, tmp_db):
        """The k parameter should limit the number of results returned."""
        from storage.sqlite_vec import SQLiteVecStore

        # Don't close the store
        store = SQLiteVecStore(tmp_db)
        emb = self._mock_embedding()
        for i in range(5):
            store.upsert_chunk(
                {"text": f"Chunk {i}", "section_path": ["S"]},
                doc_id="doc_1", embedding=emb, chunk_index=i,
            )

        mock_embedder = Mock()
        mock_embedder.embed_query = Mock(return_value=emb)
        mock_embedder.__enter__ = Mock(return_value=mock_embedder)
        mock_embedder.__exit__ = Mock(return_value=False)

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            embedder_cls.return_value = mock_embedder
            llm_mock.return_value = "answer"

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            result = mod.rag_query("test", db_path=tmp_db, k=2,
                                   db=store, embedder=mock_embedder)

        assert len(result["context"]) <= 2

    def test_system_prompt_includes_instructions(self, tmp_db):
        """The system prompt should instruct the LLM to use only context."""
        from storage.sqlite_vec import SQLiteVecStore

        # Don't close the store
        store = SQLiteVecStore(tmp_db)
        emb = self._mock_embedding()
        store.upsert_chunk(
            {"text": "relevant chunk", "section_path": ["S"]},
            doc_id="doc_1", embedding=emb, chunk_index=0,
        )

        mock_embedder = Mock()
        mock_embedder.embed_query = Mock(return_value=emb)
        mock_embedder.__enter__ = Mock(return_value=mock_embedder)
        mock_embedder.__exit__ = Mock(return_value=False)

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("embedders.create_embedder") as embedder_cls, \
             patch("formatters.call_llm_raw") as llm_mock:

            cfg_mock.return_value.format_timeout = 30
            embedder_cls.return_value = mock_embedder
            llm_mock.return_value = "answer"

            mod = __import__("pipeline.core", fromlist=["rag_query"])
            mod.rag_query("test", db_path=tmp_db, k=2, db=store, embedder=mock_embedder)

        call_args = llm_mock.call_args
        system_prompt = call_args[0][0] if call_args[0] else ""
        assert "Use ONLY the information in the context" in system_prompt
