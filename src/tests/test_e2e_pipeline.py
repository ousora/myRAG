"""End-to-end test: parse → clean → chunk → embed → store → query.

Uses hash-based pseudo-embeddings (no remote service) and a mocked LLM
formatter, so the full pipeline runs without any external dependency.
"""

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import pytest


class _FakeConfig:
    """Minimal config that routes Embedder through hash fallback."""

    embedding_mode = "remote"
    embedding_base_url = "http://localhost:11435"
    embedding_model = "bge-m3"
    embedding_timeout = 60
    embedding_hash_fallback = True
    embedding_query_instruction = ""
    format_timeout = 30


def _mock_format_future(text, source_type="pdf", **kwargs):
    future = Future()
    future.set_result({
        "title": "E2E Test Doc",
        "tags": ["e2e", "test"],
        "metadata": {"sections": [], "entities": []},
        "body": text,
    })
    return future


@pytest.fixture()
def patched_config():
    """Route Embedder through hash fallback everywhere it's imported."""
    with patch("embedders.bge_m3.get_config", return_value=_FakeConfig()):
        yield


class TestEndToEndPipeline:
    def test_ingest_and_query_roundtrip(self, tmp_path, patched_config):
        """Full pipeline: real file → DB → query → answer from stored context."""
        from pipeline import process_file_hybrid, rag_query

        # 1. Real-ish document
        doc = tmp_path / "e2e_doc.txt"
        doc.write_text(
            "The quick brown fox jumps over the lazy dog.\n\n"
            "RAG stands for retrieval augmented generation.\n\n"
            "SQLite is a serverless SQL database engine.\n"
        )

        # 2. Ingest: parse → clean → format (mocked) → chunk → embed (hash) → store
        db_path = str(tmp_path / "e2e.db")
        with patch("formatters.format_text_async", _mock_format_future):
            result = process_file_hybrid(
                filepath=str(doc),
                doc_id="e2e_doc",
                store_path=db_path,
            )

        assert result["db_path"] == db_path
        assert len(result["chunks"]) >= 1
        assert result["document"]["title"] == "E2E Test Doc"
        assert Path(db_path).exists()

        # 3. Query: embed query (hash) → hybrid search → LLM answer
        with patch("formatters.call_llm_raw", return_value="RAG stands for Retrieval Augmented Generation."):
            answer_result = rag_query(
                "What does RAG stand for?",
                db_path=db_path,
                k=3,
            )

        assert "answer" in answer_result
        assert "context" in answer_result
        assert answer_result["context"], "should have retrieved chunks"
        # Answer must echo the stored context, not pure hallucination.
        assert "RAG" in answer_result["answer"]