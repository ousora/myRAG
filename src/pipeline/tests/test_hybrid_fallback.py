"""Regression tests for graceful degradation in process_file_hybrid.

Covers the case where embedding generation fails with a typed
``MyRagException`` (e.g. ``EmbeddingError`` from a dimension/count mismatch):
the pipeline must degrade to keyword-only storage instead of propagating the
exception to the caller.
"""

import contextlib
from unittest.mock import Mock, patch

from myrag.exceptions import EmbeddingError


@contextlib.contextmanager
def _patch_common(*, embedder_raises: bool = False, doc_raises: bool = False):
    """Set up the parse/format/chunk pipeline with mocks; yield the ExitStack."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("pipeline.core._get_config"))
        resolve_mock = stack.enter_context(
            patch("pipeline.core.utils.resolve_parser")
        )
        stack.enter_context(patch("formatters.format_text_async"))
        chunker_cls = stack.enter_context(patch("pipeline.core.Chunker"))
        render_mock = stack.enter_context(
            patch("pipeline.core.markdown_utils.render_markdown")
        )
        stack.enter_context(
            patch("pipeline.core.markdown_utils.strip_reference_sections")
        )
        match_mock = stack.enter_context(
            patch("pipeline.core.markdown_utils.match_entities_to_chunks")
        )
        embedder_cls = stack.enter_context(
            patch("embedders.maybe_create_embedder")
        )

        resolve_mock.return_value = Mock(parse=Mock(return_value="test content"))
        chunker_cls.return_value.chunk.return_value = [
            {"text": "chunk1", "section_path": ["Section"]},
            {"text": "chunk2", "section_path": ["Section"]},
        ]
        body = "# Test Document\n\nBody text."
        render_mock.return_value = body
        match_mock.return_value = [
            {"text": "chunk1", "section_path": ["Section"]},
            {"text": "chunk2", "section_path": ["Section"]},
        ]

        embedder = Mock()
        if embedder_raises:
            embedder.store_chunks.side_effect = EmbeddingError(
                message="dimension mismatch", context={"actual_dim": 512}
            )
        else:
            embedder.store_chunks.return_value = [
                {"text": "chunk1", "section_path": ["Section"], "embedding": [0.1] * 1024},
                {"text": "chunk2", "section_path": ["Section"], "embedding": [0.2] * 1024},
            ]
        if doc_raises:
            embedder.store_document.side_effect = EmbeddingError(message="bad dim")
        else:
            embedder.store_document.return_value = {"embedding": [0.0] * 1024}
        embedder_cls.return_value = embedder

        yield stack


class TestProcessFileHybridEmbeddingFallback:
    """process_file_hybrid must degrade gracefully on embedding failures."""

    def test_embedding_error_degrades_to_keyword_only(self):
        """An EmbeddingError from store_chunks must not propagate; chunks are stored without vectors so keyword-only retrieval still works."""
        from pipeline.core import process_file_hybrid

        with _patch_common(embedder_raises=True):
            result = process_file_hybrid(
                "doc.txt", store_path="/tmp/fallback.db", doc_id="doc_0"
            )

        # No exception reached here — degradation succeeded.
        assert len(result["chunks"]) == 2
        for chunk in result["chunks"]:
            assert "embedding" not in chunk
            assert chunk["source_doc_id"] == "doc_0"
        # Document record is persisted without an embedding too.
        assert result["document"]["source_file"] == "doc.txt"
        # Nothing was written to the store (the failure preceded the DB write),
        # so the returned chunks are in-memory and db_path stays None.
        assert result["db_path"] is None

    def test_store_document_error_degrades_to_keyword_only(self):
        """An EmbeddingError from store_document (after chunks embedded) also degrades instead of propagating."""
        from pipeline.core import process_file_hybrid

        with _patch_common(doc_raises=True):
            result = process_file_hybrid(
                "doc.txt", store_path="/tmp/fallback.db", doc_id="doc_0"
            )

        assert len(result["chunks"]) == 2
        assert result["document"].get("embedding") is None
