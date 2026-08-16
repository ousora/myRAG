"""Tests for pipeline.ingest _ingest_markdown."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.ingest import _ingest_markdown


class TestIngestMarkdown:
    def test_file_not_found(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db:
            db_path = db.name
        try:
            with pytest.raises(FileNotFoundError):
                _ingest_markdown("nonexistent.md", db_path)
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_minimal_markdown(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as md:
            md.write("# Test Document\n\nThis is a test.\n")
            md.flush()
            md_path = md.name

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db:
            db_path = db.name

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [{"text": "test", "section_path": ["General"]}]

        mock_embedder = MagicMock()
        mock_embedder.__enter__ = lambda self: self
        mock_embedder.__exit__ = lambda *a: None
        mock_embedder.store_chunks.return_value = [{"text": "test", "embedding": [0.1] * 1024}]
        mock_embedder.store_document.return_value = {"embedding": [0.1] * 1024}

        mock_store = MagicMock()
        mock_store.__enter__ = lambda self: self
        mock_store.__exit__ = lambda *a: None

        try:
            with patch("chunkers.Chunker", return_value=mock_chunker):
                with patch("embedders.Embedder", return_value=mock_embedder):
                    with patch("storage.sqlite_vec.SQLiteVecStore", return_value=mock_store):
                        result = _ingest_markdown(md_path, db_path)
                        assert result == db_path
                        assert Path(db_path).exists()
        finally:
            Path(md_path).unlink()
            Path(db_path).unlink(missing_ok=True)

    def test_markdown_with_heading(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as md:
            md.write("# My Title\n\n## Section\n\nContent here.\n")
            md.flush()
            md_path = md.name

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db:
            db_path = db.name

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [{"text": "test", "section_path": ["General"]}]

        mock_embedder = MagicMock()
        mock_embedder.__enter__ = lambda self: self
        mock_embedder.__exit__ = lambda *a: None
        mock_embedder.store_chunks.return_value = [{"text": "test", "embedding": [0.1] * 1024}]
        mock_embedder.store_document.return_value = {"embedding": [0.1] * 1024}

        mock_store = MagicMock()
        mock_store.__enter__ = lambda self: self
        mock_store.__exit__ = lambda *a: None

        try:
            with patch("chunkers.Chunker", return_value=mock_chunker):
                with patch("embedders.Embedder", return_value=mock_embedder):
                    with patch("storage.sqlite_vec.SQLiteVecStore", return_value=mock_store):
                        result = _ingest_markdown(md_path, db_path)
                        assert result == db_path
        finally:
            Path(md_path).unlink()
            Path(db_path).unlink(missing_ok=True)

    def test_custom_doc_id(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as md:
            md.write("# Test\n\nContent\n")
            md.flush()
            md_path = md.name

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db:
            db_path = db.name

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [{"text": "test", "section_path": ["General"]}]

        mock_embedder = MagicMock()
        mock_embedder.__enter__ = lambda self: self
        mock_embedder.__exit__ = lambda *a: None
        mock_embedder.store_chunks.return_value = [{"text": "test", "embedding": [0.1] * 1024}]
        mock_embedder.store_document.return_value = {"embedding": [0.1] * 1024}

        mock_store = MagicMock()
        mock_store.__enter__ = lambda self: self
        mock_store.__exit__ = lambda *a: None

        try:
            with patch("chunkers.Chunker", return_value=mock_chunker):
                with patch("embedders.Embedder", return_value=mock_embedder):
                    with patch("storage.sqlite_vec.SQLiteVecStore", return_value=mock_store):
                        _ingest_markdown(md_path, db_path, doc_id="custom_doc_123")
        finally:
            Path(md_path).unlink()
            Path(db_path).unlink(missing_ok=True)

    def test_custom_chunk_size(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as md:
            md.write("# Test\n\nContent\n")
            md.flush()
            md_path = md.name

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db:
            db_path = db.name

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [{"text": "test", "section_path": ["General"]}]

        mock_embedder = MagicMock()
        mock_embedder.__enter__ = lambda self: self
        mock_embedder.__exit__ = lambda *a: None
        mock_embedder.store_chunks.return_value = [{"text": "test", "embedding": [0.1] * 1024}]
        mock_embedder.store_document.return_value = {"embedding": [0.1] * 1024}

        mock_store = MagicMock()
        mock_store.__enter__ = lambda self: self
        mock_store.__exit__ = lambda *a: None

        try:
            with patch("chunkers.Chunker", return_value=mock_chunker):
                with patch("embedders.Embedder", return_value=mock_embedder):
                    with patch("storage.sqlite_vec.SQLiteVecStore", return_value=mock_store):
                        _ingest_markdown(md_path, db_path, chunk_size=2048)
        finally:
            Path(md_path).unlink()
            Path(db_path).unlink(missing_ok=True)
