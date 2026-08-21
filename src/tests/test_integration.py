"""Integration tests for process_file() and process_directory().

These exercise the full parse → clean → chunk pipeline without requiring
LLM/embedding services (uses plain-text input).
"""

import os
import tempfile
from concurrent.futures import Future
from pathlib import Path


def test_process_file_text(tmp_path: Path) -> None:
    """process_file on a .txt file returns structured chunks."""
    from pipeline import process_file

    txt = tmp_path / "sample.txt"
    txt.write_text("Hello world\nThis is a test document.\n")

    chunks = process_file(str(txt))
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    for c in chunks:
        assert "text" in c
        assert "metadata" in c


def test_process_file_empty(tmp_path: Path) -> None:
    """process_file on an empty file returns no chunks."""
    from pipeline import process_file

    txt = tmp_path / "empty.txt"
    txt.write_text("")

    chunks = process_file(str(txt))
    assert chunks == []


def test_process_directory_single_file(tmp_path: Path) -> None:
    """process_directory walks a directory and processes supported files."""
    from pipeline import process_directory

    (tmp_path / "a.txt").write_text("File A content.\n")
    (tmp_path / "b.txt").write_text("File B content.\n")

    chunks = process_directory(str(tmp_path))
    assert len(chunks) >= 2


def test_process_directory_nested(tmp_path: Path) -> None:
    """process_directory recurses into subdirectories."""
    from pipeline import process_directory

    subdir = tmp_path / "sub"
    subdir.mkdir()
    (tmp_path / "root.txt").write_text("Root file.\n")
    (subdir / "nested.txt").write_text("Nested file.\n")

    chunks = process_directory(str(tmp_path))
    assert len(chunks) >= 2


def _mock_format_future(text, source_type="pdf", **kwargs):
    future: Future = Future()
    future.set_result({
        "title": "Test",
        "tags": ["test"],
        "metadata": {"sections": [], "entities": []},
        "body": text,
    })
    return future


def test_process_file_hybrid_no_llm_fallback(monkeypatch) -> None:
    """process_file_hybrid returns a valid structure even without LLM service.

    The pipeline should handle missing embedding services gracefully and still
    return chunks with metadata (no silent failures).
    """
    from pipeline import process_file_hybrid
    # Mock the async formatter so we don't hit real LLM endpoint (3600s timeout)
    monkeypatch.setattr("formatters.format_text_async", _mock_format_future)

    # Create a small text file — no external parser needed
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        Path(path).write_text("Short test document.\n")

        result = process_file_hybrid(
            filepath=path,
            doc_id="test_doc",
            store_path=None,  # No DB persistence — just test the pipeline
        )

        assert isinstance(result, dict)
        assert "chunks" in result
        assert "document" in result
        # Even without LLM/embedding, chunks list should be present
        assert isinstance(result["chunks"], list)
    finally:
        Path(path).unlink(missing_ok=True)


def test_process_file_hybrid_store_path_local_backend(tmp_path: Path, monkeypatch) -> None:
    """process_file_hybrid with store_path succeeds using a LocalEmbedder-shaped backend.

    Regression test: the storage cleanup calls e.close() in a finally block, which
    requires every embedder backend to expose close(). Uses a plain stub (not
    MagicMock) so a missing interface method raises AttributeError instead of
    being silently auto-created.
    """
    from unittest.mock import patch

    from pipeline import process_file_hybrid

    monkeypatch.setattr("formatters.format_text_async", _mock_format_future)

    class StubLocalEmbedder:
        """Mirrors LocalEmbedder's public interface — deliberately not a MagicMock."""

        def store_chunks(self, chunks, doc_id="doc_0"):
            return [{**c, "source_doc_id": doc_id, "chunk_index": i,
                     "word_count": len(c["text"].split()), "embedding": [0.1] * 1024}
                    for i, c in enumerate(chunks)]

        def store_document(self, title, tags, text_summary, source_file, total_chunks):
            return {"title": title, "tags": tags, "text_summary": text_summary[:1000],
                    "source_file": source_file, "total_chunks": total_chunks,
                    "embedding": [0.1] * 1024}

        def close(self):
            self.closed = True

    class StubStore:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def upsert_chunks(self, chunks, doc_id="doc_0"):
            pass

        def upsert_document(self, **kwargs):
            pass

    embedder = StubLocalEmbedder()
    txt = tmp_path / "sample.txt"
    txt.write_text("Short test document.\n")
    db_path = tmp_path / "test.db"

    with patch("embedders.create_embedder", return_value=embedder), \
         patch("storage.sqlite_vec.SQLiteVecStore", StubStore):
        result = process_file_hybrid(filepath=str(txt), doc_id="test_doc",
                                     store_path=str(db_path))

    assert result["db_path"] == str(db_path)
    assert embedder.closed is True


def test_process_file_with_md(tmp_path: Path, monkeypatch) -> None:
    """process_file_with_md generates a markdown file."""
    monkeypatch.setattr("formatters.format_text_async", _mock_format_future)
    from pipeline import process_file_with_md
    # Mock the async formatter so we don't hit real LLM endpoint (3600s timeout)

    txt = tmp_path / "sample.txt"
    txt.write_text("# Title\n\nSome content.\n")

    md_path = process_file_with_md(str(txt), output_dir=str(tmp_path))
    assert md_path is not None
    assert Path(md_path).exists()


def test_process_directory_extensions_filter(tmp_path: Path) -> None:
    """process_directory respects the extensions filter."""
    from pipeline import process_directory

    (tmp_path / "a.txt").write_text("Text file.\n")
    (tmp_path / "b.md").write_text("# Markdown\n")

    # Only .md files should be processed (extensions passed without leading dot)
    chunks = process_directory(str(tmp_path), extensions=["md"])
    assert len(chunks) >= 1
    for c in chunks:
        assert ".md" in c["metadata"]["source"]


def test_process_file_chunk_size(tmp_path: Path) -> None:
    """process_file respects the chunk_size parameter."""
    from pipeline import process_file

    long_text = "Word. " * 2000  # ~8000 chars
    txt = tmp_path / "long.txt"
    txt.write_text(long_text)

    chunks = process_file(str(txt), chunk_size=512)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c["text"]) <= 512 + 64  # Allow some overlap buffer


def test_process_directory_no_files(tmp_path: Path) -> None:
    """process_directory on an empty directory returns no chunks."""
    from pipeline import process_directory

    chunks = process_directory(str(tmp_path))
    assert chunks == []
