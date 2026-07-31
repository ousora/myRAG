"""Tests for process_directory_hybrid() in pipeline.core."""

import concurrent.futures
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestProcessDirectoryHybrid:
    """Tests for process_directory_hybrid — batch LLM-formatted processing."""

    def _make_parser(self, text="Hello world content"):
        p = Mock()
        p.parse = Mock(return_value=text)
        return p

    def _mock_format_result(self):
        return {
            "title": "Test Document",
            "tags": ["test"],
            "body": "# Test Document\n\nSome body text.\n\n## Section\n\nContent here.",
            "metadata": {"entities": []},
        }

    def _mock_future(self, result=None):
        f = Mock(spec=concurrent.futures.Future)
        f.result = Mock(return_value=result or self._mock_format_result())
        return f

    @pytest.fixture()
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_empty_directory_returns_empty_dict(self, tmp_dir):
        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock:
            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = None

            result = __import__("pipeline.core", fromlist=["process_directory_hybrid"]).process_directory_hybrid(tmp_dir)

        assert result == {}

    def test_no_supported_files_returns_empty_dict(self, tmp_dir):
        Path(tmp_dir, "data.csv").write_text("a,b,c", encoding="utf-8")
        Path(tmp_dir, "image.png").write_text("png data", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock:
            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = None

            result = __import__("pipeline.core", fromlist=["process_directory_hybrid"]).process_directory_hybrid(tmp_dir)

        assert result == {}

    def test_single_file_processed(self, tmp_dir):
        fp = Path(tmp_dir, "doc.txt")
        fp.write_text("test content", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk1", "section_path": ["Section"]}],
            render_mock.return_value = "# Test Document\n\nSome body text."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk1", "section_path": ["Section"]}]

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            result = mod.process_directory_hybrid(tmp_dir)

        assert len(result) == 1
        file_key = list(result.keys())[0]
        assert "doc.txt" in file_key
        assert "chunks" in result[file_key]

    def test_multiple_files_concurrent(self, tmp_dir):
        for i in range(3):
            Path(tmp_dir, f"doc{i}.txt").write_text(f"content {i}", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk", "section_path": ["Section"]}],
            render_mock.return_value = "# Test\n\nBody."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk", "section_path": ["Section"]}]

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            result = mod.process_directory_hybrid(tmp_dir)

        assert len(result) == 3

    def test_subdirectories_walked(self, tmp_dir):
        subdir = Path(tmp_dir, "sub")
        subdir.mkdir()
        Path(subdir, "nested.txt").write_text("nested content", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk", "section_path": ["Section"]}],
            render_mock.return_value = "# Test\n\nBody."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk", "section_path": ["Section"]}]

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            result = mod.process_directory_hybrid(tmp_dir)

        assert len(result) == 1
        file_key = list(result.keys())[0]
        assert "nested.txt" in file_key

    def test_unsupported_extension_skipped(self, tmp_dir):
        Path(tmp_dir, "data.csv").write_text("a,b,c", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk", "section_path": ["Section"]}],
            render_mock.return_value = "# Test\n\nBody."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk", "section_path": ["Section"]}]

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            result = mod.process_directory_hybrid(tmp_dir)

        assert result == {}

    def test_custom_extensions_filter(self, tmp_dir):
        Path(tmp_dir, "doc.md").write_text("# Doc", encoding="utf-8")
        Path(tmp_dir, "data.txt").write_text("data", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk", "section_path": ["Section"]}],
            render_mock.return_value = "# Test\n\nBody."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk", "section_path": ["Section"]}]

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            result = mod.process_directory_hybrid(tmp_dir, extensions=["md"])

        assert len(result) == 1
        file_key = list(result.keys())[0]
        assert "doc.md" in file_key

    def test_single_file_failure_does_not_abort_batch(self, tmp_dir):
        Path(tmp_dir, "good.txt").write_text("good content", encoding="utf-8")
        Path(tmp_dir, "bad.txt").write_text("bad content", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock, \
             patch("pipeline.core.process_file_hybrid") as hybrid_mock:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk", "section_path": ["Section"]}],
            render_mock.return_value = "# Test\n\nBody."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk", "section_path": ["Section"]}]

            def _side_effect(fp, **kw):
                if "bad.txt" in fp:
                    raise ValueError("parse error")
                return {"chunks": [{"text": "chunk", "section_path": ["S"]}], "document": {}}

            hybrid_mock.side_effect = _side_effect

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            result = mod.process_directory_hybrid(tmp_dir)

        assert len(result) == 2
        good_key = [k for k in result if "good.txt" in k][0]
        bad_key = [k for k in result if "bad.txt" in k][0]
        assert "chunks" in result[good_key]
        assert result[bad_key]["chunks"] == []

    def test_store_path_passed_to_each_file(self, tmp_dir):
        Path(tmp_dir, "doc.txt").write_text("content", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock, \
             patch("pipeline.core.process_file_hybrid") as hybrid_mock:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk", "section_path": ["Section"]}],
            render_mock.return_value = "# Test\n\nBody."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk", "section_path": ["Section"]}]
            hybrid_mock.return_value = {"chunks": [], "document": {}}

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            store_path = "/tmp/test.db"
            mod.process_directory_hybrid(tmp_dir, store_path=store_path)

        hybrid_mock.assert_called_once()
        call_kwargs = hybrid_mock.call_args
        assert call_kwargs[1]["store_path"] == store_path

    def test_md_output_dir_passed_to_each_file(self, tmp_dir):
        Path(tmp_dir, "doc.txt").write_text("content", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock, \
             patch("pipeline.core.process_file_hybrid") as hybrid_mock:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk", "section_path": ["Section"]}],
            render_mock.return_value = "# Test\n\nBody."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk", "section_path": ["Section"]}]
            hybrid_mock.return_value = {"chunks": [], "document": {}}

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            out_dir = "/tmp/md_out/"
            mod.process_directory_hybrid(tmp_dir, md_output_dir=out_dir)

        call_kwargs = hybrid_mock.call_args
        assert call_kwargs[1]["md_output_dir"] == out_dir

    def test_max_workers_passed_to_thread_pool(self, tmp_dir):
        """max_workers should be passed to ThreadPoolExecutor."""
        Path(tmp_dir, "doc.txt").write_text("content", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock, \
             patch("concurrent.futures.ThreadPoolExecutor") as tpe_cls:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk", "section_path": ["Section"]}],
            render_mock.return_value = "# Test\n\nBody."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk", "section_path": ["Section"]}]

            # Mock the context manager
            mock_tpe = Mock()
            mock_tpe.__enter__ = Mock(return_value=mock_tpe)
            mock_tpe.__exit__ = Mock(return_value=False)
            mock_result = [("doc.txt", {"chunks": [], "document": {}})]
            mock_tpe.map = Mock(return_value=mock_result)
            tpe_cls.return_value = mock_tpe

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            mod.process_directory_hybrid(tmp_dir, max_workers=2)

        tpe_cls.assert_called_once_with(max_workers=2)

    def test_doc_id_derived_from_relative_path(self, tmp_dir):
        """doc_id should be the file's relative path with slashes replaced by underscores."""
        subdir = Path(tmp_dir, "sub")
        subdir.mkdir()
        fp = Path(subdir, "doc.txt")
        fp.write_text("content", encoding="utf-8")

        captured_doc_ids = []

        def _capture_side_effect(fp, **kw):
            if "doc.txt" in fp:
                captured_doc_ids.append(kw.get("doc_id", ""))
            return {"chunks": [], "document": {}}

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock, \
             patch("pipeline.core.process_file_hybrid") as hybrid_mock:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk", "section_path": ["Section"]}],
            render_mock.return_value = "# Test\n\nBody."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk", "section_path": ["Section"]}]
            hybrid_mock.side_effect = _capture_side_effect

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            mod.process_directory_hybrid(tmp_dir)

        assert len(captured_doc_ids) == 1
        assert "sub" in captured_doc_ids[0]
        assert "doc.txt" in captured_doc_ids[0]
        assert "/" not in captured_doc_ids[0]

    def test_case_insensitive_extension_match(self, tmp_dir):
        Path(tmp_dir, "DOC.TXT").write_text("uppercase", encoding="utf-8")

        with patch("pipeline.core._get_config") as cfg_mock, \
             patch("pipeline.core.utils.resolve_parser") as resolve_mock, \
             patch("formatters.format_text_async") as fmt_mock, \
             patch("pipeline.core.Chunker") as chunker_cls, \
             patch("pipeline.core.markdown_utils.render_markdown_with_sections") as render_mock, \
             patch("pipeline.core.markdown_utils.strip_reference_sections") as strip_mock, \
             patch("pipeline.core.markdown_utils.match_entities_to_chunks") as match_mock:

            cfg_mock.return_value.format_timeout = 30
            resolve_mock.return_value = self._make_parser()
            fmt_mock.return_value = self._mock_future(self._mock_format_result())
            chunker_cls.return_value.chunk.return_value = [
                {"text": "chunk", "section_path": ["Section"]}],
            render_mock.return_value = "# Test\n\nBody."
            strip_mock.return_value = render_mock.return_value
            match_mock.return_value = [{"text": "chunk", "section_path": ["Section"]}]

            mod = __import__("pipeline.core", fromlist=["process_directory_hybrid"])
            result = mod.process_directory_hybrid(tmp_dir)

        assert len(result) == 1
