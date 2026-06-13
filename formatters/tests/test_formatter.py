"""Tests for formatters module."""


import json
from concurrent.futures import Future
from unittest.mock import Mock, patch

import sys; sys.path.insert(0, '/home/colinvan/workspace')

import pytest

from myrag.formatters import format_text, format_text_async
from myrag.formatters.prompts import get_system_prompt


VALID_RESPONSE = {
    "title": "Test Article",
    "tags": ["test", "python"],
    "metadata": {
        "source_type": "web",
        "total_words": 150,
        "chunk_count": 2,
        "sections": [{"level": 2, "title": "Introduction"}, {"level": 3, "title": "Usage"}],
        "created_at": "2026-06-13T14:30:00Z",
        "modified_date": None,
    },
    "chunks": [
        {"id": 1, "section_path": ["Introduction"], "text": "First chunk content."},
        {"id": 2, "section_path": ["Usage"], "text": "Second chunk content."},
    ],
}


class MockResponse:
    """Simple mock response that returns a real dict from json()."""

    def __init__(self, data: dict):
        self.data = data

    def json(self) -> dict:
        return self.data

    def raise_for_status(self) -> None:
        pass


def _mock_response(json_data: dict) -> Mock:
    """Create a mock httpx.post that returns our test response."""
    
    resp = MockResponse({"choices": [{"message": {"content": json.dumps(json_data)}}]})
    post_mock = Mock(return_value=resp)
    return post_mock


class TestFormatText:
    def test_normal_text_returns_dict(self):
        """Normal text should return a dict with expected keys."""
        raw_text = "This is a sample article about Python programming."

        with patch("myrag.formatters.httpx.post", _mock_response(VALID_RESPONSE)):
            result = format_text(raw_text, source_type="web")

        assert isinstance(result, dict)
        assert "title" in result
        assert "tags" in result
        assert "metadata" in result
        assert "chunks" in result
        assert len(result["chunks"]) == 2
        assert result["chunks"][0]["id"] == 1

    def test_short_text_single_chunk(self):
        """Short text (<50 words) should produce a single chunk."""
        short_raw = "A brief note."

        # LLM will respond with 1 chunk for short input
        response_data = {**VALID_RESPONSE, "metadata": {**VALID_RESPONSE["metadata"], "chunk_count": 1}}
        response_data["chunks"] = [{"id": 1, "section_path": ["Note"], "text": "A brief note."}]

        with patch("myrag.formatters.httpx.post", _mock_response(response_data)):
            result = format_text(short_raw, source_type="web")

        assert len(result["chunks"]) == 1

    def test_empty_input_raises_valueerror(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="Input text is empty"):
            format_text("")

    def test_whitespace_only_input_raises_valuevalue(self):
        """Whitespace-only input should raise ValueError."""
        with pytest.raises(ValueError, match="Input text is empty"):
            format_text("   \n  ")

    def test_llm_invalid_json_raises_error(self):
        """LLM returning invalid JSON should raise ValueError."""
        bad_response = Mock()
        bad_response.json.return_value = {
            "choices": [{"message": {"content": "not valid json"}}]
        }
        bad_response.raise_for_status = Mock()

        with patch("myrag.formatters.httpx.post", return_value=bad_response):
            with pytest.raises(ValueError, match="LLM returned invalid JSON"):
                format_text("some text")

    def test_llm_missing_choices_raises_error(self):
        """Missing 'choices' in response should raise ValueError."""
        bad_response = Mock()
        bad_response.json.return_value = {}  # No choices key
        bad_response.raise_for_status = Mock()

        with patch("myrag.formatters.httpx.post", return_value=bad_response):
            with pytest.raises(ValueError, match="LLM returned invalid format"):
                format_text("some text")


class TestFormatTextAsync:
    def test_format_text_async_returns_future(self):
        """format_text_async should return a Future."""
        future = format_text_async("test", source_type="web")
        assert isinstance(future, Future)

    def test_format_text_async_result_matches_sync(self):
        """async result should match sync call when mocked."""
        with patch("myrag.formatters.httpx.post", _mock_response(VALID_RESPONSE)):
            future = format_text_async("test text", source_type="web")
            result = future.result(timeout=10)

        assert isinstance(result, dict)
        assert "title" in result


class TestGetSystemPrompt:
    def test_returns_string(self):
        assert isinstance(get_system_prompt(), str)

    def test_includes_source_type(self):
        prompt = get_system_prompt(source_type="markdown")
        assert "markdown" in prompt
