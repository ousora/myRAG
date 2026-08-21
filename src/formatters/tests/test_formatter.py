"""Tests for formatters module."""


import json
from concurrent.futures import Future
from unittest.mock import Mock, patch

import pytest

from formatters import (
    _fix_bare_quotes_in_body_field,
    _preprocess_json,
    call_llm,
    format_text,
    format_text_async,
)
from formatters.prompts import get_system_prompt

VALID_RESPONSE = {
    "title": "Test Article",
    "tags": ["test", "python"],
    "metadata": {
        "source_type": "web",
        "total_words": 150,
        "sections": [{"level": 2, "title": "Introduction"}, {"level": 3, "title": "Usage"}],
        "created_at": "2026-06-13T14:30:00Z",
        "modified_date": None,
    },
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
    return Mock(return_value=resp)


class TestFormatText:
    def test_normal_text_returns_dict(self):
        """Normal text should return a dict with expected keys."""
        raw_text = "This is a sample article about Python programming."

        with patch("formatters.httpx.post", _mock_response(VALID_RESPONSE)):
            result = format_text(raw_text, source_type="web")

        assert isinstance(result, dict)
        assert "title" in result
        assert "tags" in result
        assert "metadata" in result

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

        with patch("formatters.httpx.post", return_value=bad_response):
            with pytest.raises(ValueError, match="no JSON-like content"):
                format_text("some text")

    def test_llm_missing_choices_raises_error(self):
        """Missing 'choices' in response should raise ValueError."""
        bad_response = Mock()
        bad_response.json.return_value = {}  # No choices key
        bad_response.raise_for_status = Mock()

        with patch("formatters.httpx.post", return_value=bad_response):
            with pytest.raises(ValueError, match="LLM returned invalid format"):
                format_text("some text")


class TestFormatTextAsync:
    def test_format_text_async_returns_future(self):
        """format_text_async should return a Future."""
        future = format_text_async("test", source_type="web")
        assert isinstance(future, Future)

    def test_format_text_async_result_matches_sync(self):
        """Async result should match sync call when mocked."""
        with patch("formatters.httpx.post", _mock_response(VALID_RESPONSE)):
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


class TestPreprocessJson:
    """Tests for _preprocess_json — strip markdown fences, extract first JSON object."""

    def test_plain_valid_json(self):
        result = _preprocess_json('{"a": 1}')
        assert result is not None
        assert json.loads(result) == {"a": 1}

    def test_markdown_fence_json(self):
        raw = '```json\n{"x": [1,2,3]}\n```'
        result = _preprocess_json(raw)
        assert result is not None
        assert json.loads(result) == {"x": [1, 2, 3]}

    def test_no_fence_but_wrapped_with_text(self):
        raw = 'Here is the answer:\n{ "hello": "world" }'
        result = _preprocess_json(raw)
        assert result is not None
        assert '"hello"' in result and '"world"' in result

    def test_balanced_braces_returns_first_object(self):
        """Balanced brace matching should return only the first complete JSON object."""
        raw = '{"a": 1} explanation {"b": 2}'
        result = _preprocess_json(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"a": 1}, "Should stop at the first balanced }"

    def test_no_json_returns_none(self):
        assert _preprocess_json("just plain english text") is None

    def test_non_string_input_returns_none(self):
        assert _preprocess_json(12345) is None


class TestFixBareQuotes:
    """Tests for _fix_bare_quotes_in_body_field — escape unescaped quotes in body value."""

    def test_no_body_key_returns_none(self):
        content = '{"title": "Test"}'
        assert _fix_bare_quotes_in_body_field(content) is None

    def test_empty_content_returns_none(self):
        assert _fix_bare_quotes_in_body_field("") is None

    def test_simple_valid_json_returns_none(self):
        """Valid JSON with no bare quotes — function returns None (no fix needed)."""
        content = '{"title": "Test", "body": "Hello world"}'
        result = _fix_bare_quotes_in_body_field(content)
        assert result is None

    def test_valid_json_with_proper_escapes_returns_none(self):
        """JSON with properly escaped quotes — function returns None (no fix needed)."""
        content = '{"title": "Test", "body": "He said \\"hi\\""}'
        result = _fix_bare_quotes_in_body_field(content)
        assert result is None

    def test_valid_json_is_parseable(self):
        """Confirm the original valid JSON parses correctly."""
        content = '{"title": "Test", "body": "Hello world"}'
        parsed = json.loads(content)
        assert parsed == {"title": "Test", "body": "Hello world"}

    def test_escaped_quotes_are_parseable(self):
        """Confirm the original escaped JSON parses correctly."""
        content = '{"title": "Test", "body": "He said \\"hi\\""}'
        parsed = json.loads(content)
        assert parsed["body"] == 'He said "hi"'

    def test_bare_quote_inside_value_gets_escaped(self):
        """Bare quotes inside body value get escaped so JSON becomes parseable."""
        q = chr(34)  # literal double-quote character
        content = '{"title": "Test", "body": "She said ' + q + "hello" + q + ' world"}'
        result = _fix_bare_quotes_in_body_field(content)
        assert result is not None
        parsed = json.loads(result)
        # After escaping, the body should contain the bare quotes as literal chars.
        assert isinstance(parsed["body"], str)

    def test_quote_before_body_key_not_matched(self):
        """Malformed body key (quotes inside the key name) — regex won't match."""
        content = '{"bo"dy": "Test"}'
        result = _fix_bare_quotes_in_body_field(content)
        assert result is None, f"Malformed body key should not trigger fix: got {result!r}"


class TestCallLlmSchemaRetry:
    """Regression: the schema-rejected retry branch was unreachable because.

    the HTTP response was never recovered from the exception chain.
    """

    def _recording_post(self, seen: list, responses: list):
        """Mock httpx.post that snapshots each payload.

        The retry mutates the same dict in place, so Mock's call_args
        would alias it.

        """
        import copy

        def _post(url, json=None, timeout=None, **kwargs):
            seen.append(copy.deepcopy(json))
            return responses[len(seen) - 1]
        return _post

    def test_schema_rejection_retries_without_schema(self):
        import httpx

        req = httpx.Request("POST", "http://localhost/v1/chat")
        err = httpx.Response(
            500,
            text='{"error": {"message": "failed to parse schema: PEG error at char 3"}}',
            request=req,
        )
        ok = httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(VALID_RESPONSE)}}]},
            request=req,
        )
        seen: list = []
        post_mock = self._recording_post(seen, [err, ok])

        with patch("formatters.httpx.post", post_mock):
            result = call_llm("system prompt", "user text", schema={"type": "object"})

        assert result == VALID_RESPONSE
        assert len(seen) == 2
        # First attempt carried the schema; retry must drop it.
        assert "response_format" in seen[0]
        assert "response_format" not in seen[1]

    def test_non_schema_500_reraises(self):
        import httpx

        req = httpx.Request("POST", "http://localhost/v1/chat")
        err = httpx.Response(500, text='{"error": "internal server panic"}', request=req)
        seen: list = []
        post_mock = self._recording_post(seen, [err])

        with patch("formatters.httpx.post", post_mock):
            with pytest.raises(RuntimeError, match="LLM API request failed"):
                call_llm("system prompt", "user text", schema={"type": "object"})

        assert len(seen) == 1


class TestEffectiveChunkThreshold:
    """Token budget must stay constant across language mixes.

    Harmonic-mean scaling with no discontinuities.
    """

    def _threshold(self, ratio: float, base: int = 20000) -> int:
        from unittest.mock import patch as _patch

        from formatters._internal import effective_chunk_threshold

        cfg = Mock(chunk_threshold_chars=base)
        with _patch("formatters._internal._get_config", return_value=cfg):
            # Synthesize a text whose CJK/latin char ratio ≈ *ratio*.
            n_cjk = int(1000 * ratio)
            n_latin = 1000 - n_cjk
            text = "汉" * n_cjk + "a" * n_latin
            return effective_chunk_threshold(text)

    def test_english_unchanged(self):
        assert self._threshold(0.0) == 20000

    def test_pure_cjk_quarter(self):
        assert self._threshold(1.0) == 5000

    def test_monotonic_decreasing(self):
        import itertools

        ratios = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        values = [self._threshold(r) for r in ratios]
        assert all(a >= b for a, b in itertools.pairwise(values))

    def test_no_cliff_at_half(self):
        """Old piecewise formula jumped ~3.7x at r=0.5; new one is smooth."""
        below = self._threshold(0.49)
        at = self._threshold(0.50)
        assert at > 0  # sanity
        assert below / at < 1.3  # gentle step, not a cliff

    def test_token_budget_constant(self):
        """chars/threshold ≈ constant token budget across mixes."""
        en_cpt, cjk_cpt = 4.0, 1.0
        for ratio in (0.25, 0.5, 0.75):
            t = self._threshold(ratio)
            budget = t * (ratio / cjk_cpt + (1 - ratio) / en_cpt)
            assert budget == pytest.approx(5000, rel=0.05)
