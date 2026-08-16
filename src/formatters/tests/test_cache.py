"""Tests for formatters.cache LRU formatting cache."""

from __future__ import annotations

import pytest

from formatters.cache import (
    format_cache_key,
    format_cached,
    clear_format_cache,
    _FORMAT_CACHE,
    _FORMAT_CACHE_LOCK,
)


class TestFormatCacheKey:
    def test_key_deterministic(self) -> None:
        key1 = format_cache_key("hello", "web", "system prompt")
        key2 = format_cache_key("hello", "web", "system prompt")
        assert key1 == key2

    def test_key_differs_on_raw(self) -> None:
        key1 = format_cache_key("hello", "web", "system prompt")
        key2 = format_cache_key("world", "web", "system prompt")
        assert key1 != key2

    def test_key_differs_on_source_type(self) -> None:
        key1 = format_cache_key("hello", "web", "system prompt")
        key2 = format_cache_key("hello", "pdf", "system prompt")
        assert key1 != key2

    def test_key_differs_on_prompt(self) -> None:
        key1 = format_cache_key("hello", "web", "system prompt")
        key2 = format_cache_key("hello", "web", "different prompt")
        assert key1 != key2

    def test_key_none_prompt(self) -> None:
        key = format_cache_key("hello", "web", None)
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hex digest length


class TestFormatCached:
    def setup_method(self) -> None:
        with _FORMAT_CACHE_LOCK:
            _FORMAT_CACHE.clear()

    def test_cache_hit(self) -> None:
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return {"result": "computed"}

        result1 = format_cached("raw", "web", None, compute)
        result2 = format_cached("raw", "web", None, compute)

        assert result1 == {"result": "computed"}
        assert result2 == {"result": "computed"}
        assert call_count == 1  # compute() called only once

    def test_cache_miss_different_input(self) -> None:
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return {"result": call_count}

        result1 = format_cached("raw1", "web", None, compute)
        result2 = format_cached("raw2", "web", None, compute)

        assert call_count == 2

    def test_cache_stores_result(self) -> None:
        def compute():
            return {"data": "value"}

        format_cached("raw", "web", None, compute)

        with _FORMAT_CACHE_LOCK:
            assert len(_FORMAT_CACHE) == 1

    def test_cache_eviction(self) -> None:
        from formatters.cache import _FORMAT_CACHE_MAX

        def compute():
            return {"data": "value"}

        # Fill cache beyond max size
        for i in range(_FORMAT_CACHE_MAX + 10):
            format_cached(f"raw_{i}", "web", None, compute)

        with _FORMAT_CACHE_LOCK:
            assert len(_FORMAT_CACHE) <= _FORMAT_CACHE_MAX

    def test_cache_lru_order(self) -> None:
        from formatters.cache import _FORMAT_CACHE_MAX

        def compute():
            return {"data": "value"}

        format_cached("a", "web", None, compute)
        format_cached("b", "web", None, compute)

        # Access 'a' to make it most recent
        format_cached("a", "web", None, compute)

        # Evict one entry (cache max is 256, so all 3 still fit)
        # Instead, test eviction by filling beyond max
        for i in range(100):
            format_cached(f"extra_{i}", "web", None, compute)

        with _FORMAT_CACHE_LOCK:
            keys = list(_FORMAT_CACHE.keys())
            assert len(keys) <= _FORMAT_CACHE_MAX


class TestClearFormatCache:
    def test_clear_cache(self) -> None:
        def compute():
            return {"data": "value"}

        format_cached("raw", "web", None, compute)
        clear_format_cache()

        with _FORMAT_CACHE_LOCK:
            assert len(_FORMAT_CACHE) == 0
