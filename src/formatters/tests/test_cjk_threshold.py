"""Tests for CJK-aware chunk threshold logic."""

import pytest
from formatters import _detect_cjk_ratio, effective_chunk_threshold


@pytest.fixture(autouse=True)
def clear_config_cache():
    """Clear the lru_cache on get_config() between tests to avoid stale values."""
    from config import get_config
    get_config.cache_clear()
    yield
    get_config.cache_clear()


class TestDetectCjkRatio:
    """Test CJK character ratio detection."""

    def test_empty_text_returns_zero(self):
        assert _detect_cjk_ratio("") == 0.0

    def test_pure_english_text(self):
        text = "Hello world this is English text with some numbers 123"
        ratio = _detect_cjk_ratio(text)
        assert ratio == 0.0, f"Pure English should have 0 CJK ratio, got {ratio}"

    def test_pure_cjk_text(self):
        text = "这是一个中文测试文档包含多个段落。"
        ratio = _detect_cjk_ratio(text)
        assert ratio > 0.9, f"Pure CJK should have ~1.0 ratio, got {ratio}"

    def test_mixed_text(self):
        # Mixed English and CJK — expect roughly equal split.
        text = "Hello world 你好世界 Python编程测试."
        ratio = _detect_cjk_ratio(text)
        assert 0.3 <= ratio <= 0.7, f"Mixed text ratio out of range: {ratio}"

    def test_only_digits_and_punctuation(self):
        # Digits and punctuation count as CJK-like (non-Latin).
        text = "12345 !@#$%^&*()"
        ratio = _detect_cjk_ratio(text)
        assert ratio > 0.5, f"Only digits/punct should be mostly CJK: {ratio}"


class TestEffectiveChunkThreshold:
    """Test CJK-aware threshold adjustment."""

    def test_english_text_uses_base_threshold(self):
        from config import get_config
        base = get_config().chunk_threshold_chars
        cfg = effective_chunk_threshold("Hello world English text.")
        # Should return base since ratio is 0.
        assert cfg == base, f"English should use base threshold: {cfg}"

    def test_pure_cjk_text_reduces_threshold(self):
        from config import get_config
        base = get_config().chunk_threshold_chars
        cjk_text = "这是一个中文测试文档。" * 100
        cfg = effective_chunk_threshold(cjk_text)
        # Should be ~base/4 for pure CJK (since CJK is 1 char/token vs English 4).
        assert cfg <= base // 3, f"CJK threshold should be reduced: {cfg}"

    def test_mixed_text_interpolates(self):
        from config import get_config
        base = get_config().chunk_threshold_chars
        mixed = "Hello world " + "你好世界 " * 20
        cfg = effective_chunk_threshold(mixed)
        # Should be between base/4 and base for mixed text.
        assert base // 4 <= cfg <= base, f"Mixed threshold out of range: {cfg}"

    def test_very_short_text_still_works(self):
        from config import get_config
        base = get_config().chunk_threshold_chars
        # Edge case: very short text shouldn't cause issues.
        cfg = effective_chunk_threshold("Hi")
        assert cfg == base
