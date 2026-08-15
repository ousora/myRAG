"""Tests for formatters.tags tag extraction."""

from __future__ import annotations

import pytest

from formatters.tags import (
    extract_tags_from_body,
    _detect_body_script,
    _tokenize_latin,
    _tokenize_cjk,
    _extract_latin_proper_nouns,
    _extract_cjk_entities,
)


class TestDetectScript:
    def test_cjk_text(self) -> None:
        assert _detect_body_script("你好世界") == "cjk"

    def test_latin_text(self) -> None:
        assert _detect_body_script("Hello World") == "latin"

    def test_mixed_text(self) -> None:
        result = _detect_body_script("Hello 你好 World")
        assert result in ("cjk", "latin")

    def test_empty_text(self) -> None:
        assert _detect_body_script("") == "latin"

    def test_very_low_cjk_ratio(self) -> None:
        text = " ".join(["hello"] * 100) + " 你好"
        assert _detect_body_script(text) == "latin"


class TestTokenizeLatin:
    def test_basic_tokenization(self) -> None:
        body_tokens, title_tokens = _tokenize_latin("Hello world test", "My Title Here")
        assert "hello" in body_tokens
        assert "world" in body_tokens
        assert "title" in title_tokens
        assert "here" in title_tokens

    def test_short_words_excluded(self) -> None:
        body_tokens, _ = _tokenize_latin("A AB ABC", "Title")
        assert "abc" in body_tokens
        assert "ab" not in body_tokens


class TestTokenizeCjk:
    def test_cjk_bigrams(self) -> None:
        body_tokens, _ = _tokenize_cjk("人工智能技术", "标题")
        assert "人工" in body_tokens or "智能" in body_tokens

    def test_ws_tokens(self) -> None:
        body_tokens, _ = _tokenize_cjk("机器学习 深度学习", "标题")
        assert "机器学习" in body_tokens


class TestExtractProperNouns:
    def test_latin_proper_nouns(self) -> None:
        nouns = _extract_latin_proper_nouns("Apple Inc. announced the iPhone 15")
        assert any("apple" in n.lower() for n in nouns)

    def test_cjk_entities(self) -> None:
        entities = _extract_cjk_entities("苹果公司发布了iPhone")
        assert any("苹果" in e for e in entities)


class TestExtractTagsFromBody:
    def test_latin_tags(self) -> None:
        body = """
        Machine learning is a subset of artificial intelligence.
        Deep learning uses neural networks for classification.
        The TensorFlow framework is widely used in deep learning.
        """
        tags = extract_tags_from_body(body, "Introduction to Machine Learning")
        assert isinstance(tags, list)
        assert len(tags) <= 5
        # Should contain domain-specific terms
        assert any("machine" in t.lower() or "learning" in t.lower() or "deep" in t.lower() for t in tags)

    def test_cjk_tags(self) -> None:
        body = "人工智能和机器学习是现代技术的重要组成部分。深度学习是机器学习的子集。"
        tags = extract_tags_from_body(body, "人工智能导论")
        assert isinstance(tags, list)
        assert len(tags) <= 5

    def test_empty_body(self) -> None:
        tags = extract_tags_from_body("", "Empty Title")
        assert isinstance(tags, list)

    def test_tags_are_lowercase(self) -> None:
        body = "The Python programming language is great for data science."
        tags = extract_tags_from_body(body, "Python Programming")
        for tag in tags:
            assert tag == tag.lower()

    def test_generic_words_filtered(self) -> None:
        body = "This is a document about the system and data."
        tags = extract_tags_from_body(body, "System Data Document")
        assert "system" not in tags or len(tags) == 0
        assert "data" not in tags or len(tags) == 0
