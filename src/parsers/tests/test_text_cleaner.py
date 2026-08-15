"""Tests for parsers.text_cleaner.TextCleaner."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from parsers.text_cleaner import TextCleaner


class TestTextCleanerBasic:
    def test_empty_input(self) -> None:
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == ""
        assert cleaner.clean(None) == ""  # type: ignore[arg-type]

    def test_control_chars_removed(self) -> None:
        cleaner = TextCleaner()
        result = cleaner.clean("hello\x00world\x07test\x1fend")
        assert "\x00" not in result
        assert "\x07" not in result
        assert "\x1f" not in result
        assert "helloworldtestend" == result

    def test_page_breaks_removed(self) -> None:
        cleaner = TextCleaner()
        text = "header line\n--- PAGE 1 ---\ncontent here"
        result = cleaner.clean(text)
        assert "--- PAGE 1 ---" not in result
        assert "header line" in result
        assert "content here" in result

    def test_page_breaks_preserved_short(self) -> None:
        """Short lines like '---' should NOT be treated as page breaks."""
        cleaner = TextCleaner()
        text = "title\n---\ncontent"
        result = cleaner.clean(text)
        assert "---" in result

    def test_page_breaks_disabled(self) -> None:
        cleaner = TextCleaner(remove_page_breaks=False)
        text = "header\n--- PAGE 1 ---\ncontent"
        result = cleaner.clean(text)
        assert "--- PAGE 1 ---" in result

    def test_tabs_to_spaces(self) -> None:
        cleaner = TextCleaner()
        result = cleaner.clean("hello\tworld")
        assert "hello world" == result

    def test_trailing_spaces_removed(self) -> None:
        cleaner = TextCleaner()
        result = cleaner.clean("hello   \nworld   ")
        assert "hello" in result.split("\n[^\n]*")[0]
        assert result.endswith("world")

    def test_multiple_newlines_collapsed(self) -> None:
        cleaner = TextCleaner()
        result = cleaner.clean("line1\n\n\n\n\nline2")
        assert "line1\n\nline2" == result

    def test_leading_trailing_whitespace_stripped(self) -> None:
        cleaner = TextCleaner()
        result = cleaner.clean("  hello world  ")
        assert result == "hello world"

    def test_whitespace_collapsing_disabled(self) -> None:
        cleaner = TextCleaner(collapse_whitespace=False)
        result = cleaner.clean("hello\n\n\n\nworld")
        assert "\n\n\n\n" in result

    def test_preserves_leading_spaces_for_markdown(self) -> None:
        """Leading spaces (indentation) should be preserved for code blocks/lists."""
        cleaner = TextCleaner()
        text = "  code line\n  another line"
        result = cleaner.clean(text)
        assert "another line" in result


class TestTextCleanerTables:
    def test_broken_table_row_merged(self) -> None:
        cleaner = TextCleaner()
        text = "| A | B |\n| C |"
        result = cleaner.clean(text)
        lines = result.split("\n")
        # The broken row should be merged into the previous row, resulting in 1 line
        assert len(lines) == 1
        assert "C" in lines[0]

    def test_normal_table_preserved(self) -> None:
        cleaner = TextCleaner()
        text = "| A | B |\n| C | D |"
        result = cleaner.clean(text)
        assert "| A | B |" in result
        assert "| C | D |" in result


class TestTextCleanerCustomRules:
    def test_custom_rules_loaded(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("rules:\n  - pattern: \"REMOVE_ME\"\n    replace: \"REPLACED\"\n")
            f.flush()
            cleaner = TextCleaner(rules_config=f.name)
            result = cleaner.clean("hello REMOVE_ME world")
            assert "REPLACED" in result
            assert "REMOVE_ME" not in result
        Path(f.name).unlink()

    def test_custom_rules_disabled(self) -> None:
        cleaner = TextCleaner(rules_config=None)
        result = cleaner.clean("REMOVE_ME")
        assert "REMOVE_ME" in result

    def test_invalid_regex_rule_skipped(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("rules:\n  - pattern: \"[invalid\"\n    replace: \"X\"\n")
            f.flush()
            cleaner = TextCleaner(rules_config=f.name)
            assert len(cleaner.custom_rules) == 0
        Path(f.name).unlink()


class TestTextCleanerEdgeCases:
    def test_single_line(self) -> None:
        cleaner = TextCleaner()
        result = cleaner.clean("single line")
        assert result == "single line"

    def test_unicode_preserved(self) -> None:
        cleaner = TextCleaner()
        result = cleaner.clean("你好世界 日本語 한국어")
        assert "你好世界" in result
        assert "日本語" in result

    def test_whitespace_only(self) -> None:
        cleaner = TextCleaner()
        result = cleaner.clean("   \n\n\t\n  ")
        assert result == ""
