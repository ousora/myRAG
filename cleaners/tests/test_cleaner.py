"""Tests for text cleaner."""


import sys; sys.path.insert(0, '/home/colinvan/workspace')

from myrag.cleaners import clean_text, TextCleaner


def test_normal_text_unchanged():
    """Normal text should be preserved (only excessive whitespace collapsed)."""
    raw = "Hello world\nGoodbye"  # Single newline — cleaner preserves this
    cleaned = clean_text(raw)
    assert cleaned == "Hello world Goodbye", f"Got: {cleaned!r}"


def test_control_characters_removed():
    """PDFs often contain \x07 BELL chars — should be replaced with spaces."""
    raw = "Line 1\x07\n\x07Line 2"
    cleaned = clean_text(raw)
    assert "\x07" not in cleaned, f"BELL char still present: {cleaned!r}"


def test_page_breaks_removed():
    """Page breaks and separators should be removed."""
    raw = "Before\n--- PAGE 1 ---\nAfter"
    cleaned = clean_text(raw)
    # After cleaning, "---" with surrounding text should not appear as a page separator
    assert "PAGE 1" not in cleaned.lower(), f"Page break still present: {cleaned!r}"


def test_whitespace_collapsed():
    """Excess whitespace (tabs, multiple newlines) should be normalized."""
    raw = "Hello\tWorld\n\n\n\nGoodbye"
    cleaned = clean_text(raw)
    assert "\t" not in cleaned, f"Tab still present: {cleaned!r}"


def test_empty_input_returns_empty():
    """Empty or whitespace-only input should return empty string."""
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""
