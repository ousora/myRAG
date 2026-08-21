"""Tests for the deterministic markdown normalizer.

Run:  uv run pytest src/parsers/tests/test_normalizer.py -v
"""

from parsers.markdown_normalizer import normalize_markdown


def test_heading_promotion():
    raw = """Some intro text.

Section A
Content under section A.

subsection
Deeper content.

Section B
Content under B.
"""
    out = normalize_markdown(raw)
    assert "## Section A" in out
    assert "## Section B" in out
    assert "## subsection" in out


def test_list_normalization():
    raw = """1) First item
2) Second item
3) Third item
"""
    out = normalize_markdown(raw)
    assert "1. First item" in out
    assert "2. Second item" in out
    assert "3. Third item" in out


def test_bullet_normalization():
    raw = """• Bullet one
• Bullet two
* Bullet three
+ Bullet four
"""
    out = normalize_markdown(raw)
    assert "- Bullet one" in out
    assert "- Bullet two" in out
    assert "- Bullet three" in out
    assert "- Bullet four" in out


def test_link_formatting():
    raw = """See http://example.com for more.
Visit https://docs.example.org/path?q=1 now.
"""
    out = normalize_markdown(raw)
    assert "[http://example.com](http://example.com)" in out
    assert "[https://docs.example.org/path?q=1](https://docs.example.org/path?q=1)" in out


def test_link_formatting_with_inline_code():
    raw = """See `http://example.com` for more.
URL: https://example.org outside code.
"""
    out = normalize_markdown(raw)
    # URL inside backticks should NOT be formatted
    assert "`http://example.com`" in out
    # URL outside code SHOULD be formatted
    assert "[https://example.org](https://example.org)" in out


def test_unclosed_bold():
    raw = """This sentence has an **unclosed bold.
"""
    out = normalize_markdown(raw)
    tail = out[out.rfind("bold."):]
    assert "**" in tail


def test_unclosed_italic():
    raw = """Next sentence has *unclosed italic.
"""
    out = normalize_markdown(raw)
    tail = out[out.rfind("italic."):]
    assert "*" in tail


def test_table_alignment():
    raw = """| Col1 | Col2 |
| A | B |
| C | D E F |
"""
    out = normalize_markdown(raw)
    assert "| Col1 | Col2 |" in out
    assert "| A | B |" in out
    assert "| C | D E F |" in out


def test_code_block_skipped():
    raw = """```python
print("http://example.com")
```
"""
    out = normalize_markdown(raw)
    assert "[http://example.com](http://example.com)" not in out


def test_empty_input():
    assert normalize_markdown("") == ""
    assert normalize_markdown("   ") == ""
    assert normalize_markdown("\n\n") == ""


def test_already_valid_markdown():
    raw = """# Title

## Section

- Item 1
- Item 2
"""
    out = normalize_markdown(raw)
    assert "# Title" in out
    assert "## Section" in out
    assert "- Item 1" in out


def test_mixed_content():
    raw = """Intro text.

Section A
Content here.

1) First item
2) Second item

See http://example.com for more.

| Col1 | Col2 |
| A | B |
"""
    out = normalize_markdown(raw)
    assert "## Section A" in out
    assert "1. First item" in out
    assert "[http://example.com](http://example.com)" in out
    assert "| Col1 | Col2 |" in out


def test_lettered_list():
    raw = """a) First
b) Second
"""
    out = normalize_markdown(raw)
    assert "a) First" in out
    assert "b) Second" in out


def test_numbered_list_with_dots():
    raw = """1. First
2. Second
"""
    out = normalize_markdown(raw)
    assert "1. First" in out
    assert "2. Second" in out


def test_fenced_code_with_language():
    raw = """```python
code here
```
"""
    out = normalize_markdown(raw)
    assert "```python" in out
    assert "```" in out


def test_inline_code_preserved():
    raw = """See `code` and `more` text.
"""
    out = normalize_markdown(raw)
    assert "`code`" in out
    assert "`more`" in out

def test_multiline_fence_interior_urls_untouched():
    """Regression test for multi-line fence URL formatting.

    Interior lines of a multi-line fence used to get URL formatting
    applied because only lines containing ``` were skipped.
    """
    raw = """Run this:

```bash
curl https://api.example.com/v1/data
echo done
```

Docs at https://docs.example.com/guide.
"""
    out = normalize_markdown(raw)
    # Interior fence line must stay untouched
    assert "curl https://api.example.com/v1/data\n" in out
    assert "[https://api.example.com](https://api.example.com)" not in out
    # URL outside the fence SHOULD be formatted
    assert "[https://docs.example.com/guide](https://docs.example.com/guide)" in out


def test_unclosed_fence_disables_link_formatting_to_end():
    """An unterminated fence protects everything after it (defensive)."""
    raw = "```bash\ncurl https://api.example.com/x\n"
    out = normalize_markdown(raw)
    assert "https://api.example.com/x" in out
    assert "](" not in out
