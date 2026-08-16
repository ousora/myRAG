"""Self-check for the deterministic markdown normalizer.

Run:  uv run python src/parsers/tests/test_normalizer.py
"""

from parsers.markdown_normalizer import normalize_markdown


def main() -> None:
    raw = """Some intro text.

Section A
Content under section A.

subsection
Deeper content.

Section B
Content under B.

1) First item
2) Second item
3) Third item

See http://example.com for more.
Visit https://docs.example.org/path?q=1 now.

This sentence has an **unclosed bold.
Next sentence has *unclosed italic.

| Col1 | Col2 |
| A | B |
| C | D E F |
"""

    out = normalize_markdown(raw)

    # 1. Standalone short lines become ## headings
    assert "## Section A" in out, "Section A should be promoted"
    assert "## Section B" in out, "Section B should be promoted"
    assert "## subsection" in out, "subsection should be promoted"

    # 2. List markers normalized
    assert "1. First item" in out, "1) should become 1."
    assert "2. Second item" in out, "2) should become 2."

    # 3. Bare URLs become markdown links
    assert "[http://example.com](http://example.com)" in out, "bare URL should be linked"
    assert "[https://docs.example.org/path?q=1](https://docs.example.org/path?q=1)" in out

    # 4. Unclosed bold/italic get closed (each appends a closing marker)
    tail = out[out.rfind("bold."):]
    assert "**" in tail, f"bold should be closed; tail={tail!r}"
    assert "*" in tail, f"italic should be closed; tail={tail!r}"

    # 5. Table cells aligned
    assert "| Col1 | Col2 |" in out, "table header should be aligned"
    assert "| A | B |" in out, "table row should be aligned"
    assert "| C | D E F |" in out, "table row preserved"

    print("OK: all 5 normalizations applied")
    print("---")
    print(out)


if __name__ == "__main__":
    main()