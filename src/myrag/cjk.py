r"""Canonical CJK character definitions shared across the pipeline.

Historically each module (storage, rerank, parsers, formatters) rolled its own
CJK regex with slightly different Unicode coverage — and at least one variant
built ranges as ``\\u4e00-9fff`` (only the start escaped), which silently
degrades to a literal-string match and disables CJK handling entirely.
This module is the single source of truth; ranges are built from integer
codepoints via ``chr()`` so astral-plane blocks work correctly:

- :data:`IDEOGRAPH_RANGES` — CJK Unified Ideographs, Extensions A–F.
  Used for FTS tokenization and word counting.
- :data:`KANA_RANGES` — Hiragana + Katakana. Japanese kana carry no word
  boundaries either, so text-matching code treats them like ideographs.
- :func:`contains_cjk` / :func:`count_cjk` — detection helpers covering
  ideographs + kana.
"""

from __future__ import annotations

import re

# Codepoint ranges for CJK Unified Ideographs Blocks A–F as (start, end) pairs.
# The basic U+4E00–9FFF range alone misses ~1% of modern Chinese text in
# Extensions A–F; G (U+30000–3134A) and later are historical/archaic and excluded.
IDEOGRAPH_RANGES: list[tuple[int, int]] = [
    (0x4E00, 0x9FFF),     # CJK Unified Ideographs (basic plane)
    (0x3400, 0x4DBF),     # Extension A — historical/archaic characters
    (0x20000, 0x2A6DF),   # Extension B — rare characters, names, place names
    (0x2A700, 0x2EBEF),   # Extensions C–F — rare variants and archaic forms
]

# Additional ranges treated as "no word boundaries" alongside ideographs.
KANA_RANGES: list[tuple[int, int]] = [
    (0x3040, 0x309F),     # Hiragana
    (0x30A0, 0x30FF),     # Katakana
]


def _char_class(ranges: list[tuple[int, int]]) -> str:
    """Build a regex character class like ``[一-鿿...]`` from codepoint pairs."""
    parts = [f"{chr(start)}-{chr(end)}" for start, end in ranges]
    return "[" + "".join(parts) + "]"


def _alternation(ranges: list[tuple[int, int]]) -> str:
    """Build an alternation like ``[一-鿿]|[㐀-䶿]`` usable inside larger patterns."""
    return "|".join(_char_class([r]) for r in ranges)


#: Character class matching any single CJK ideograph or kana character.
CJK_CHAR_RE: re.Pattern[str] = re.compile(_char_class(IDEOGRAPH_RANGES + KANA_RANGES))

#: Alternation of ideograph-only classes — FTS token patterns embed this.
IDEOGRAPH_ALTERNATION: str = _alternation(IDEOGRAPH_RANGES)

_KANA_ALTERNATION: str = _alternation(KANA_RANGES)

#: Alternation covering ideographs + kana, for findall-style tokenization.
CJK_TOKEN_ALTERNATION: str = f"{IDEOGRAPH_ALTERNATION}|{_KANA_ALTERNATION}"


def contains_cjk(text: str) -> bool:
    """Return True if *text* contains any CJK ideograph or kana character."""
    return CJK_CHAR_RE.search(text) is not None


def count_cjk(text: str) -> int:
    """Count CJK ideographs/kana characters in *text*."""
    return len(CJK_CHAR_RE.findall(text))
