"""Deterministic markdown normalizer — repairs and structures plain parser output.

Used by the non-LLM path (``use_llm=False``) to turn MarkItDown/Trafilatura's
raw text into structured markdown without any model call. All transformations
are regex/line-based and deterministic.

Transformations (in order):
    1. Heading promotion — standalone short lines → ``##`` headings
    2. List normalization — ``1)``, ``a)``, ``•``, ``*`` → standardized markers
    3. Link auto-formatting — bare URLs → ``[url](url)``
    4. Bold/italic repair — close unclosed ``**``/``*`` markers
    5. Table column alignment — normalize pipe spacing in grid tables
"""

from __future__ import annotations

import re

# ── Pre-compiled patterns ──────────────────────────────────────────────

# Bare URLs (not already inside a markdown link)
_URL_RE = re.compile(
    r"(?<![\]\)])(https?://[^\s)\]]+)",
    re.IGNORECASE,
)

# ATX heading already present
_ATX_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# List markers: 1), 1., a), a., •, *, -
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:"
    r"(?:\d+)[\)\.]|"       # 1) or 1.
    r"(?:[a-zA-Z])[\)\.]|"  # a) or a.
    r"[•·\*\-+]"            # bullet chars
    r")\s+",
)

# Table row: starts and ends with |
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def normalize_markdown(text: str) -> str:
    """Apply all deterministic normalizations to *text*.

    Returns structured markdown with promoted headings, normalized lists,
    formatted links, repaired bold/italic, and aligned tables.
    """
    if not text or not text.strip():
        return ""

    lines = text.split("\n")

    # Phase 1: Heading promotion (skip lines already under ATX headings)
    lines = _promote_headings(lines)

    # Phase 2: List normalization
    lines = _normalize_lists(lines)

    # Phase 3: Link formatting (per-line, avoid touching code blocks)
    lines = [_format_links(line) for line in lines]

    # Phase 4: Bold/italic repair (whole text)
    result = "\n".join(lines)
    result = _repair_bold_italic(result)

    # Phase 5: Table alignment
    return _align_tables(result)


# ── Heading promotion ──────────────────────────────────────────────────


def _promote_headings(lines: list[str]) -> list[str]:
    """Promote standalone short lines that look like section titles to ``##``.

    A candidate line:
      - Is not already an ATX heading, blockquote, list item, or table row
      - Has no sentence-ending punctuation (.!?:;)
      - Is ≤ 80 chars
      - Is not empty
      - Is surrounded by blank lines (or at document start/end)
    """
    result: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines, existing headings, quotes, lists, tables
        if not stripped:
            result.append(line)
            continue
        if _ATX_HEADING_RE.match(stripped):
            result.append(line)
            continue
        if stripped.startswith(">"):
            result.append(line)
            continue
        if _LIST_ITEM_RE.match(stripped):
            result.append(line)
            continue
        if _TABLE_ROW_RE.match(stripped):
            result.append(line)
            continue

        # Must be a plausible title: short, no sentence punctuation
        if len(stripped) > 80:
            result.append(line)
            continue
        if re.search(r"[.!?;:]", stripped):
            result.append(line)
            continue
        # Must look like a title, not a code fragment or URL
        if stripped.startswith(("http://", "https://", "```", "    ")):
            result.append(line)
            continue

        # Surrounded by blank lines (or at boundary)?
        # ponytail: only require the PREVIOUS line to be blank (or line 0).
        # Requiring both sides blank misses the common parser pattern where a
        # heading is immediately followed by content on the next line.
        prev_blank = (i == 0) or not lines[i - 1].strip()
        if prev_blank:
            result.append(f"## {stripped}")
        else:
            result.append(line)

    return result


# ── List normalization ─────────────────────────────────────────────────


def _normalize_lists(lines: list[str]) -> list[str]:
    """Normalize list markers to a consistent form.

    - ``1)`` / ``1.`` → ``1.``
    - ``a)`` / ``a.`` → ``a)``
    - ``•`` / ``·`` → ``-``
    - ``*`` / ``+`` → ``-`` (only at line start, not inside emphasis)
    """
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        m = _LIST_ITEM_RE.match(stripped)
        if not m:
            result.append(line)
            continue

        # Determine indentation
        indent = line[: len(line) - len(line.lstrip())]
        content = stripped[m.end():]

        # Normalize the marker
        marker = m.group(0).strip()
        num_match = re.match(r"^(\d+)[\).]$", marker)
        if num_match:
            # Numbered: keep number, use dot
            new_marker = f"{num_match.group(1)}."
        else:
            letter_match = re.match(r"^([a-zA-Z])[\).]$", marker)
            if letter_match:
                # Letter: keep letter, use paren
                new_marker = f"{letter_match.group(1)})"
            elif marker in ("•", "·", "*", "+"):
                new_marker = "-"
            else:
                new_marker = marker

        result.append(f"{indent}{new_marker} {content}")

    return result


# ── Link formatting ────────────────────────────────────────────────────


def _format_links(line: str) -> str:
    """Replace bare URLs with markdown links ``[url](url)``.

    Skips lines inside code spans (single backtick pairs).
    """
    if "```" in line:
        return line  # skip fenced code blocks entirely
    if line.count("`") >= 2:
        # Inline code — protect the span
        parts = line.split("`")
        # Only format even-indexed parts (outside code spans)
        for i in range(0, len(parts), 2):
            if i < len(parts):
                parts[i] = _URL_RE.sub(r"[\1](\1)", parts[i])
        return "`".join(parts)

    return _URL_RE.sub(r"[\1](\1)", line)


# ── Bold / italic repair ───────────────────────────────────────────────


def _repair_bold_italic(text: str) -> str:
    """Close unclosed bold (``**``) and italic (``*``) markers.

    Strategy: count marker pairs across the whole text. If odd, append a
    closing marker at the end. This is intentionally simple — the non-LLM
    path is a fallback, not a renderer.
    """
    # Bold: count ** pairs (not inside code blocks)
    # Simple approach: strip code spans first
    protected: list[tuple[int, int]] = []
    for m in re.finditer(r"```[\s\S]*?```", text):
        protected.append((m.start(), m.end()))

    def _is_protected(pos: int) -> bool:
        return any(s <= pos < e for s, e in protected)

    # Count ** outside code
    bold_count = 0
    i = 0
    while i < len(text) - 1:
        if text[i] == "*" and text[i + 1] == "*" and not _is_protected(i):
            bold_count += 1
            i += 2
        else:
            i += 1

    if bold_count % 2 == 1:
        text = text + "\n\n**"

    # Italic: count single * outside code and not adjacent to another *
    italic_count = 0
    i = 0
    while i < len(text):
        if text[i] == "*" and not _is_protected(i):
            # Not part of ** (check neighbors)
            prev_star = i > 0 and text[i - 1] == "*"
            next_star = i < len(text) - 1 and text[i + 1] == "*"
            if not prev_star and not next_star:
                italic_count += 1
        i += 1

    if italic_count % 2 == 1:
        text = text + "\n\n*"

    return text


# ── Table alignment ────────────────────────────────────────────────────


def _align_tables(text: str) -> str:
    """Normalize pipe spacing in markdown tables.

    Ensures each cell has exactly one space padding: ``| a | b |`` not
    ``|a|b|``. Does not change column count.
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not _TABLE_ROW_RE.match(line):
            result.append(line)
            i += 1
            continue

        # Collect consecutive table rows
        table_rows: list[str] = []
        while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
            table_rows.append(lines[i].strip())
            i += 1

        # Normalize each row: | cell | cell |
        normalized: list[str] = []
        for row in table_rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            normalized.append("| " + " | ".join(cells) + " |")

        result.extend(normalized)

    return "\n".join(result)


__all__ = ["normalize_markdown"]
