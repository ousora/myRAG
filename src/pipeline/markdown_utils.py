"""Markdown rendering and entity matching helpers for the RAG pipeline.

This module contains utilities for:
- Rendering LLM formatter output into structured markdown with headers
- Stripping reference/bibliography sections before chunking
- Matching document-level entities to individual chunks
"""

from __future__ import annotations

import re
from typing import Any

from myrag.cjk import contains_cjk

_FRONTMATTER_FENCE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(md: str) -> dict[str, Any]:
    """Extract YAML front matter fields from a markdown document.

    Returns a dict of the parsed fields (possibly empty) when the document
    starts with a ``---`` fenced block; malformed YAML yields ``{}`` rather
    than raising, since front matter is advisory metadata.
    """
    m = _FRONTMATTER_FENCE.match(md)
    if not m:
        return {}
    import yaml
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}

# Reference-section keywords. The keyword must anchor the START of the
# heading; anything after it may only be decoration (see _TAIL_DECORATION_RE).
# English keywords are word-boundary anchored; CJK keywords use prefix match.
_REFERENCE_HEAD_RE = re.compile(
    r"(?:references?|reference list|bibliography|further reading|"
    r"works cited|sources?|footnotes?|endnotes?|citations?|"
    r"see also|external links)\b",
    re.IGNORECASE,
)
_REFERENCE_CJK_HEADS = (
    "参考文献", "参考资料", "參考文獻", "參考資料",
    "引用文献", "引用文獻", "延伸閱讀", "延伸阅读",
    "脚注", "注脚", "註腳", "文獻", "注释", "註釋",
    "参考", "參考", "引用", "外部链接", "外部連結",
)

# Decoration that may follow a reference keyword without disqualifying the
# heading: whitespace, digits, CJK numerals (for numbering like 参考文献（三）),
# and punctuation (brackets, colons, CJK marks…). Ordinary words after the
# keyword mean it is a real section title, e.g. "Reference Architecture" or
# "系统引用说明" — those must NOT be stripped.
_TAIL_DECORATION_RE = re.compile(
    r"""[\s\d一二三四五六七八九十百千零〇两.,;:!?)('"“”'’\-—–·。、；：！？（）【】《》\[\]]*"""
)
# Leading numbering/decoration stripped before matching ("3. References",
# "三、参考文献").
_LEADING_DECORATION_RE = re.compile(r"^[\s(（\[【]*[\d一二三四五六七八九十]+[.、.)）】\]]*\s*")


def render_markdown(result: dict[str, Any]) -> str:
    """Render a clean markdown document from the LLM formatter output.

    The formatter's ``body`` is non-deterministic: it usually already contains
    its own ``##``/``###`` headings. When it does, we keep that structure and
    only drop a duplicate top-level title H1 (which we render ourselves). When
    it does not, we prepend the title and leave the body as-is — the chunker
    then falls back to plain-text splitting, which is preferable to fabricating
    header scaffolding that mis-nests the content.
    """
    title = result.get("title", "Untitled")
    body = result.get("body", "") or ""

    if re.search(r"^#{1,6}\s+", body):
        body_lines = body.split("\n")
        kept = []
        title_seen = False
        for line in body_lines:
            stripped = line.strip()
            if not title_seen and re.match(rf"^#\s+{re.escape(title)}$", stripped, re.IGNORECASE):
                title_seen = True
                continue  # drop duplicate title H1; we render our own below
            kept.append(line)
        body_block = "\n".join(kept).strip()
        return f"# {title}\n\n{body_block}\n"

    # No headings in body — prepend title only; let the chunker handle structure.
    return f"# {title}\n\n{body.strip()}\n"


#: Backward-compatibility alias. The function no longer renders headers from
#: ``metadata.sections`` (the body's own headings are kept instead), so the
#: old "with_sections" name was misleading.
render_markdown_with_sections = render_markdown


def strip_reference_sections(md: str) -> str:
    """Remove reference / bibliography sections from markdown.

    Reference lists (References, 参考文献, 參考, Bibliography, Further reading,
    …) are high in keyword overlap but carry no answer-worthy content, so they
    pollute retrieval. They are stripped *before chunking/embedding*; the human
    `.md` output is left untouched.

    A section is removed from its heading line up to (but not including) the next
    heading of the same or higher level, or end-of-file.
    """
    lines = md.split("\n")

    headings: list[tuple[int, int, str]] = []  # (line_index, level, title)
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    ref_lines = {i for i, _lvl, title in headings if is_reference_title(title)}
    if not ref_lines:
        return md

    delete: set[int] = set()
    for i, level, _title in headings:
        if i not in ref_lines:
            continue
        end = len(lines)
        for j, lvl, _ in headings:
            if j > i and lvl <= level:
                end = j
                break
        delete.update(range(i, end))

    return "\n".join(lines[k] for k in range(len(lines)) if k not in delete)


def is_reference_title(title: str) -> bool:
    """Return True if a section heading looks like a references/bibliography block.

    The reference keyword must anchor the *start* of the heading; trailing text
    may only be digits/punctuation decoration ("References (2024)",
    "参考文献（三）", "參考"). A keyword followed by ordinary words is a
    legitimate section title ("Reference Architecture", "系统引用说明",
    "引用格式說明") and is kept — substring matching here used to silently
    delete content sections.
    """
    t = _LEADING_DECORATION_RE.sub("", title.strip(), count=1)

    m = _REFERENCE_HEAD_RE.match(t)
    if m and _TAIL_DECORATION_RE.fullmatch(t[m.end():]):
        return True
    return any(
        t.startswith(kw) and _TAIL_DECORATION_RE.fullmatch(t[len(kw):])
        for kw in _REFERENCE_CJK_HEADS
    )


def match_entities_to_chunks(chunks: list[dict[str, Any]], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match document-level entities to individual chunks by text presence.

    Scans each chunk's text for entity names (case-insensitive).
    Only entities that actually appear in a chunk get tagged on that chunk.
    This keeps entity search granular — querying 'GPT-4' returns only chunks
    that mention GPT-4, not every chunk from the same document.

    Args:
        chunks: List of chunk dicts, each with at least a 'text' key.
        entities: List of entity dicts with 'name' keys from formatter output.

    Returns:
        Same chunks list with 'entity_names' added to each chunk.

    """
    if not entities:
        return chunks

    # Pre-classify entities: CJK names have no word boundaries, so \b never
    # matches them. For those we use a plain substring test; for Latin names we
    # keep the case-insensitive word-boundary match to avoid partial matches.
    cjk_entities = [e["name"] for e in entities if contains_cjk(e["name"])]
    latin_entities = [e["name"] for e in entities if not contains_cjk(e["name"])]
    latin_patterns = [
        (name, re.compile(r"\b" + re.escape(name.lower()) + r"\b")) for name in latin_entities
    ]

    for chunk in chunks:
        chunk_text_lower = chunk["text"].lower()
        matched = [name for name in cjk_entities if name.lower() in chunk_text_lower]
        matched += [name for name, pat in latin_patterns if pat.search(chunk_text_lower)]
        chunk["entity_names"] = matched
    return chunks
