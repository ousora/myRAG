"""Internal formatting helpers — paragraph splitting, single/chunked LLM formatting.

This module contains the internal implementation of text formatting:
- CJK-aware chunk threshold calculation
- Paragraph splitting with sentence-aware boundaries
- Single-shot and chunked LLM formatting

LLM transport (HTTP calls, JSON repair) lives in ``_llm.py``; public API is
re-exported from ``formatters/__init__.py``.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import TYPE_CHECKING, Any

from myrag.cjk import CJK_TOKEN_ALTERNATION

from ._llm import (
    _fix_bare_quotes_in_body_field,  # noqa: F401 — re-exported for backward compat
    _preprocess_json,  # noqa: F401 — re-exported for backward compat
    call_llm,
)
from .cache import format_cached
from .constants import CHUNKED_SCHEMA, FORMATTER_SCHEMA
from .prompts import (
    get_chunked_system_prompt,
    get_system_prompt,
    try_fix_common_issues,
    validate_format_output,
)
from .tags import extract_tags_from_body

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


def _get_config() -> Config:
    """Lazy-load config on first call."""
    from config import get_config
    return get_config()


# ── Chunking threshold ──────────────────────────────────────────────────
# Texts above this many characters trigger chunked processing.
# ~28K chars ≈ 7000 tokens — safe for most local LLMs (English, 4 chars/token).

# Default chars-per-token ratios by language family used to adjust the threshold.
_CJK_CHARS_PER_TOKEN = 1.0
_ENGLISH_CHARS_PER_TOKEN = 4.0

# Pre-compiled regex for paragraph splitting.
_PARAGRAPH_SPLIT = re.compile(r"\n\n+")

# Pre-compiled token counters for CJK-ratio estimation (C-speed, no per-char
# Python loop). Kana counts as CJK-like; ASCII alphanumerics count as English.
_CJK_TOKEN_RE = re.compile(CJK_TOKEN_ALTERNATION)
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]")


def _detect_cjk_ratio(text: str) -> float:
    """Estimate the proportion of CJK characters in text (0.0–1.0).

    Counts only actual CJK ranges (CJK Unified Ideographs, Hiragana, Katakana)
    as CJK; ASCII letters/digits as English-like; everything else (Cyrillic,
    Arabic, Devanagari, emoji, punctuation) is excluded from the ratio to avoid
    skewing documents that contain non-CJK multilingual content.
    Whitespace is also excluded. Counting runs through compiled regexes at
    C speed instead of a per-character Python loop.
    """
    if not text:
        return 0.0
    cjk = len(_CJK_TOKEN_RE.findall(text))
    latin = len(_LATIN_TOKEN_RE.findall(text))
    total = cjk + latin
    return cjk / total if total > 0 else 0.0


def _get_chunk_threshold() -> int:
    """Lazy-evaluate the chunk threshold from config on each call."""
    return _get_config().chunk_threshold_chars


def effective_chunk_threshold(text: str) -> int:
    """Return a CJK-aware chunk threshold adjusted for the input text's language mix.

    The base ``chunk_threshold_chars`` (default 20000) is calibrated for English
    at ~4 chars/token.  For predominantly CJK text (~1 char/token), we lower the
    threshold proportionally so that token budgets stay roughly constant across
    languages.
    """
    cfg = _get_config()
    base = cfg.chunk_threshold_chars  # ≈5000 tokens for English

    ratio = _detect_cjk_ratio(text)
    if ratio >= 0.5:
        # Mostly CJK — scale down so we don't exceed ~7000 tokens.
        return int(base * (_CJK_CHARS_PER_TOKEN / _ENGLISH_CHARS_PER_TOKEN))
    if ratio > 0.1:
        # Mixed — linear interpolation between the two extremes.
        mix = (ratio - 0.1) / 0.4  # 0 at 10% CJK, 1 at 50% CJK
        chars_per_token = _ENGLISH_CHARS_PER_TOKEN - mix * (
            _ENGLISH_CHARS_PER_TOKEN - _CJK_CHARS_PER_TOKEN
        )
        return int(base * (_CJK_CHARS_PER_TOKEN / chars_per_token))
    # Mostly English — use base threshold unchanged.
    return base


def _get_last_n_lines(md_parts: list[str], n: int = 10) -> str:
    """Extract the last N non-empty lines from accumulated markdown parts.

    Args:
        md_parts: Accumulated markdown parts from previous chunks.
        n: Number of trailing lines to extract.

    Returns:
        Empty string if no parts yet, otherwise the last N content lines.

    """
    if not md_parts:
        return ""
    full = "\n\n".join(md_parts)
    lines = [line for line in full.split("\n") if line.strip()]
    return "\n".join(lines[-n:])


def _split_by_paragraph(text: str, max_chars: int | None = None) -> list[str]:
    r"""Split text at paragraph boundaries, chunk oversized paragraphs at sentences.

    Normal paragraphs are grouped up to max_chars. If a single paragraph exceeds
    max_chars, it's split at sentence boundaries (`. `, `! `, `? `, or `\n`).

    Chunks do NOT physically overlap — continuity across chunks is provided
    via the prompt context (last 10 lines of previous output + summary).


    Args:
        text: The cleaned text to split.
        max_chars: Maximum characters per chunk (≈ tokens × 4). Defaults to config value.

    Returns:
        List of paragraph-boundary-aligned text chunks, each ≤ max_chars.

    """
    paragraphs = _PARAGRAPH_SPLIT.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    # Resolve default threshold from config (lazy, per-call)
    if max_chars is None:
        max_chars = _get_chunk_threshold()

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def _flush() -> None:
        """Flush accumulated paragraphs as a chunk."""
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for p in paragraphs:
        p_len = len(p) + 2  # +2 for \n\n separator

        # If this single paragraph already exceeds max_chars, split it inline
        if p_len > max_chars + 2:
            _flush()
            # Split at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", p)
            sent_buf: list[str] = []
            sent_len = 0
            for s in sentences:
                s_len = len(s) + 1
                if sent_len + s_len > max_chars and sent_buf:
                    chunks.append(" ".join(sent_buf))
                    sent_buf = []
                    sent_len = 0
                sent_buf.append(s)
                sent_len += s_len
            if sent_buf:
                chunks.append(" ".join(sent_buf))
            continue

        # Normal paragraph: accumulate until threshold
        if current_len + p_len > max_chars and current:
            _flush()

        current.append(p)
        current_len += p_len

    _flush()
    return chunks


def _format_text_single(raw: str, source_type: str = "web", *, system_prompt: str | None = None) -> dict[str, Any]:
    """Single-shot formatting — original behavior for small documents."""
    def _compute() -> dict[str, Any]:
        prompt = system_prompt if system_prompt is not None else get_system_prompt(source_type)
        result = call_llm(prompt, raw.strip(), schema=FORMATTER_SCHEMA)

        # Validate output against expected schema and fix common issues.
        errors = validate_format_output(result)
        if errors:
            logger.warning("format_text returned %d validation error(s): %s", len(errors), "; ".join(errors))
            result = try_fix_common_issues(result)

        # Fix placeholder metadata that the LLM copies from the prompt template.
        body = result.get("body", "")
        if isinstance(body, str):
            result.setdefault("metadata", {})["total_words"] = len(body.split())
        if "created_at" in result.get("metadata", {}):
            result["metadata"]["created_at"] = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        return result

    return format_cached(raw, source_type, system_prompt, _compute)


def _format_text_chunked_uncached(raw: str, source_type: str = "pdf", *, system_prompt: str | None = None) -> dict[str, Any]:
    """Chunked formatting for large documents (uncached worker).

    Splits text by paragraph, processes each chunk with LLM context
    (last 10 lines of previous output + cumulative summary), then
    merges results into a single structured output.

    Returns the same dict shape as _format_text_single() for pipeline compat.
    """
    chunks = _split_by_paragraph(raw)
    total = len(chunks)
    logger.info("Chunked processing: %d chunks, %d chars total", total, len(raw))

    all_parts: list[str] = []
    cumulative_summary = ""

    # Determine the system prompt once — custom prompts override defaults for all chunks.
    # Custom prompts override defaults for all chunks; title stays empty until merge.
    base_system_prompt = (
        system_prompt if system_prompt is not None else get_chunked_system_prompt(0, total)
    )

    for i, chunk_text in enumerate(chunks):
        # For non-first chunks with a document title available from metadata,
        # regenerate the prompt to include it. Otherwise use the base prompt.
        if system_prompt is None and i > 0:
            current_system_prompt = get_chunked_system_prompt(i, total)
        else:
            current_system_prompt = base_system_prompt

        prev_tail = _get_last_n_lines(all_parts, 10)
        prev_tail_block = (
            prev_tail
            or "This is the first chunk; no prior context needed."
        )
        summary_block = (
            cumulative_summary
            or "This is the first chunk."
        )

        user_message = (
            f"[Previous Context]\n"
            f"{prev_tail_block}\n\n"
            f"[Summary of Previous Chunks]\n"
            f"{summary_block}\n\n"
            f"[Current Chunk Text]\n"
            f"{chunk_text}"
        )

        logger.info("Chunk %d/%d: %d chars input — calling LLM...",
                     i + 1, total, len(chunk_text))
        cfg = _get_config()
        result = call_llm(
            current_system_prompt,
            user_message,
            max_tokens=cfg.chunk_max_tokens,
            timeout=cfg.chunk_timeout,
            schema=CHUNKED_SCHEMA,
        )

        part_md = result.get("part_md", "").strip()
        summary = result.get("summary", "").strip()
        logger.info("Chunk %d/%d: %d chars output, summary='%s'",
                     i + 1, total, len(part_md), summary[:80] if summary else "(empty)")

        if part_md:
            all_parts.append(part_md)

        chunk_summary = result.get("summary", "").strip()
        if chunk_summary:
            cumulative_summary = f"{cumulative_summary}{chunk_summary} "

    # Merge all parts into the final body
    body = "\n\n".join(all_parts)

    # Extract title from the first `# Title` in body
    title = "Untitled Document"
    title_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    # Post-process: strip duplicate top-level headings matching the document title
    if title and title != "Untitled Document":
        lines = body.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^#\s+", stripped) and stripped.startswith(f"# {title}"):
                continue  # skip duplicate title heading
            cleaned_lines.append(line)
        body = "\n".join(cleaned_lines)

    # Extract sections from ## and ### headers in body (after dedup)
    sections: list[dict] = []
    for match in re.finditer(r"^(#{2,3})\s+(.+)$", body, re.MULTILINE):
        level = len(match.group(1))
        section_title = match.group(2).strip()
        sections.append({"level": level, "title": section_title})

    logger.info("Chunked merge complete: %d parts → %d chars, %d sections",
                len(all_parts), len(body), len(sections))

    # Generate tags from body content (chunked mode doesn't get LLM-generated tags)
    tags = extract_tags_from_body(body, title)
    result = {
        "title": title,
        "tags": tags,
        "metadata": {
            "source_type": source_type,
            "total_words": len(body.split()),
            "sections": sections,
            "chunks_processed": total,
        },
        "body": body,
    }

    # Validate and fix common issues in the merged output.
    errors = validate_format_output(result)
    if errors:
        logger.warning("chunked format returned %d validation error(s): %s", len(errors), "; ".join(errors))
        result = try_fix_common_issues(result)

    return result


def _format_text_chunked(raw: str, source_type: str = "pdf", *, system_prompt: str | None = None) -> dict[str, Any]:
    """Chunked formatting for large documents, with process-wide caching.

    Delegates to ``_format_text_chunked_uncached`` and caches the merged result
    so re-ingesting identical text skips the (expensive) multi-call LLM pass.
    """
    def _compute() -> dict[str, Any]:
        return _format_text_chunked_uncached(raw, source_type, system_prompt=system_prompt)

    return format_cached(raw, source_type, system_prompt, _compute)


def _format_text_async_impl(
    raw: str,
    source_type: str,
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    if not raw.strip():
        msg = "Input text is empty"
        raise ValueError(msg)
    raw_len = len(raw)
    threshold = effective_chunk_threshold(raw)
    if raw_len > threshold:
        return _format_text_chunked(raw, source_type, system_prompt=system_prompt)
    return _format_text_single(raw, source_type, system_prompt=system_prompt)
