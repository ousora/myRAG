"""Text formatter — structured output from raw copied text.

Handles both single-shot and chunked (large document) modes.
Auto-detects which path to use based on input size.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict

import httpx

from .prompts import get_system_prompt, get_chunked_system_prompt

# ── Chunking threshold ──────────────────────────────────────────────────
# Texts above this many characters trigger chunked processing.
# ~28K chars ≈ 7000 tokens — safe for most local LLMs.
_CHUNK_THRESHOLD_CHARS = 28000


# ── Internal helpers ────────────────────────────────────────────────────


def _get_config():
    """Lazy-load config on first call."""
    from myrag.config import get_config
    return get_config()


_executor = None


def get_executor() -> ThreadPoolExecutor:
    """Lazy-initialize the shared thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2)
    return _executor


def _call_llm(system_prompt: str, user_message: str) -> dict:
    """Make a single LLM API call and return the parsed JSON response.

    Args:
        system_prompt: System-level instruction for the LLM.
        user_message: The user input text.

    Returns:
        Parsed JSON dict from the LLM response.

    Raises:
        RuntimeError: On HTTP/network errors.
        ValueError: On invalid JSON or unexpected response format.
    """
    cfg = _get_config()

    try:
        response = httpx.post(
            cfg.llm_endpoint,
            json={
                "model": cfg.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": cfg.llm_temperature,
                "max_tokens": cfg.llm_max_tokens,
            },
            timeout=cfg.llm_timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"LLM API request failed: {e}") from e

    try:
        raw_content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"LLM returned invalid format: {e}") from e

    # Strip code block markers from LLM response
    content = re.sub(
        r'^```(?:json)?\s*\n', '', raw_content
    ).strip() if isinstance(raw_content, str) else raw_content

    # Extract the first valid JSON object
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        content = json_match.group(0)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON: {e}. Raw response: {content!r}"
        ) from e


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


def _split_by_paragraph(text: str, max_chars: int = _CHUNK_THRESHOLD_CHARS) -> list[str]:
    """Split text at double-newline paragraph boundaries.

    Chunks do NOT overlap — continuity is provided via the prompt context
    (last 10 lines of previous markdown + cumulative summary).
    Each chunk ≤ max_chars to stay within the LLM's reliable context window.

    Args:
        text: The cleaned text to split.
        max_chars: Maximum characters per chunk (≈ tokens × 4).

    Returns:
        List of paragraph-boundary-aligned text chunks.
    """
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p) + 2  # +2 for the \n\n separator
        if current_len + p_len > max_chars and current:
            chunks.append('\n\n'.join(current))
            current = []
            current_len = 0
        current.append(p)
        current_len += p_len

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def _format_text_single(raw: str, source_type: str = "web") -> Dict[str, Any]:
    """Single-shot formatting — original behavior for small documents."""
    prompt = get_system_prompt(source_type)
    return _call_llm(prompt, raw.strip())


def _format_text_chunked(raw: str, source_type: str = "pdf") -> Dict[str, Any]:
    """Chunked formatting for large documents.

    Splits text by paragraph, processes each chunk with LLM context
    (last 10 lines of previous output + cumulative summary), then
    merges results into a single structured output.

    Returns the same dict shape as _format_text_single() for pipeline compat.
    """
    chunks = _split_by_paragraph(raw)
    total = len(chunks)

    all_parts: list[str] = []
    cumulative_summary = ""

    for i, chunk_text in enumerate(chunks):
        system_prompt = get_chunked_system_prompt(i, total)

        prev_tail = _get_last_n_lines(all_parts, 10)
        prev_tail_block = (
            prev_tail
            or "（这是文档的第一部分，无需参考前文。）"
        )
        summary_block = (
            cumulative_summary
            or "（这是文档的第一部分。）"
        )

        user_message = (
            f"【前文收尾】\n"
            f"{prev_tail_block}\n\n"
            f"【前文摘要】\n"
            f"{summary_block}\n\n"
            f"【本段原文】\n"
            f"{chunk_text}"
        )

        result = _call_llm(system_prompt, user_message)

        part_md = result.get("part_md", "").strip()
        if part_md:
            all_parts.append(part_md)

        chunk_summary = result.get("summary", "").strip()
        if chunk_summary:
            cumulative_summary = f"{cumulative_summary}{chunk_summary} "

    # Merge all parts into the final body
    body = "\n\n".join(all_parts)

    # Extract title from the first `# Title` in body
    title = "Untitled Document"
    title_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    # Extract sections from ## and ### headers in body
    sections: list[dict] = []
    for match in re.finditer(r'^(#{2,3})\s+(.+)$', body, re.MULTILINE):
        level = len(match.group(1))
        section_title = match.group(2).strip()
        sections.append({"level": level, "title": section_title})

    return {
        "title": title,
        "tags": [],
        "metadata": {
            "source_type": source_type,
            "total_words": len(body.split()),
            "sections": sections,
            "chunks_processed": total,
        },
        "body": body,
    }


# ── Public API ──────────────────────────────────────────────────────────


def format_text(raw: str, source_type: str = "web") -> Dict[str, Any]:
    """Format raw extracted text into structured knowledge content.

    Auto-detects the best processing mode:
    - Small text (< ~28K chars): Single LLM call (original behavior).
    - Large text: Split by paragraph and process chunk-by-chunk with
      context (last 10 lines + summary) for continuity.

    Args:
        raw: The raw text extracted from a document.
        source_type: Source context for the LLM ('web', 'markdown', 'pdf_clip').

    Returns:
        Dict with keys: title, tags, metadata, body.

    Raises:
        httpx.HTTPError: If the API request fails.
        ValueError: If the LLM returns invalid JSON or unexpected response format.
    """
    if not raw.strip():
        raise ValueError("Input text is empty")

    # Auto-dispatch based on text length
    if len(raw) > _CHUNK_THRESHOLD_CHARS:
        return _format_text_chunked(raw, source_type)

    return _format_text_single(raw, source_type)


def format_text_async(raw: str, source_type: str = "web") -> Future[Dict[str, Any]]:
    """Submit formatting task to thread pool. Returns a Future."""
    future = get_executor().submit(format_text, raw, source_type)
    return future


__all__ = ["format_text", "format_text_async"]

# Re-export writer functions for convenience
from .writer import format_md, write_to_md  # noqa: F401, E402
