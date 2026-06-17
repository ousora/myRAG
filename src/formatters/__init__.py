"""Text formatter — structured output from raw copied text.

Handles both single-shot and chunked (large document) modes.
Auto-detects which path to use based on input size.
"""

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict

import httpx

from .prompts import get_system_prompt, get_chunked_system_prompt

logger = logging.getLogger(__name__)

# ── Internal helpers ────────────────────────────────────────────────────


def _get_config():
    """Lazy-load config on first call."""
    from config import get_config
    return get_config()


# ── Chunking threshold ──────────────────────────────────────────────────
# Texts above this many characters trigger chunked processing.
# ~28K chars ≈ 7000 tokens — safe for most local LLMs.
_CHUNK_THRESHOLD_CHARS = _get_config().chunk_threshold_chars

_executor = None


def get_executor() -> ThreadPoolExecutor:
    """Lazy-initialize the shared thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2)
    return _executor


def call_llm(system_prompt: str, user_message: str, *,
             max_tokens: int | None = None,
             timeout: int | None = None) -> dict:
    """Make a single LLM API call and return the parsed JSON response.

    Args:
        system_prompt: System-level instruction for the LLM.
        user_message: The user input text.
        max_tokens: Override max output tokens. Uses config default if None.
        timeout: Override HTTP timeout in seconds. Uses config default if None.

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
                "max_tokens": max_tokens or cfg.llm_max_tokens,
            },
            timeout=timeout or cfg.llm_timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("LLM call failed after %.1fs: %s",
                      (timeout or cfg.llm_timeout), e)
        raise RuntimeError(f"LLM API request failed: {e}") from e

    try:
        raw_content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logger.error("LLM returned unexpected response structure: %s", e)
        raise ValueError(f"LLM returned invalid format: {e}") from e

    input_chars = len(user_message)
    output_chars = len(raw_content)
    logger.info("LLM call: %d chars in → %d chars out (max_tokens=%s, timeout=%ss)",
                input_chars, output_chars,
                max_tokens or cfg.llm_max_tokens,
                timeout or cfg.llm_timeout)

    # Save raw response for debugging
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use a simple hash of the input to create a unique filename
    input_hash = hashlib.md5(user_message.encode()).hexdigest()[:8]
    output_path = f"test/llmoutput/resp_{timestamp}_{input_hash}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(raw_content)
    logger.info("Saved raw response to %s", output_path)

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
    except json.JSONDecodeError:
        # The LLM is producing very large nested strings that break standard json.loads.
        # We attempt to manually extract the "part_md" field if it exists.
        try:
            # Try to find the part_md key and its value
            match = re.search(r'"part_md":\s*"(.*?)"', content, re.DOTALL)
            if match:
                # This is a simplified fallback; in a production system, 
                # we'd use a proper streaming parser or a more robust regex.
                return {
                    "part_md": match.group(1).replace('\\n', '\n').replace('\\"', '"'),
                    "summary": "Extracted via fallback"
                }
            
            # If that fails, try to fix common escape issues and try one last time
            cleaned_content = content.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            return json.loads(cleaned_content)
        except Exception:
            raise ValueError(
                f"LLM returned invalid JSON after multiple fallback attempts. Raw response: {content!r}"
            ) from None


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
    """Split text at paragraph boundaries, chunk oversized paragraphs at sentences.

    Normal paragraphs are grouped up to max_chars. If a single paragraph exceeds
    max_chars, it's split at sentence boundaries (`. `, `! `, `? `, or `\n`).

    Chunks do NOT physically overlap — continuity across chunks is provided
    via the prompt context (last 10 lines of previous output + summary).

    Args:
        text: The cleaned text to split.
        max_chars: Maximum characters per chunk (≈ tokens × 4).

    Returns:
        List of paragraph-boundary-aligned text chunks, each ≤ max_chars.
    """
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def _flush():
        """Flush accumulated paragraphs as a chunk."""
        nonlocal current, current_len
        if current:
            chunks.append('\n\n'.join(current))
            current = []
            current_len = 0

    for p in paragraphs:
        p_len = len(p) + 2  # +2 for \n\n separator

        # If this single paragraph already exceeds max_chars, split it inline
        if p_len > max_chars + 2:
            _flush()
            # Split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', p)
            sent_buf: list[str] = []
            sent_len = 0
            for s in sentences:
                s_len = len(s) + 1
                if sent_len + s_len > max_chars and sent_buf:
                    chunks.append(' '.join(sent_buf))
                    sent_buf = []
                    sent_len = 0
                sent_buf.append(s)
                sent_len += s_len
            if sent_buf:
                chunks.append(' '.join(sent_buf))
            continue

        # Normal paragraph: accumulate until threshold
        if current_len + p_len > max_chars and current:
            _flush()

        current.append(p)
        current_len += p_len

    _flush()
    return chunks


def _extract_tags_from_body(body: str, title: str) -> list[str]:
    """Generate tags from body content for chunked processing mode.

    Uses keyword frequency analysis on the merged body text to extract
    meaningful domain-specific terms as tags. Falls back to simple
    noun-phrase extraction if no strong keywords are found.
    """
    import re
    from collections import Counter

    # Common English stop words to filter out
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    }

    # Extract meaningful words from body (alphanumeric sequences >= 3 chars)
    words = re.findall(r'[a-zA-Z]{3,}', body.lower())
    word_freq = Counter(w for w in words if w not in STOP_WORDS and len(w) > 2)

    # Also extract title keywords
    title_words = re.findall(r'[a-zA-Z]{3,}', title.lower())
    title_freq = Counter(title_words)

    # Combine: title words get higher weight
    combined = Counter(word_freq)
    for w, c in title_freq.items():
        combined[w] += c * 2

    # Filter out very common technical terms that aren't useful as tags
    OVERLY_COMMON = {
        'system', 'payment', 'china', 'country', 'bank', 'data',
        'information', 'process', 'service', 'user', 'network',
        'document', 'file', 'text', 'content', 'example',
    }

    # Select top tags (5-8), preferring multi-word phrases and domain-specific terms
    candidates = []
    for word, count in combined.most_common(30):
        if word not in OVERLY_COMMON:
            candidates.append(word)

    # If we have enough single words, use them; otherwise fall back to title-based tags
    if len(candidates) >= 5:
        return [c for c in candidates[:8]]

    # Fallback: extract from title and first few paragraphs
    fallback_words = re.findall(r'[a-zA-Z]{3,}', (title + ' ' + body[:2000]).lower())
    fb_freq = Counter(w for w in fallback_words if w not in STOP_WORDS)
    return [w for w, _ in fb_freq.most_common(8)]


def _format_text_single(raw: str, source_type: str = "web", *, system_prompt: str | None = None) -> Dict[str, Any]:
    """Single-shot formatting — original behavior for small documents."""
    prompt = system_prompt if system_prompt is not None else get_system_prompt(source_type)
    result = call_llm(prompt, raw.strip())

    # Fix placeholder metadata that the LLM copies from the prompt template.
    body = result.get("body", "")
    if isinstance(body, str):
        result.setdefault("metadata", {})["total_words"] = len(body.split())
    if "created_at" in result.get("metadata", {}):
        import datetime
        result["metadata"]["created_at"] = datetime.datetime.now().isoformat()

    return result


def _format_text_chunked(raw: str, source_type: str = "pdf") -> Dict[str, Any]:
    """Chunked formatting for large documents.

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

    for i, chunk_text in enumerate(chunks):
        system_prompt = get_chunked_system_prompt(i, total)

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
            system_prompt,
            user_message,
            max_tokens=cfg.chunk_max_tokens,
            timeout=cfg.chunk_timeout,
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
    title_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    # Post-process: strip duplicate top-level headings matching the document title
    if title and title != "Untitled Document":
        lines = body.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if re.match(r'^#\s+', stripped) and stripped.startswith(f'# {title}'):
                continue  # skip duplicate title heading
            cleaned_lines.append(line)
        body = '\n'.join(cleaned_lines)

    # Extract sections from ## and ### headers in body (after dedup)
    if title_match:
        title = title_match.group(1).strip()

    # Extract sections from ## and ### headers in body
    sections: list[dict] = []
    for match in re.finditer(r'^(#{2,3})\s+(.+)$', body, re.MULTILINE):
        level = len(match.group(1))
        section_title = match.group(2).strip()
        sections.append({"level": level, "title": section_title})

    logger.info("Chunked merge complete: %d parts → %d chars, %d sections",
                len(all_parts), len(body), len(sections))

    # Generate tags from body content (chunked mode doesn't get LLM-generated tags)
    tags = _extract_tags_from_body(body, title)

    return {
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
    raw_len = len(raw)
    if raw_len > _CHUNK_THRESHOLD_CHARS:
        logger.info("Large text: %d chars — calling chunked processor", raw_len)
        return _format_text_chunked(raw, source_type)

    logger.info("Small text: %d chars — single-shot", raw_len)
    return _format_text_single(raw, source_type)


def _format_text_async_impl(raw: str, source_type: str, *, system_prompt: str | None = None) -> Dict[str, Any]:
    """Internal implementation of async formatting that respects custom system prompts."""
    if not raw.strip():
        raise ValueError("Input text is empty")

    raw_len = len(raw)
    if raw_len > _CHUNK_THRESHOLD_CHARS:
        return _format_text_chunked(raw, source_type)

    return _format_text_single(raw, source_type, system_prompt=system_prompt)


def format_text_async(raw: str, source_type: str = "web", *, system_prompt: str | None = None) -> Future[Dict[str, Any]]:
    """Submit formatting task to thread pool. Returns a Future.

    Args:
        raw: The text to process.
        source_type: Source context for the LLM ('web', 'markdown', 'pdf_clip').
        system_prompt: Optional custom system prompt (overrides default).
                       Useful for RAG queries where you want a different prompt style.
    """
    future = get_executor().submit(_format_text_async_impl, raw, source_type, system_prompt=system_prompt)
    return future


def format_text_with_system(raw: str, source_type: str = "web", *, system_prompt: str | None = None) -> Dict[str, Any]:
    """Format text with an optional custom system prompt.

    Convenience wrapper that delegates to _format_text_single() or _format_text_chunked()
    depending on input size, passing the system_prompt through.

    Args:
        raw: The text to process.
        source_type: Source context for the LLM ('web', 'markdown', 'pdf_clip').
        system_prompt: Optional custom system prompt (overrides default).

    Returns:
        Dict with keys: title, tags, metadata, body.
    """
    if not raw.strip():
        raise ValueError("Input text is empty")

    raw_len = len(raw)
    if raw_len > _CHUNK_THRESHOLD_CHARS:
        return _format_text_chunked(raw, source_type)

    return _format_text_single(raw, source_type, system_prompt=system_prompt)


__all__ = ["call_llm", "format_text", "format_text_async", "format_text_with_system"]

# Re-export writer functions for convenience
from .writer import format_md, write_to_md  # noqa: F401, E402
