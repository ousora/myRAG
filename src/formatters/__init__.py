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
from .constants import FORMATTER_SCHEMA, CHUNKED_SCHEMA

logger = logging.getLogger(__name__)

# ── Internal helpers ────────────────────────────────────────────────────


def _get_config():
    """Lazy-load config on first call."""
    from config import get_config
    return get_config()


# ── Chunking threshold ──────────────────────────────────────────────────
# Texts above this many characters trigger chunked processing.
# ~28K chars ≈ 7000 tokens — safe for most local LLMs.


def _get_chunk_threshold() -> int:
    """Lazy-evaluate the chunk threshold from config on each call."""
    return _get_config().chunk_threshold_chars

_executor = None


def get_executor() -> ThreadPoolExecutor:
    """Lazy-initialize the shared thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2)
    return _executor


def call_llm(system_prompt: str, user_message: str, *,
             max_tokens: int | None = None,
             timeout: int | None = None,
             schema: dict | None = None) -> dict:
    """Make a single LLM API call and return the parsed JSON response.

    Args:
        system_prompt: System message content.
        user_message: User message content.
        max_tokens: Token limit for generation (defaults to config).
        timeout: Request timeout in seconds (defaults to config).
        schema: Optional JSON Schema dict sent as ``response_format``.
                When provided, llama.cpp / OpenAI servers enforce output structure.
    """
    cfg = _get_config()

    payload: dict[str, Any] = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": cfg.llm_temperature,
        "max_tokens": max_tokens or cfg.llm_max_tokens,
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_object",
            "schema": schema,
        }

    try:
        response = httpx.post(cfg.llm_endpoint, json=payload, timeout=timeout or cfg.llm_timeout)
        response.raise_for_status()
    except httpx.HTTPError as e:
        # Some llama.cpp backends fail on JSON Schema enforcement
        # (peg-grammar incompatibility). Retry without schema if this happens.
        resp_for_retry = getattr(e, "response", None)
        schema_fallback_codes = {500, 503, 429}
        if schema is not None and resp_for_retry is not None and resp_for_retry.status_code in schema_fallback_codes:
            err_body = resp_for_retry.text
            if "peg" in err_body.lower() or "format" in err_body.lower():
                logger.warning("Schema-based response_format rejected by server (HTTP %d), retrying without schema", resp_for_retry.status_code)
                payload.pop("response_format", None)
                try:
                    response = httpx.post(cfg.llm_endpoint, json=payload, timeout=timeout or cfg.llm_timeout)
                    response.raise_for_status()
                except httpx.HTTPError as e2:
                    logger.error("LLM call failed (no schema fallback): %s", e2)
                    raise RuntimeError(f"LLM API request failed: {e2}") from e2
            else:
                logger.error("LLM call failed after %.1fs: %s",
                             (timeout or cfg.llm_timeout), e)
                raise RuntimeError(f"LLM API request failed: {e}") from e
        else:
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
    input_hash = hashlib.md5(user_message.encode()).hexdigest()[:8]
    output_path = f"tmp/raw/resp_{timestamp}_{input_hash}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(raw_content)
    logger.info("Saved raw response to %s", output_path)

    # ── Parse JSON (with fallback + retries) ────────────────────────
    max_retries = 3
    content = _preprocess_json(raw_content)
    if content is None:
        raise ValueError(
            f"LLM returned no JSON-like content. Raw response (first 500 chars): {raw_content[:500]!r}"
        )

    for attempt in range(max_retries):
        try:
            return json.loads(content, strict=True)  # fast path — works when response_format succeeded
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse attempt %d failed (%s)", attempt + 1, exc.msg)
            if attempt == max_retries - 1:
                raise ValueError(
                    f"Failed to parse LLM JSON after {max_retries} attempts. "
                    f"Raw content (first 500 chars): {content[:500]!r}"
                ) from exc
            # Retry with relaxed parser, then fix bare quotes if needed
            try:
                return json.loads(content, strict=False)
            except json.JSONDecodeError:
                fixed = _fix_bare_quotes_in_body_field(content)
                if fixed is not None:
                    content = fixed
                    continue  # re-try with fixed content
                break  # give up this path

    raise ValueError("JSON parsing failed after all fallback strategies.")


def _preprocess_json(raw_content: str) -> str | None:
    """Strip markdown code blocks, extract first JSON object.

    Returns None if no JSON-like content can be found (e.g., plain English text).
    This lets the caller distinguish "no JSON at all" from "JSON but broken."
    """
    if not isinstance(raw_content, str):
        return None
    # Strip markdown code blocks
    stripped = re.sub(r'^```(?:json)?\s*\n', '', raw_content.strip())
    # Extract first JSON object
    json_match = re.search(r'\{.*\}', stripped, re.DOTALL)
    if not json_match:
        return None  # No JSON-like content — let json.loads raise a clear error
    return json_match.group(0)


def _fix_bare_quotes_in_body_field(content: str) -> str | None:
    """Find the body field value and escape unescaped quotes inside it.

    Walks through the JSON string character-by-character, recognizing escaped
    sequences (\\", \\\\, \\n, etc.) so real closing-quotes are not confused
    with bare quotes in the content.
    """
    m = re.search(r'"body"\s*:\s*', content)
    if not m:
        return None

    after_key = m.end()
    if after_key >= len(content) or content[after_key] != '"':
        return None

    # Walk forward, skipping escaped sequences, find the real closing quote
    j = after_key + 1
    while j < len(content):
        c = content[j]

        if c == '\\' and j + 1 < len(content) and content[j+1] in ('"', '\\', '/', 'n', 't', 'r'):
            j += 2
            continue

        if c == '"':
            rest_after_quote = content[j+1:].lstrip()
            if not rest_after_quote or rest_after_quote[0] in (',', '}'):
                raw_body = content[after_key + 1 : j]
                fixed_parts: list[str] = []
                k = 0
                while k < len(raw_body):
                    ch = raw_body[k]
                    if ch == '\\' and k + 1 < len(raw_body) and raw_body[k+1] in ('"', '\\', '/', 'n', 't', 'r'):
                        fixed_parts.append(ch)
                        fixed_parts.append(raw_body[k+1])
                        k += 2
                    elif ch == '"':
                        fixed_parts.append('\\"')
                        k += 1
                    else:
                        fixed_parts.append(ch)
                        k += 1

                before = content[: after_key + 1]
                after = content[j:]
                return before + ''.join(fixed_parts) + '"' + after
        j += 1

    return None


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
    """Split text at paragraph boundaries, chunk oversized paragraphs at sentences.

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
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    # Resolve default threshold from config (lazy, per-call)
    if max_chars is None:
        max_chars = _get_chunk_threshold()

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

    Key improvements over naive word-frequency:
      - Extracts proper nouns (capitalized entities) from title + body
      - Filters out single generic English words not in a whitelist
      - Combines adjacent frequent terms into multi-word phrases when useful
      - Prefers domain-specific terms (brands, organizations, systems)
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

    # Generic single words that are almost never useful as tags
    GENERIC_WORDS = {
        'the', 'and', 'from', 'into', 'over', 'under', 'between', 'through',
        'during', 'before', 'after', 'above', 'below', 'within', 'across',
        'about', 'against', 'along', 'among', 'around', 'behind', 'beyond',
        'since', 'until', 'upon', 'toward', 'towards',
        'system', 'payment', 'china', 'country', 'bank', 'data',
        'information', 'process', 'service', 'user', 'network',
        'document', 'file', 'text', 'content', 'example',
        'channel', 'series', 'program', 'programs', 'programming',
        'original', 'retrieved', 'archived', 'published', 'based',
    }

    # Words that ARE useful as tags (domain-specific)
    USEFUL_SINGLE = {
        'china', 'france', 'london', 'tokyo', 'beijing', 'america', 'germany',
        'russia', 'japan', 'united states', 'european union',
        'international', 'government', 'regulation', 'compliance',
        'security', 'encryption', 'blockchain', 'cryptocurrency',
    }

    # Extract words from body (lowercase, >= 3 chars)
    body_words = re.findall(r'[a-zA-Z]{3,}', body.lower())
    word_freq = Counter(w for w in body_words if w not in STOP_WORDS and len(w) > 2)

    # Also extract title keywords with higher weight
    title_words = re.findall(r'[a-zA-Z]{3,}', title.lower())
    title_freq = Counter(title_words)

    # Combine: title words get higher weight
    combined = Counter(word_freq)
    for w, c in title_freq.items():
        combined[w] += c * 2

    # ── Extract proper nouns from title and body ────────────────
    # Title is usually the most important entity.
    # Also extract capitalized phrases from the first few paragraphs (likely entities).
    def _extract_proper_nouns(text: str) -> list[str]:
        """Extract capitalized words/phrases that look like proper nouns."""
        # Title itself
        title_parts = re.findall(r'[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*', text[:200])
        # Capitalized entities in body (first 5K chars)
        entity_phrases = re.findall(
            r'(?<![a-z])([A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+){1,3})(?![a-z])',
            text[:min(len(text), 5000)]
        )
        return title_parts + entity_phrases

    proper_nouns = _extract_proper_nouns(title) + _extract_proper_nouns(body)
    noun_freq = Counter(p for p in proper_nouns if len(p.split()) <= 3 and len(p) > 2)

    # ── Build tag candidates ────────────────────────────────────
    tags: list[str] = []
    seen: set[str] = set()

    # Phase 1: Proper nouns (highest priority - they're entity-specific)
    for noun, count in noun_freq.most_common(3):
        if len(tags) >= 5:
            break
        tag_lower = noun.lower().strip()
        if tag_lower not in seen and tag_lower not in GENERIC_WORDS and len(tag_lower) > 2:
            tags.append(tag_lower)
            seen.add(tag_lower)

    # Phase 2: High-frequency domain words (only those with >= 3 occurrences)
    for word, count in combined.most_common(40):
        if len(tags) >= 5:
            break
        if word in seen or word in STOP_WORDS:
            continue
        # Only include single-word tags that are either generic-useful OR appear frequently
        if count < 3 and word not in USEFUL_SINGLE:
            continue
        tags.append(word)
        seen.add(word)

    # Phase 3: Title-based fallback (title words we haven't used yet)
    for w, _ in title_freq.most_common(10):
        if len(tags) >= 5:
            break
        if w not in seen and w.lower() not in GENERIC_WORDS and w.lower() not in STOP_WORDS:
            tags.append(w.lower())
            seen.add(w.lower())

    # Final filter: remove any remaining single generic words
    final_tags = [t for t in tags if t not in GENERIC_WORDS or len(t.split()) > 1]

    return final_tags[:5] if len(final_tags) >= 3 else final_tags


def _format_text_single(raw: str, source_type: str = "web", *, system_prompt: str | None = None) -> Dict[str, Any]:
    """Single-shot formatting — original behavior for small documents."""
    prompt = system_prompt if system_prompt is not None else get_system_prompt(source_type)
    result = call_llm(prompt, raw.strip(), schema=FORMATTER_SCHEMA)

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
    threshold = _get_chunk_threshold()
    if raw_len > threshold:
        logger.info("Large text: %d chars — calling chunked processor", raw_len)
        return _format_text_chunked(raw, source_type)

    logger.info("Small text: %d chars — single-shot", raw_len)
    return _format_text_single(raw, source_type)


def _format_text_async_impl(raw: str, source_type: str, *, system_prompt: str | None = None) -> Dict[str, Any]:
    """Internal implementation of async formatting that respects custom system prompts."""
    if not raw.strip():
        raise ValueError("Input text is empty")

    raw_len = len(raw)
    threshold = _get_chunk_threshold()
    if raw_len > threshold:
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
    threshold = _get_chunk_threshold()
    if raw_len > threshold:
        return _format_text_chunked(raw, source_type)

    return _format_text_single(raw, source_type, system_prompt=system_prompt)


__all__ = ["call_llm", "format_text", "format_text_async", "format_text_with_system"]

# Re-export writer functions for convenience
from .writer import format_md, write_to_md  # noqa: F401, E402
