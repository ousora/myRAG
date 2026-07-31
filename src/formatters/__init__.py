"""Text formatter — structured output from raw copied text.

Handles both single-shot and chunked (large document) modes.
Auto-detects which path to use based on input size.
"""

import atexit
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict

import httpx

from .prompts import (
    get_system_prompt,
    get_chunked_system_prompt,
    validate_format_output,
    try_fix_common_issues,
)
from .constants import FORMATTER_SCHEMA, CHUNKED_SCHEMA
from .tags import extract_tags_from_body
from .cache import format_cached

logger = logging.getLogger(__name__)

# ── Process-wide formatting cache ───────────────────────────────────────
# Delegates to formatters.cache module for LRU caching logic.


def _shutdown_executor():
    """Cleanup the shared thread pool executor on process exit."""
    global _executor
    if _executor is not None:
        logger.info("Shutting down formatter thread pool executor")
        try:
            _executor.shutdown(wait=True)
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup at exit
            logger.debug("Executor shutdown raised: %s", exc)
        finally:
            _executor = None


atexit.register(_shutdown_executor)

# ── Internal helpers ────────────────────────────────────────────────────


def _get_config():
    """Lazy-load config on first call."""
    from config import get_config
    return get_config()


# ── Chunking threshold ──────────────────────────────────────────────────
# Texts above this many characters trigger chunked processing.
# ~28K chars ≈ 7000 tokens — safe for most local LLMs (English, 4 chars/token).

# Default chars-per-token ratios by language family used to adjust the threshold.
_CJK_CHARS_PER_TOKEN = 1.0
_ENGLISH_CHARS_PER_TOKEN = 4.0


def _detect_cjk_ratio(text: str) -> float:
    """Estimate the proportion of CJK characters in text (0.0–1.0).

    Counts only actual CJK ranges (CJK Unified Ideographs, Hiragana, Katakana)
    as CJK; ASCII letters/digits as English-like; everything else (Cyrillic,
    Arabic, Devanagari, emoji, punctuation) is excluded from the ratio to avoid
    skewing documents that contain non-CJK multilingual content.
    Whitespace is also excluded.
    """
    if not text:
        return 0.0

    def _is_cjk(ch: str) -> bool:
        cp = ord(ch)
        return (
            (0x4E00 <= cp <= 0x9FFF) or   # CJK Unified Ideographs
            (0x3040 <= cp <= 0x309F) or   # Hiragana
            (0x30A0 <= cp <= 0x30FF)      # Katakana
        )

    cjk = 0
    latin = 0
    for ch in text:
        if ch.isspace():
            continue
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            latin += 1
        elif _is_cjk(ch):
            cjk += 1
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
    elif ratio > 0.1:
        # Mixed — linear interpolation between the two extremes.
        mix = (ratio - 0.1) / 0.4  # 0 at 10% CJK, 1 at 50% CJK
        chars_per_token = _ENGLISH_CHARS_PER_TOKEN - mix * (
            _ENGLISH_CHARS_PER_TOKEN - _CJK_CHARS_PER_TOKEN
        )
        return int(base * (_CJK_CHARS_PER_TOKEN / chars_per_token))
    else:
        # Mostly English — use base threshold unchanged.
        return base

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

    # Save raw response for debugging — only when explicitly enabled.
    if getattr(cfg, "debug_log_llm_responses", False):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_hash = hashlib.md5(user_message.encode()).hexdigest()[:8]
        output_path = f"tmp/raw/resp_{timestamp}_{input_hash}.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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


def call_llm_raw(system_prompt: str, user_message: str, *,
                 max_tokens: int | None = None,
                 timeout: int | None = None) -> str:
    """Call the LLM and return the raw response text (no JSON parsing).

    Use this for free-text generation (e.g. RAG answer synthesis) where the
    model is expected to reply in natural language rather than a JSON object.
    Mirrors ``call_llm``'s request/error handling but skips JSON extraction.
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
    try:
        response = httpx.post(cfg.llm_endpoint, json=payload, timeout=timeout or cfg.llm_timeout)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("LLM call failed after %.1fs: %s", (timeout or cfg.llm_timeout), e)
        raise RuntimeError(f"LLM API request failed: {e}") from e

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logger.error("LLM returned unexpected response structure: %s", e)
        raise ValueError(f"LLM returned invalid format: {e}") from e


def _preprocess_json(raw_content: str) -> str | None:
    """Strip markdown code blocks, extract first JSON object with balanced braces.

    Returns None if no JSON-like content can be found (e.g., plain English text).
    This lets the caller distinguish "no JSON at all" from "JSON but broken."
    """
    if not isinstance(raw_content, str):
        return None
    # Strip markdown code blocks
    stripped = re.sub(r'^```(?:json)?\s*\n', '', raw_content.strip())

    brace_start = stripped.find("{")
    if brace_start == -1:
        return None  # No JSON-like content — let json.loads raise a clear error

    depth, end = 0, -1
    in_string = False
    escape_next = False
    for i in range(brace_start, len(stripped)):
        ch = stripped[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1 or end <= brace_start:
        return None  # Unbalanced braces — not a valid JSON object start

    return stripped[brace_start : end + 1]


def _fix_bare_quotes_in_body_field(content: str) -> str | None:
    """Find the body field value and escape unescaped quotes inside it.

    Walks through the JSON string character-by-character, recognizing escaped
    sequences (\\", \\\\, \\n, etc.) so real closing-quotes are not confused
    with bare quotes in the content.

    Returns the modified JSON string if any bare quotes were found and escaped,
    or None if no modification is needed (valid JSON).
    """
    m = re.search(r'"body"\s*:\s*', content)
    if not m:
        return None

    after_key = m.end()
    if after_key >= len(content) or content[after_key] != '"':
        return None

    # Walk forward, skipping escaped sequences, find the real closing quote.
    # Track whether any bare quotes were encountered (i.e., actual changes needed).
    has_bare_quotes = False
    j = after_key + 1
    while j < len(content):
        c = content[j]

        if c == '\\' and j + 1 < len(content) and content[j+1] in ('"', '\\', '/', 'n', 't', 'r', 'u'):
            skip = 2 if content[j+1] != 'u' else 6
            j += skip
            continue

        if c == '"':
            rest_after_quote = content[j+1:].lstrip()
            if not rest_after_quote or rest_after_quote[0] in (',', '}'):
                raw_body = content[after_key + 1 : j]
                fixed_parts: list[str] = []
                k = 0
                while k < len(raw_body):
                    ch = raw_body[k]
                    if ch == '\\' and k + 1 < len(raw_body) and raw_body[k+1] in ('"', '\\', '/', 'n', 't', 'r', 'u'):
                        fixed_parts.append(ch)
                        fixed_parts.append(raw_body[k+1])
                        k += 2
                    elif ch == '"':
                        has_bare_quotes = True
                        fixed_parts.append('\\"')
                        k += 1
                    else:
                        fixed_parts.append(ch)
                        k += 1

                if not has_bare_quotes:
                    # No bare quotes found — the original JSON is already valid.
                    return None

                before = content[:after_key]
                after = content[j + 1:]  # skip past the closing quote itself
                return before + '"' + ''.join(fixed_parts) + '"' + after
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


def _format_text_single(raw: str, source_type: str = "web", *, system_prompt: str | None = None) -> Dict[str, Any]:
    """Single-shot formatting — original behavior for small documents."""
    def _compute() -> Dict[str, Any]:
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
            import datetime
            result["metadata"]["created_at"] = datetime.datetime.now().isoformat()

        return result

    return format_cached(raw, source_type, system_prompt, _compute)


def _format_text_chunked_uncached(raw: str, source_type: str = "pdf", *, system_prompt: str | None = None) -> Dict[str, Any]:
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
    if system_prompt is not None:
        base_system_prompt = system_prompt
    else:
        base_system_prompt = get_chunked_system_prompt(0, total)  # title will be empty until merge

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
    sections: list[dict] = []
    for match in re.finditer(r'^(#{2,3})\s+(.+)$', body, re.MULTILINE):
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


def _format_text_chunked(raw: str, source_type: str = "pdf", *, system_prompt: str | None = None) -> Dict[str, Any]:
    """Chunked formatting for large documents, with process-wide caching.

    Delegates to ``_format_text_chunked_uncached`` and caches the merged result
    so re-ingesting identical text skips the (expensive) multi-call LLM pass.
    """
    def _compute() -> Dict[str, Any]:
        return _format_text_chunked_uncached(raw, source_type, system_prompt=system_prompt)

    return format_cached(raw, source_type, system_prompt, _compute)


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

    # Auto-dispatch based on CJK-aware text length.
    raw_len = len(raw)
    threshold = effective_chunk_threshold(raw)
    if raw_len > threshold:
        logger.info(
            "Large text (%d chars, CJK ratio=%.2f): calling chunked processor",
            raw_len, _detect_cjk_ratio(raw),
        )
        return _format_text_chunked(raw, source_type)

    logger.info("Small text: %d chars — single-shot", raw_len)
    return _format_text_single(raw, source_type)


def _format_text_async_impl(raw: str, source_type: str, *, system_prompt: str | None = None) -> Dict[str, Any]:
    """Internal implementation of async formatting that respects custom system prompts."""
    if not raw.strip():
        raise ValueError("Input text is empty")

    raw_len = len(raw)
    threshold = effective_chunk_threshold(raw)
    if raw_len > threshold:
        return _format_text_chunked(raw, source_type, system_prompt=system_prompt)

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
    threshold = effective_chunk_threshold(raw)
    if raw_len > threshold:
        return _format_text_chunked(raw, source_type, system_prompt=system_prompt)

    return _format_text_single(raw, source_type, system_prompt=system_prompt)


__all__ = ["call_llm", "call_llm_raw", "format_text", "format_text_async", "format_text_with_system"]

# Re-export writer functions for convenience
from .writer import format_md, write_to_md  # noqa: F401, E402
