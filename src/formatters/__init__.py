"""Text formatter — structured output from raw copied text.

Handles both single-shot and chunked (large document) modes.
Auto-detects which path to use based on input size.
"""

from __future__ import annotations

import atexit
import httpx
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from ._internal import (
    effective_chunk_threshold,
    _detect_cjk_ratio,
    _format_text_chunked,
    _format_text_single,
    _format_text_async_impl,
    call_llm,
    call_llm_raw,
    _preprocess_json,
    _fix_bare_quotes_in_body_field,
    format_cached,
)
from .writer import format_md, write_to_md

logger = logging.getLogger(__name__)

# ── Process-wide formatting cache ───────────────────────────────────────
# Delegates to formatters.cache module for LRU caching logic.


def _shutdown_executor() -> None:
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

_executor: ThreadPoolExecutor | None = None


def get_executor() -> ThreadPoolExecutor:
    """Lazy-initialize the shared thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2)
    return _executor


def shutdown() -> None:
    """Shutdown the shared thread pool executor.

    Call this when the formatter module is no longer needed (e.g. at the end
    of a long-running session or in test teardown) to release worker threads
    immediately instead of waiting for ``atexit``.
    """
    _shutdown_executor()


# ── Public API ──────────────────────────────────────────────────────────


def format_text(raw: str, source_type: str = "web") -> dict[str, Any]:
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


def format_text_async(
    raw: str,
    source_type: str = "web",
    *,
    system_prompt: str | None = None,
) -> Future[dict[str, Any]]:
    """Submit formatting task to thread pool. Returns a Future.

    Args:
        raw: The text to process.
        source_type: Source context for the LLM ('web', 'markdown', 'pdf_clip').
        system_prompt: Optional custom system prompt (overrides default).
                       Useful for RAG queries where you want a different prompt style.

    """
    future = get_executor().submit(
        _format_text_async_impl, raw, source_type, system_prompt=system_prompt,
    )
    return future


def format_text_with_system(
    raw: str,
    source_type: str = "web",
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
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


__all__ = [
    "call_llm",
    "call_llm_raw",
    "format_text",
    "format_text_async",
    "format_text_with_system",
    "format_md",
    "write_to_md",
    "shutdown",
]
