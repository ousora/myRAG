"""Text formatter — structured output from raw copied text."""


import json
import re
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict

import httpx

from .prompts import get_system_prompt


ENDPOINT = "http://192.168.191.112:8081/v1/chat/completions"
MODEL = "Qwen/QwQ-32B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Compact.gguf"

_executor = None


def get_executor() -> ThreadPoolExecutor:
    """Lazy-initialize the shared thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2)
    return _executor


def format_text(raw: str, source_type: str = "web") -> Dict[str, Any]:
    """Format raw copied text into structured knowledge content.

    Args:
        raw: The raw text copied from a webpage (Ctrl+A + Copy).
        source_type: Source context for the LLM to determine cleanup aggressiveness.
                     Options: 'web', 'markdown', 'pdf_clip'.

    Returns:
        Dict with keys: title, tags, metadata, chunks.

    Raises:
        httpx.HTTPError: If the API request fails.
        ValueError: If the LLM returns invalid JSON or unexpected response format.
    """
    if not raw.strip():
        raise ValueError("Input text is empty")

    prompt = get_system_prompt(source_type)

    try:
        response = httpx.post(
            ENDPOINT,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": raw.strip()},
                ],
                "temperature": 0.3,
                "max_tokens": 8192,
            },
            timeout=180,
        )

        response.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"LLM API request failed: {e}") from e

    try:
        raw_content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"LLM returned invalid format: {e}") from e

    # Strip code block markers from LLM response
    content = re.sub(r'^```(?:json)?\s*\n', '', raw_content).strip() if isinstance(raw_content, str) else raw_content
    
    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON: {e}. Raw response: {content!r}"
        ) from e

    return result


def format_text_async(raw: str, source_type: str = "web") -> Future[Dict[str, Any]]:
    """Submit formatting task to thread pool. Returns a Future."""
    future = get_executor().submit(format_text, raw, source_type)
    return future


__all__ = ["format_text", "format_text_async"]

# Re-export writer functions for convenience
from .writer import format_md, write_to_md  # noqa: F401