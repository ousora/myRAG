"""Text formatter — structured output from raw copied text."""


import json
import re
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict

import httpx

from .prompts import get_system_prompt


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

    cfg = _get_config()

    try:
        response = httpx.post(
            cfg.llm_endpoint,
            json={
                "model": cfg.llm_model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": raw.strip()},
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
    content = re.sub(r'^```(?:json)?\s*\n', '', raw_content).strip() if isinstance(raw_content, str) else raw_content
    
    # Extract the first valid JSON object — handles cases where LLM outputs extra text after JSON
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        content = json_match.group(0)

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