"""LLM API transport layer — raw chat-completion calls and JSON repair.

This module isolates everything that talks to the OpenAI-compatible LLM
endpoint:
- ``call_llm``      — JSON-mode call with schema rejection retry + parse repair
- ``call_llm_raw``  — free-text call (used for RAG answer generation)
- ``_call_llm_api`` — shared HTTP POST with error wrapping

Formatting logic that *uses* these lives in ``_internal.py``; public API is
re-exported from ``formatters/__init__.py``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


def _get_config() -> Config:
    """Lazy-load config on first call."""
    from config import get_config
    return get_config()


# Error-message substrings that indicate the endpoint rejected the
# ``response_format``/JSON-schema payload itself (rather than failing for an
# unrelated reason). When matched on a transient HTTP status we retry once
# without the schema constraint:
#   - "peg": llama.cpp converts JSON schemas to grammars internally and its
#     converter reports failures as "PEG error at char N".
#   - "schema" / "response_format": vLLM, Ollama and llama.cpp all mention one
#     of these words when the requested structured-output feature is
#     unavailable or the schema fails to compile.
_SCHEMA_REJECT_MARKERS = ("peg", "schema", "response_format")


def _preprocess_json(raw_content: object) -> str | None:
    """Strip markdown code blocks, extract first JSON object with balanced braces.

    Returns None if no JSON-like content can be found (e.g., plain English text).
    This lets the caller distinguish "no JSON at all" from "JSON but broken."
    """
    if not isinstance(raw_content, str):
        return None
    # Strip markdown code blocks
    stripped = re.sub(r"^```(?:json)?\s*\n", "", raw_content.strip())

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
    r"""Find the body field value and escape unescaped quotes inside it.

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

        if c == "\\" and j + 1 < len(content) and content[j+1] in ('"', "\\", "/", "n", "t", "r", "u"):
            skip = 2 if content[j+1] != "u" else 6
            j += skip
            continue

        if c == '"':
            rest_after_quote = content[j+1:].lstrip()
            if not rest_after_quote or rest_after_quote[0] in (",", "}"):
                raw_body = content[after_key + 1 : j]
                fixed_parts: list[str] = []
                k = 0
                while k < len(raw_body):
                    ch = raw_body[k]
                    if ch == "\\" and k + 1 < len(raw_body) and raw_body[k+1] in ('"', "\\", "/", "n", "t", "r", "u"):
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
                return before + '"' + "".join(fixed_parts) + '"' + after
        j += 1

    return None


def _call_llm_api(payload: dict[str, Any], timeout: int | None) -> httpx.Response:
    cfg = _get_config()
    try:
        response = httpx.post(cfg.llm_endpoint, json=payload, timeout=timeout or cfg.llm_timeout)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.exception("LLM call failed (timeout=%ss)", timeout or cfg.llm_timeout)
        err_msg = f"LLM API request failed: {e}"
        raise RuntimeError(err_msg) from e
    else:
        return response


def call_llm(
    system_prompt: str,
    user_message: str,
    *,
    max_tokens: int | None = None,
    timeout: int | None = None,
    schema: dict | None = None,
) -> dict:
    """Send a chat completion expecting JSON back, with layered recovery.

    Recovery ladder, in order:
      1. Strict JSON parse of the extracted object.
      2. Non-strict parse (tolerates NaN / control chars).
      3. Escape bare quotes inside the ``body`` field, then parse again.
    If the endpoint rejects the schema itself (see ``_SCHEMA_REJECT_MARKERS``)
    with a transient HTTP status, the request is retried once without
    ``response_format`` so plain-JSON prompting can still succeed.
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
        payload["response_format"] = {"type": "json_object", "schema": schema}
    try:
        response = _call_llm_api(payload, timeout)
    except RuntimeError as exc:
        # _call_llm_api wraps the httpx error; recover the HTTP response from
        # the cause chain so schema rejections can be detected and retried.
        cause = exc.__cause__
        resp_for_retry = getattr(cause, "response", None)
        schema_rejected = (
            isinstance(cause, httpx.HTTPStatusError)
            and resp_for_retry is not None
            and resp_for_retry.status_code in {500, 503, 429}
            and any(marker in resp_for_retry.text.lower() for marker in _SCHEMA_REJECT_MARKERS)
        )
        if schema_rejected and resp_for_retry is not None:
            logger.warning(
                "Schema rejected (HTTP %d), retrying without schema",
                resp_for_retry.status_code,
            )
            payload.pop("response_format", None)
            response = _call_llm_api(payload, timeout)
        else:
            raise
    try:
        raw_content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logger.exception("LLM returned unexpected response structure")
        invalid_msg = f"LLM returned invalid format: {e}"
        raise ValueError(invalid_msg) from e
    input_chars = len(user_message)
    output_chars = len(raw_content)
    logger.info("LLM call: %d chars in -> %d chars out", input_chars, output_chars)
    if getattr(cfg, "debug_log_llm_responses", False):
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        input_hash = hashlib.md5(user_message.encode()).hexdigest()[:8]
        output_path = f"tmp/raw/resp_{timestamp}_{input_hash}.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(raw_content)
    max_retries = 3
    content = _preprocess_json(raw_content)
    if content is None:
        no_json_msg = f"LLM returned no JSON-like content. Raw: {raw_content[:500]!r}"
        raise ValueError(no_json_msg)

    last_error: json.JSONDecodeError | None = None

    def _loads(text: str, *, lenient: bool) -> dict | None:
        """Parse *text* as a JSON object; None when parsing fails or not an object."""
        nonlocal last_error
        try:
            loaded = json.loads(text, strict=not lenient)
        except json.JSONDecodeError as exc:
            last_error = exc
            return None
        if isinstance(loaded, dict):
            return loaded
        logger.warning("LLM JSON payload is %s, expected an object", type(loaded).__name__)
        return None

    for attempt in range(max_retries):
        parsed = _loads(content, lenient=False)
        if parsed is not None:
            return parsed
        detail = last_error.msg if last_error is not None else "not a JSON object"
        logger.warning("JSON parse attempt %d failed (%s)", attempt + 1, detail)
        if attempt == max_retries - 1:
            parse_fail_msg = (
                f"Failed to parse LLM JSON after {max_retries} attempts. Raw: {content[:500]!r}"
            )
            raise ValueError(parse_fail_msg) from last_error
        # Lenient pass tolerates NaN / control characters.
        parsed = _loads(content, lenient=True)
        if parsed is not None:
            return parsed
        fixed = _fix_bare_quotes_in_body_field(content)
        if fixed is not None:
            content = fixed
            continue
        break
    msg = "JSON parsing failed after all fallback strategies."
    raise ValueError(msg)


def call_llm_raw(
    system_prompt: str,
    user_message: str,
    *,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> str:
    """Free-text chat completion — returns the message content verbatim."""
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
    response = _call_llm_api(payload, timeout)
    try:
        answer: str = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logger.exception("LLM returned unexpected response structure")
        invalid_msg = f"LLM returned invalid format: {e}"
        raise ValueError(invalid_msg) from e
    else:
        return answer
