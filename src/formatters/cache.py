"""Process-wide formatting cache for the RAG pipeline.

Identical (raw, source_type, system_prompt) inputs are formatted only once
per process — avoids re-running the (expensive) LLM call on re-ingest.
Bounded LRU keyed by a content hash.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any, Callable

logger = logging.getLogger(__name__)

_FORMAT_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_FORMAT_CACHE_LOCK = threading.Lock()
_FORMAT_CACHE_MAX = 256


def format_cache_key(raw: str, source_type: str, system_prompt) -> str:
    """Generate a cache key from the input parameters."""
    sp = system_prompt or ""
    return hashlib.md5(
        f"{source_type}|{sp}|{raw}".encode("utf-8")
    ).hexdigest()


def format_cached(raw: str, source_type: str, system_prompt, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Compute and cache a formatting result using LRU cache.
    
    Args:
        raw: The raw text to format.
        source_type: The source type hint.
        system_prompt: The system prompt used.
        compute: Function that computes the result if not cached.
        
    Returns:
        The formatting result (from cache or computed).
    """
    key = format_cache_key(raw, source_type, system_prompt)
    with _FORMAT_CACHE_LOCK:
        if key in _FORMAT_CACHE:
            _FORMAT_CACHE.move_to_end(key)
            return _FORMAT_CACHE[key]
    result = compute()
    with _FORMAT_CACHE_LOCK:
        _FORMAT_CACHE[key] = result
        if len(_FORMAT_CACHE) > _FORMAT_CACHE_MAX:
            _FORMAT_CACHE.popitem(last=False)
    return result


def clear_format_cache() -> None:
    """Clear the entire format cache. Useful for testing or memory pressure."""
    with _FORMAT_CACHE_LOCK:
        _FORMAT_CACHE.clear()
