"""Process-wide formatting cache for the RAG pipeline.

Identical (raw, source_type, system_prompt) inputs are formatted only once
per process — avoids re-running the (expensive) LLM call on re-ingest.
Bounded LRU keyed by a content hash.

Stored results are deep-copied on both put and get so callers can freely
mutate their returned dict without poisoning the cache for future lookups.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_FORMAT_CACHE: OrderedDict[str, dict] = OrderedDict()
_FORMAT_CACHE_LOCK = threading.Lock()
_FORMAT_CACHE_MAX = 256


def format_cache_key(raw: str, source_type: str, system_prompt: str | None) -> str:
    """Generate a cache key from the input parameters.

    Hashes each component separately to avoid the system prompt (several KB)
    dominating the hash computation.
    """
    sp_hash = hashlib.sha256((system_prompt or "").encode()).hexdigest()
    raw_hash = hashlib.sha256(raw.encode()).hexdigest()
    return hashlib.md5(
        f"{source_type}|{sp_hash}|{raw_hash}".encode()
    ).hexdigest()


def format_cached(raw: str, source_type: str, system_prompt: str | None,
                  compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Compute and cache a formatting result using LRU cache.

    Args:
        raw: The raw text to format.
        source_type: The source type hint.
        system_prompt: The system prompt used.
        compute: Function that computes the result if not cached.

    Returns:
        A deep copy of the formatting result (from cache or computed). The
        copy protects the cache: mutating the returned dict never affects
        subsequent lookups.

    """
    key = format_cache_key(raw, source_type, system_prompt)
    with _FORMAT_CACHE_LOCK:
        if key in _FORMAT_CACHE:
            _FORMAT_CACHE.move_to_end(key)
            return copy.deepcopy(_FORMAT_CACHE[key])
    result = compute()
    with _FORMAT_CACHE_LOCK:
        _FORMAT_CACHE[key] = copy.deepcopy(result)
        if len(_FORMAT_CACHE) > _FORMAT_CACHE_MAX:
            _FORMAT_CACHE.popitem(last=False)
    return result


def clear_format_cache() -> None:
    """Clear the entire format cache. Useful for testing or memory pressure."""
    with _FORMAT_CACHE_LOCK:
        _FORMAT_CACHE.clear()
