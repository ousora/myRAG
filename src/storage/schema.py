"""Schema definitions, table creation, and sqlite-vec loader."""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)


# Hex ranges for CJK Unified Ideographs Blocks A–D, converted to \\uXXXX regex patterns at module load time. The basic range alone misses ~1% of modern Chinese text in Extensions E/F/G which are rarely used outside specialized domains (historical/archaic).
_CJK_RANGE_HEX = [
    "4e00-9fff",        # CJK Unified Ideographs (basic plane)
    "3400-4dbf",        # Block A — historical/archaic characters
    "20000-2a6df",      # Block B — rare characters, names, place names
    "2a700-2ebef",      # Block C — rare variants and archaic forms
]

# Each entry like \\u4e00-\\u9fff is a valid regex character class range with \u escapes.
_CJK_RANGE = [f"\\u{r}" for r in _CJK_RANGE_HEX]


# ---------------------------------------------------------------------------
# Dynamic loader for the third-party ``sqlite-vec`` package.
#
# Strategy 1: direct ``import sqlite_vec`` — works when installed normally.
# Strategy 2: filesystem path via
#   ``importlib.metadata.distribution("sqlite-vec").files`` — robust across
#   editable installs, wheels, and different Python versions.
# ---------------------------------------------------------------------------
_sqlite_vec: object | None = None


def _load_sqlite_vec() -> object:
    """Return the third-party ``sqlite_vec`` module (loaded once).

    Tries two strategies in order:
      1. ``importlib.import_module("sqlite_vec")`` — works for pip/uv-installed
         packages (the canonical case).
      2. Locate via distribution metadata → file-based loading — robust across
         editable installs, wheels, and different Python versions.

    Raises RuntimeError if neither strategy succeeds.
    """
    global _sqlite_vec
    if _sqlite_vec is not None:
        return _sqlite_vec

    # Strategy 1: Direct import (canonical install path).
    try:
        _sqlite_vec = importlib.import_module("sqlite_vec")
    except ImportError:
        pass
    else:
        return _sqlite_vec

    import importlib.util as _util
    from importlib.metadata import PackageNotFoundError, distribution

    # Strategy 2: Locate __init__.py via the distribution's file list — robust
    # across editable installs, wheels, and different Python versions.
    try:
        dist = distribution("sqlite-vec")
    except PackageNotFoundError as exc:  # type: ignore[attr-defined]
        _msg = ("The 'sqlite-vec' package is required but not installed.\n"
                "Install it with: pip install sqlite-vec\n"
                "(or: uv add --dev sqlite-vec)")
        raise RuntimeError(_msg) from exc

    init_py = next(
        (f for f in dist.files or [] if str(f) == "sqlite_vec/__init__.py"),
        None,
    )
    if init_py is None:
        _msg = ("Could not locate sqlite_vec.__init__ inside the 'sqlite-vec' distribution.\n"
                "The installed version may be corrupted or incompatible.")
        raise RuntimeError(_msg)

    spec = _util.spec_from_file_location(
        "_third_party_sqlite_vec", str(dist.locate_file(init_py)),
    )
    mod = _util.module_from_spec(spec)  # type: ignore[union-attr]
    sys.modules["_third_party_sqlite_vec"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _sqlite_vec = mod
    return mod


# Cached reference for convenience — callers use this instead of calling
# ``_load_sqlite_vec()`` every time.
_SQLITE_VEC = _load_sqlite_vec()
