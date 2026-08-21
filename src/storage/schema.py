"""Schema definitions, table creation, and sqlite-vec loader."""

from __future__ import annotations

import importlib
import logging
import sys
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


@runtime_checkable
class SQLiteVecModule(Protocol):
    """Structural type for the dynamically loaded ``sqlite_vec`` package.

    The loader returns an untyped module object; this protocol gives static
    checkers (and readers) the subset of its API we rely on.
    """

    def serialize_float32(self, vector: list[float]) -> bytes:
        """Pack a float vector into the little-endian float32 BLOB format."""
        ...

    def load(self, conn: sqlite3.Connection) -> None:
        """Register the vec0 extension functions on *conn*."""
        ...


# ---------------------------------------------------------------------------
# Dynamic loader for the third-party ``sqlite-vec`` package.
#
# Strategy 1: direct ``import sqlite_vec`` — works when installed normally.
# Strategy 2: filesystem path via
#   ``importlib.metadata.distribution("sqlite-vec").files`` — robust across
#   editable installs, wheels, and different Python versions.
# ---------------------------------------------------------------------------
_sqlite_vec: SQLiteVecModule | None = None


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
    except PackageNotFoundError as exc:
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
    if spec is None or spec.loader is None:
        _msg = "Failed to create import spec for sqlite_vec.__init__"
        raise RuntimeError(_msg)
    mod = _util.module_from_spec(spec)
    sys.modules["_third_party_sqlite_vec"] = mod
    spec.loader.exec_module(mod)
    _sqlite_vec = mod
    return mod


# Cached reference for convenience — callers use this instead of calling
# ``_load_sqlite_vec()`` every time.
_SQLITE_VEC: SQLiteVecModule = _load_sqlite_vec()  # type: ignore[assignment]


class _StoreBase:
    """Shared state/contract for the insert and search operation mixins.

    ``SQLiteVecStore`` owns the connection lifecycle; the mixins declare the
    attributes they rely on here so static checkers see a complete type.
    """

    conn: sqlite3.Connection
    _schema_ready: bool

    def _setup_schema(self) -> None:
        """Create tables and triggers if needed (implemented by _InsertOps)."""
        raise NotImplementedError
