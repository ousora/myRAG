"""Shared fixtures and helpers for storage tests."""

import contextlib
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def store():
    """Create a temp-file SQLiteVecStore for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from storage.sqlite_vec import SQLiteVecStore
    db = SQLiteVecStore(path)
    yield db
    with contextlib.suppress(Exception):
        db.close()
    Path(path).unlink(missing_ok=True)


def _make_embedding(dim: int = 1024) -> list[float]:
    """Return a deterministic dummy embedding list."""
    import random
    rng = random.Random(42)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]
