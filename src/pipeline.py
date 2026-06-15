"""RAG data cleanup pipeline — parse → clean → format → chunk → embed (Hybrid A+B).

This module re-exports the public API from submodules for backward compatibility.
All imports work as before: `from pipeline import process_file, ...`

For direct CLI usage: `python -m pipeline.cli` or `python src/pipeline.py`.
"""

# Re-export everything from __init__.py for backward compat with code that does
# `import pipeline; pipeline.process_file(...)` (i.e., imports the module itself).
from pipeline import *  # noqa: F401, F403 — re-exports core + ingest public API
