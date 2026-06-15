# myRAG — TODO

## Completed

- [x] Multi-format parser (PDF/DOCX/HTML/MD/TXT) via MarkItDown + Trafilatura
- [x] TextCleaner with YAML rule support
- [x] LLM structured output (`format_text_async()`)
- [x] Markdown writer (`write_to_md()` / `format_md()`)
- [x] LangChain chunking (header-aware + oversized split + plain-text fallback)
- [x] bge-m3 embedding client (`Embedder`)
- [x] sqlite-vec persistence (chunks + documents + FTS5) via `process_file_hybrid()`
- [x] Centralized config (`conf/config.yaml` + `config.py`)
- [x] End-to-end verification on cncc.txt — 20 chunks, accurate vector retrieval
- [x] Unit tests — 45 passed (chunkers 8, formatters 9, cleaners 5, storage 13, integration 9, sqlite_vec loader 1)
- [x] **Improve sqlite_vec import detection** (`src/storage/sqlite_vec.py`)
  - Replaced fragile `sys.path` walking with `importlib.metadata.distribution("sqlite-vec").files` search
  - Added explicit `PackageNotFoundError` handling with actionable error message
  - Works across editable installs, wheels, and different Python versions

---

## Backlog

### P0 — Critical

- [ ] **RAG query interface** (`rag_query(question, db_path)`)
  - Retrieve → assemble context → call LLM to generate answer

### P1 — Important

- [ ] **Fix Embedder httpx.Client leak**
  - Add `__enter__/__exit__` context manager or explicit `close()` method

- [ ] **Add SQLiteVecStore context manager**
  - Has `close()` but no `__enter__/__exit__` for safe resource handling

- [ ] **Fix `search_documents` missing vector search**
  - Accepts `query_vector` parameter but never uses it in SQL
  - Add `ORDER BY vec_distance_cosine` when vector is provided

- [ ] **Fix ThreadPoolExecutor leak** (`formatters/__init__.py`)
  - Global `_executor` is never shut down
  - Add `shutdown()` method or context manager pattern

- [ ] **CLI search subcommand**
  - `python -m pipeline search "question" --db data/doc.db`

- [ ] **Batch ingest into sqlite-vec**
  - Wire `process_directory()` through storage layer

### P2 — Code Quality

- [x] **Split pipeline.py** (was 549 lines → now split across 3 modules, all under 500)
  - `pipeline/core.py` (356 lines) — core functions: process_file, process_directory, process_file_hybrid, rag_query
  - `pipeline/cli.py` (128 lines) — CLI entry point with main() and argparse subcommands
  - `pipeline/ingest.py` (81 lines) — _ingest_markdown function
  - Reduces duplication between `process_file_hybrid` and `_ingest_markdown`

- [x] **Replace bare except in `process_file_hybrid`**
  - Changed `except Exception` → `except (httpx.HTTPError, RuntimeError)` with typed log message

- [x] **Verify `_format_text_single()` accepts `system_prompt` parameter**
  - Added `system_prompt: str | None = None` to `_format_text_single()`, `_format_text_async_impl()`, and new public `format_text_with_system()` wrapper

- [x] **Fix `summary_text` scope bug** in exception handler
  - Moved title/tags extraction and summary_text construction before the try block so they're always available in the except handler; removed `'summary_text' in dir()` code smell (always True since we initialized it)

- [x] **Make `_call_llm` public or use proper LLM client abstraction** ✅
  - Renamed to `call_llm()`, exported in `__all__`, updated all callers (including `rag_query()`).

- [ ] **Translate Chinese prompt labels** in `CHUNKED_SYSTEM_PROMPT`
  - `【前文收尾】` → `Previous Tail` (or keep Chinese as supplementary note per project rules)
  - `【前文摘要】` → `Previous Summary`
  - `【本段原文】` → `Current Raw Text`

- [ ] **Add config validation** in `Config` class
  - Validate required fields and types (e.g., `temperature` must be float in [0, 1])

- [ ] **Batch ingest via `process_directory()`** — wire through storage layer

### P3 — Nice to Have

- [ ] **Fix hardcoded path in `format_md()`** (`/tmp/md_format_output`)
  - Accept configurable `output_dir` parameter

- [ ] **Cache parser instances** in `resolve_parser()`
  - Avoid re-initializing MarkItDown converter on every call

- [ ] **Add mypy configuration** to `pyproject.toml` per project rules

- [ ] **Add `__main__.py`** for cleaner `python -m src` invocation

- [ ] **Deduplicate / update strategy** for repeated ingest with same `doc_id`

---

## Review Summary (2026-06-16)

| Category | Count |
|----------|-------|
| P0 — Critical | 4 |
| P1 — Important | 6 |
| P2 — Code Quality | 6 |
| P3 — Nice to Have | 5 |
| **Total** | **21** |

**Assessment**: Solid RAG pipeline with clean architecture. Primary concerns are the embedding serialization bug in SQLiteVecStore, missing storage tests, and resource leaks (httpx.Client, ThreadPoolExecutor). The Query interface is the next major milestone.
