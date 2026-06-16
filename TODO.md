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
- [x] Unit tests — 48 passed (config 11, chunkers 8, formatters 9, cleaners 5, embedders 7, parsers 13)
- [x] **Improve sqlite_vec import detection** — replaced fragile sys.path walking with importlib.metadata-based detection
- [x] **Split pipeline.py** — split 549-line file into core.py, cli.py, ingest.py (all under 500 lines)
- [x] **Replace bare except in `process_file_hybrid`** — changed to specific httpx.HTTPError and RuntimeError
- [x] **Add system_prompt param to format functions** — added optional system_prompt to _format_text_single, _format_text_async_impl, and format_text_with_system
- [x] **Fix summary_text scope bug** — moved extraction before try block, removed dir() check
- [x] **Make _call_llm public** — renamed to call_llm(), added to __all__, updated all callers

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

- [x] **Translate Chinese prompt labels** in `CHUNKED_SYSTEM_PROMPT`
  - Changed to English: `[Previous Context]`, `[Summary of Previous Chunks]`, `[Current Chunk Text]`

- [ ] **Add config validation** in `Config` class
  - Validate required fields and types (e.g., `temperature` must be float in [0, 1])
  - ✅ Partially done: `_validate()` method added for timeout/size constraints

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
| P2 — Code Quality | 2 |
| P3 — Nice to Have | 5 |
| **Total** | **17** |

**Assessment**: Solid RAG pipeline with clean architecture. Primary concerns are the embedding serialization bug in SQLiteVecStore, missing storage tests, and resource leaks (httpx.Client, ThreadPoolExecutor). The Query interface is the next major milestone.
