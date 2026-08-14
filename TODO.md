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
- [x] Unit tests — 185 passed
- [x] **Fixed `response` undefined on schema fallback** in `formatters/__init__.py`
- [x] **Fixed `chunk_size` default mismatch** — unified to 1024 across all pipeline entry points
- [x] **Fixed config default values** — `llm_max_tokens` 8192→16384, `llm_timeout` 180→300
- [x] **Fixed invalid mypy config** — removed `warn_unused_comments`, fixed `exclude` regex
- [x] **Fixed duplicate chunk/doc construction** in `pipeline/core.py`
- [x] **Fixed quoted type annotations** in `rag_query()` signature
- [x] **Cleaned up unused imports** across 8 files
- [x] **Unified `Optional[X]` → `X | None`** syntax in storage modules
- [x] **Improve sqlite_vec import detection** — replaced fragile sys.path walking with importlib.metadata-based detection
- [x] **Split pipeline.py** — split 549-line file into core.py, cli.py, ingest.py (all under 500 lines)
- [x] **Replace bare except in `process_file_hybrid`** — changed to specific httpx.HTTPError and RuntimeError
- [x] **Add system_prompt param to format functions** — added optional system_prompt to _format_text_single, _format_text_async_impl, and format_text_with_system
- [x] **Fix summary_text scope bug** — moved extraction before try block, removed dir() check
- [x] **Make _call_llm public** — renamed to call_llm(), added to __all__, updated all callers
- [x] **Translate Chinese prompt labels in CHUNKED_SYSTEM_PROMPT** — changed to English: `[Previous Context]`, `[Summary of Previous Chunks]`, `[Current Chunk Text]`
- [x] **Fix hashlib import missing in formatters/__init__.py** — added `import hashlib` (used for LLM response debug logging)
- [x] **Fix total_words = 0 in metadata** — placeholder value from prompt template was passed through; now computed as `len(body.split())`
- [x] **Fix tags not displayed in markdown output** — tags are at result level (`result["tags"]`) but writer.py read from `metadata.get("tags")`; updated `_write_metadata_block()` to accept full result dict
- [x] **Fix placeholder metadata in single-shot mode** — LLM copies template placeholders (created_at: "ISO-8601", total_words: 0); now overridden with real values in `_format_text_single()`
- [x] **Fix split table headers from PDF extraction** — added `_merge_table_continuation_lines()` in TextCleaner; detects continuation rows by column count heuristic and merges them
- [x] **H2: Fix missing `hybrid` subcommand in CLI** — added `subparsers.add_parser("hybrid")` registration
- [x] **M6: Fix config.yaml inconsistency** — added missing `embedding.mode`, `query_instruction`, `debug_log_llm_responses`
- [x] **M7: Fix duplicate `[tool.pytest.ini_options]` in pyproject.toml** — merged into single section
- [x] **M9: Add `types-PyYAML` and `mypy` to dev extras** — local mypy now works
- [x] **H1: Fix `ingest.py` resource leak** — wrapped `Embedder` and `SQLiteVecStore` in `with` statements
- [x] **H3: Fix `search_documents` not using `query_vector`** — added `ORDER BY vec_distance_cosine`
- [x] **H5: Add `__enter__`/`__exit__` to `Embedder` and `SQLiteVecStore`** — both support `with` statements
- [x] **M4: Fix duplicate CJK regex compilation in `_build_fts_query`** — pre-compiled once at module level
- [x] **M5: Fix `_count_words` character-by-character iteration** — replaced with `re.sub` + `re.findall`
- [x] **M11: Fix case-sensitive title matching in `markdown_utils.py`** — added `re.IGNORECASE`
- [x] **M12: Added return type `dict[str, Any]` to `rag_query`**
- [x] **M13: Fixed `LocalEmbedder.embed`/`embed_query` type annotations** — return types now match actual behavior
- [x] **M14: Fixed shallow copy in `try_fix_common_issues`** — replaced with `copy.deepcopy`
- [x] **M15: Added `mypy` step to CI workflow**
- [x] **M8: Removed outdated `<3` upper bound on `sentence-transformers`**
- [x] **M1: Extracted `_call_llm_api` shared function** — DRY violation fixed
- [x] **M2: Pre-compiled `_PARAGRAPH_SPLIT` regex** — moved to module level
- [x] **M3: Made `upsert_document` atomic** — replaced SELECT+INSERT/UPDATE with `INSERT ... ON CONFLICT`
- [x] **M10: Added file check for `logs/` directory in CLI**
- [x] **L1: Added `[project.scripts]` entry point** — `myrag` command available after install
- [x] **L2: Added `*.db`, `*.sqlite3`, `*.sqlite` to `.gitignore`**
- [x] **L3: Fixed cache key to hash components separately** — avoids system prompt dominating hash
- [x] **L5: Fixed httpx timeout exception handling** — catches `ReadTimeout`/`WriteTimeout` for httpx 1.0+
- [x] **L7: Pre-compiled `MarkdownIt` in `Chunker.__init__`** — reused across `chunk()` calls
- [x] **L8: Added `batch_size` parameter to `store_chunks`** — splits large batches to avoid API limits
- [x] **L9: Added 1000-item cap on `IN` clause placeholders** — prevents SQLite variable number overflow
- [x] **L10: Added `UNIQUE(source_file)` constraint on `documents` table** — required for ON CONFLICT
- [x] **Code audit: exception f-string literals** — EM101/EM102 fixed in cli.py, ingest.py, writer.py
- [x] **Code audit: zip() strict mode** — B905 fixed in core.py, rerank.py, test_sqlite_vec.py
- [x] **Code audit: RET504 unnecessary assignments** — fixed in core.py, rerank.py, text_cleaner.py
- [x] **Code audit: W293 trailing whitespace** — fixed in core.py docstrings
- [x] **Code audit: D401 imperative mood** — fixed in text_cleaner.py, rerank.py
- [x] **Code audit: N806 variable naming** — renamed _MAX_IN_CLAUSE to max_in_clause
- [x] **Code audit: D102 missing docstrings** — added to core.py facade methods
- [x] **Code audit: SIM110** — simplified _is_inside_protected() in writer.py
- [x] **Code audit: TRY300** — restructured _load_sqlite_vec() in schema.py
- [x] **Code audit: D301 raw docstrings** — added r prefix to text_cleaner.py
- [x] **Code audit: py.typed marker** — added src/py.typed for mypy import resolution

---

## Backlog

### P0 — Critical

- [ ] **RAG query interface** (`rag_query(question, db_path)`)
  - Retrieve → assemble context → call LLM to generate answer
  - ✅ Already implemented in `pipeline.core.rag_query()` (was already done)

### P1 — Important

- [x] **Fix Embedder httpx.Client leak**
  - Added `__enter__/__exit__` context manager and `close()` method

- [x] **Add SQLiteVecStore context manager**
  - Added `__enter__/__exit__` for safe resource handling

- [x] **Fix `search_documents` missing vector search**
  - Added `ORDER BY vec_distance_cosine` when vector is provided

- [ ] **Fix ThreadPoolExecutor leak** (`formatters/__init__.py`)
  - Global `_executor` is never shut down
  - Add `shutdown()` method or context manager pattern

- [ ] **CLI search subcommand**
  - `python -m pipeline search "question" --db data/doc.db`

- [ ] **Batch ingest into sqlite-vec**
  - Wire `process_directory()` through storage layer

### P2 — Code Quality

- [x] **Add config validation** in `Config` class
  - Validate required fields and types (e.g., `temperature` must be float in [0, 1])
  - ✅ Partially done: `_validate()` method added for timeout/size constraints
  - ✅ Added missing fields to `config.yaml`

### P3 — Nice to Have

- [ ] **Fix hardcoded path in `format_md()`** (`/tmp/md_format_output`)
  - Accept configurable `output_dir` parameter

- [ ] **Cache parser instances** in `resolve_parser()`
  - Avoid re-initializing MarkItDown converter on every call

- [x] **Add mypy configuration** to `pyproject.toml` per project rules
  - Added `mypy>=1.0` and `types-PyYAML` to dev extras

- [ ] **Add `__main__.py`** for cleaner `python -m src` invocation

- [ ] **Deduplicate / update strategy** for repeated ingest with same `doc_id`
  - ✅ Fixed: `documents` table now has `UNIQUE(source_file)` with `ON CONFLICT` upsert

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
