# Changelog — myRAG Pipeline

## [0.5.0] — 2026-06-19

### Fixed

- **Issue 1: conftest.py test collection error** — Removed empty `src/storage/tests/__init__.py` that caused pytest to resolve conftest as `tests.conftest` with missing module. (src/storage/tests/)
- **Issue 3: Chunker adjacent-heading merge bug** — `_split_by_headings()` now creates a new section at every heading boundary, not only when body content exists between headings. Previously consecutive headings without body were merged into one section, giving all sub-chunks wrong `section_path` metadata. (src/chunkers/__init__.py)
- **Issue 4: Formatter module-level config caching** — `_CHUNK_THRESHOLD_CHARS` was evaluated at import time and cached. Replaced with `_get_chunk_threshold()` lazy-evaluated function so config changes take effect on each call. (src/formatters/__init__.py)
- **Issue 5: LLM schema fallback only caught HTTP 500** — Expanded retry to include 503 and 429 status codes for better coverage of schema-incompatible backends. Added HTTP status code to warning log message. (src/formatters/__init__.py)
- **Issue 6: Entity substring false positives** — `_match_entities_to_chunks()` now uses `re.search(r'\b...\b')` word-boundary matching instead of plain `in` substring check. Previously "AI" matched inside "algorithm". (src/pipeline/core.py)
- **Issue 7: hybrid_search rank not normalized** — Replaced naive tuple sort `(fts_rank, vec_score)` with RRF (Reciprocal Rank Fusion) algorithm. FTS5 BM25 rank and cosine distance had vastly different scales, causing FTS to completely dominate. Now both signals contribute fairly via `1/(rank + k)`. (src/storage/sqlite_vec.py)
- **Issue 8: section_filter LIKE query always returned empty** — `json_extract(section_path, '$')` returns full JSON array string `'["General"]'`, so `LIKE '"General"'` did not match. Changed to `LIKE '%General%'` wildcard matching. (src/storage/sqlite_vec.py)
- **Issue 2: Two inconsistent markdown generation paths** — `process_file_hybrid()` now accepts optional `md_output_dir` parameter and delegates to `write_to_md()` for structured markdown output, same as `process_file_with_md()`. Both pipelines now produce identical `.md` content. (src/pipeline/core.py)

### Changed

- **Chunker zero LangChain**: Replaced `langchain-text-splitters` (MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter) with pure Python + `markdown-it-py`. Same output format (`text`, `section_path`, `metadata`), all 8 existing tests pass. Headers parsed via markdown-it-py AST (handles ATX + setext headers natively). Consecutive headings with no body text between them are merged into one section. (src/chunkers/__init__.py, pyproject.toml)
- **Dependency cleanup**: Removed `langchain-text-splitters` + 23 transitive LangChain deps. Added `markdown-it-py>=3.0,<4`. (pyproject.toml)
- **Sentence-aware recursive split**: Custom `_split_by_sentence()` handles Chinese（`。！？`）and English (`.!?`) sentence boundaries. Falls back to character-level split only when a single sentence exceeds `chunk_size`. (src/chunkers/__init__.py)
- **Local bge-m3 embedding**: New `LocalEmbedder` class (`src/embedders/local_bge.py`) using sentence-transformers for offline/CUDA inference. Config `embedding.mode: "remote" | "local"` switches between HTTP API and local model at runtime via `Embedder.__new__()` dispatch. Added `local-embeddings` extra (`uv sync --extra local-embeddings`). Validation integrated into `Config._validate()`. (src/embedders/bge_m3.py, src/embedders/local_bge.py, src/config.py, conf/config.yaml, conf/config.example.yaml)
- **Entity extraction + entity_names**: Formatter prompt now outputs `metadata.entities` (list of `{name, type}` with 5 types: PERSON/ORG/PRODUCT/LOCATION/CONCEPT). JSON Schema enforced via `response_format`. `validate_format_output()` validates entity format. Writer `_insert_wikilinks()` applies `[[wikiname]]` only to .md display files (not chunk text). Pipeline `_match_entities_to_chunks()` matches entities per chunk. sqlite-vec `chunks` table has `entity_names TEXT` column for entity-based retrieval queries. (src/formatters/constants.py, src/formatters/prompts.py, src/formatters/writer.py, src/pipeline/core.py, src/storage/sqlite_vec.py)
- **Schema fallback for incompatible LLM backends**: `call_llm()` now catches 500 errors with PEG-grammar mismatch (`response_format` rejected by llama.cpp), retries without schema automatically. Fixes compatibility with gemma-4 and other models that don't support structured JSON Schema enforcement. (src/formatters/__init__.py)
- **Formatter JSON Schema enforcement**: `call_llm()` now accepts a `schema=` parameter that sends JSON Schema via `response_format`, letting llama.cpp / OpenAI servers enforce output structure natively. Schemas defined in new `constants.py` file. ([src/formatters/constants.py](src/formatters/constants.py), [src/formatters/__init__.py](src/formatters/__init__.py))
- **Formatter JSON parsing robustness**: Multi-level retry — `strict=True` fast path, then `strict=False`, then bare-quote fix for body field (`_fix_bare_quotes_in_body_field()`). Bare quotes inside string values no longer cause parse failures. ([src/formatters/__init__.py](src/formatters/__init__.py))
- **Tag extraction quality**: Removed generic single-word tags ("banking", "company", "system"), introduced proper noun extraction from title + body, multi-word phrase merging, and a whitelist of useful domain-specific terms. Tags now describe document subject matter so a reader can understand what it's about at a glance. ([src/formatters/__init__.py](src/formatters/__init__.py))
- **Few-shot examples added**: Prompts now include concrete input/output examples for both single-shot and chunked formatting, improving output consistency across LLM calls. ([src/formatters/prompts.py](src/formatters/prompts.py))
- **Few-shot examples generalised**: Replaced FX Networks-specific example with a generic research paper example so tags demonstrate domain-agnostic patterns rather than topic-specific values. ([src/formatters/prompts.py](src/formatters/prompts.py))

### Added

- **Output validation helpers**: `validate_format_output(result) → list[str]` checks required fields; `try_fix_common_issues(result)` auto-fixes bad tags / missing metadata without re-calling LLM. ([src/formatters/prompts.py](src/formatters/prompts.py))
- **JSON schema constants**: `FORMATTER_SCHEMA` and `CHUNKED_SCHEMA` extracted from inline prompts into [src/formatters/constants.py](src/formatters/constants.py) for reuse by `call_llm(schema=...)`. ([src/formatters/constants.py](src/formatters/constants.py))

### Changed

- **Parser lazy loading**: Moved MarkItDown/Trafilatura imports from module level into `__init__`. Parsers now load on first use, allowing the module to be imported even when optional deps are missing (fail-fast in `__init__`). ([src/parsers/dispatcher.py](src/parsers/dispatcher.py))
- **TrafilaturaParser encoding**: Now reads HTML files with UTF-8 → GBK fallback instead of passing filepath directly. Fixes silent parse failures on non-UTF-8 encoded Chinese web pages.

## [0.3.0] — 2026-06-17

### Fixed (2026-06-17)

- **hashlib import missing**: Added `import hashlib` to `formatters/__init__.py` (used for LLM response debug logging).
- **total_words = 0 in metadata**: Placeholder value from prompt template was passed through unchanged. Now computed as `len(body.split())`.
- **tags not displayed in markdown output**: Tags are at result level (`result["tags"]`) but writer.py read from `metadata.get("tags")`. Updated `_write_metadata_block()` to accept full result dict and prioritize `result["tags"]`.
- **Placeholder metadata in single-shot mode**: LLM copies template placeholders (created_at: "ISO-8601", total_words: 0). Now overridden with real values in `_format_text_single()`.
- **Split table headers from PDF extraction** → renamed to `_fix_broken_tables()` and rewritten. Uses a more robust approach: detects continuation rows by column count heuristic, appends content into the last cell of the preceding header row to preserve Markdown structure.
- **TextCleaner rewrite**: Major overhaul of `parsers/text_cleaner.TextCleaner`:
  - **Generalized page-break regex** — now matches `"--- PAGE N ---"`, `"=== Section ==="`, etc. (previously only matched single separator characters). Split-based filtering with length safeguard (>8 chars) prevents false positives on short lines like bullet points.
  - **Control character handling** — deleted entirely instead of replacing with space; excludes `\n` and `\t` to avoid breaking text structure.
  - **YAML flags parsing** (`_parse_flags`) now supports `int`, `str`, or `list[str]` (case-insensitive), e.g. `"IGNORECASE"`.
  - **Custom rules pre-compiled** in `__init__` instead of on each `clean()` call — avoids repeated regex compilation cost.
  - **Whitespace collapse** now only trims trailing spaces (`_TRIM_TRAILING_SPACE_RE`) — leading indentation (code blocks, lists) preserved.

### Removed

- **Backward-compat facades**: Deleted `src/cleaners/` directory and its tests. Canonical implementation is now exclusively in `parsers/text_cleaner.TextCleaner`. The convenience function `clean_text()` was removed — use `TextCleaner().clean(text)` directly.
- **Top-level pipeline shim**: Deleted `src/pipeline.py` (re-export wrapper). All imports should come from submodules: `pipeline.core`, `pipeline.ingest`, etc.

### Changed

- **AGENTS.md moved**: Relocated from `.github/` to root directory so agents can always find it.
- **Pipeline module split**: `pipeline.py` (549 lines) → `pipeline/core.py` (356), `pipeline/cli.py` (128), `pipeline/ingest.py` (81). All under 500-line limit.
- **Formatter public API**: `_call_llm()` renamed to `call_llm()`, exported in `__all__`. New `format_text_with_system(raw, source_type, *, system_prompt)` convenience wrapper. Added `system_prompt` parameter chain through `format_text_async()`.

### Fixed

- **sqlite_vec import detection**: Replaced fragile `sys.path` walking with `importlib.metadata.distribution("sqlite-vec").files`. Works across editable installs, wheels, and different Python versions. Added explicit `PackageNotFoundError` handling.

### Added (Auto-Chunking for Large Docs — 2026-06-14)

- **Chunked formatter**: Texts >28K chars auto-split at paragraph boundaries and processed chunk-by-chunk. Each chunk LLM call receives the last 10 lines of previous markdown output + cumulative summary as continuity context. Single-shot path unchanged for small docs.
- **CHUNKED_SYSTEM_PROMPT**: New prompt with concrete input/output example, `Do NOT summarize — preserve ALL substantive content` instruction, markdown style rules, and JSON output schema (`part_md` + `summary`).
- **uv package management**: Replaced pip + requirements.txt with `uv sync`. Single source of truth in `pyproject.toml`. `uv.lock` committed for reproducible installs.

### Changed

- **Package layout**: Flat root → `src/` standard layout. All imports changed from `myrag.xxx` to `xxx`.
- **Dependencies cleaned**: Removed 6 ghost deps (`pymupdf`, `python-docx`, `beautifulsoup4`, `readability-lxml`, `markdown`, `tiktoken`). Added `markitdown[pdf]`, `trafilatura`, `httpx`, `langchain-text-splitters`, `PyYAML`.
- **Config defaults**: `max_tokens` 8192 → 16384, `timeout` 180 → 300.
- **Pipeline timeout**: `future.result()` 300 → 3600 (1h for large doc processing).
- **Test files**: Removed `sys.path.insert` hacks; imports now work via proper package install.

### Fixed

- **Chunk summary bug**: First run produced 20:1 compression (115 lines for 57-page PDF) — missing `DO NOT summarize` in chunked prompt. Fixed + added concrete example → now 619 lines with 8 tables, full glossary, all technical data preserved.
- **str.format KeyError**: JSON curly braces in EXAMPLE block needed `{{`/`}}` escaping.

- **LangChain Chunker**: Replaced custom regex Chunker with `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`. Splits on markdown header boundaries; oversized sections get recursive character split. Plain-text fallback when no headers detected.
- **sqlite-vec storage**: `process_file_hybrid(store_path=...)` now persists embeddings to sqlite-vec database. 17+ chunks, FTS5 full-text index, vector similarity search.
- **Config centralization**: All endpoints (LLM, embedding) in `conf/config.yaml` + `config.py` loader. Resolution chain: `$MYRAG_CONFIG` → `conf/config.yaml` → `conf/config.example.yaml`.
- **Formatter prompt v2**: Added few-shot example, Wikipedia chrome removal rules, CRITICAL body completeness constraint, explicit section level definitions.
- **Chunker unit tests**: 8 tests covering empty input, header splits, hierarchical metadata, oversized sections, plain text fallback, pipeline facade.

### Changed (Post-Audit Cleanup — 2026-06-14)

- **Chunker unified**: Removed duplicate Chunker from `pipeline.py`, canonical implementation in `chunkers/`.
- **writer.py fix**: H1 collision bug, removed hardcoded Chinese document filtering.
- **process_file_hybrid()**: Chunks rendered markdown (from `metadata.sections`), not raw text. Added `store_path` param.
- **pipeline section rendering**: `_render_markdown_with_sections()` generates reliable `##`/`###` headers from LLM metadata.
- **pyproject.toml**: Dependencies updated to `markitdown`, `trafilatura`, `httpx`, `langchain-text-splitters`.

### Fixed

- **sqlite-vec commit bug**: `upsert_chunks` and `upsert_document` now call `conn.commit()` — documents table was always 0 rows.
- **Chunker metadata bug**: Fixed `_metadata_to_section_path` returning `["General"]` for docs without H1 header.

### Removed

- `doc/markitdown-migration-plan.md`, `output/China_National_Clearing_Center.md`, `storage/__init__.py`, all `__pycache__/`, `.pytest_cache/`.

---

## [0.2.0] — MarkItDown Migration — 2026-06-13

### Changed

- **Unified parser backend**: Replaced individual parsers with MarkItDownParser + TrafilaturaParser.
- **TextCleaner reorganized**: Moved to `parsers/text_cleaner.py` with YAML config.
- **Formatter prompt updated**: Removed `chunks` from LLM output, added `body` field.
- **Chunker redesigned**: Auto markdown header parsing, header-enriched embeddings.

## [0.1.0] — 2026-06-13

### Added

- Parser dispatcher, PDF/DOCX/HTML/MD/TXT parsers
- TextCleaner, Chunker, bge-m3 embedder, CLI with argparse
