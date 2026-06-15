# Changelog — myRAG Pipeline

## [Unreleased]

### Changed

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
