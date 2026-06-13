# Changelog — myRAG Pipeline

## [Unreleased]

### Added (LangChain Chunker + sqlite-vec — 2026-06-14)

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
