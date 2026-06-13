# Changelog — myrag-pipeline

## [Unreleased]

### Changed
- **Cleaners module refactored** (`cleaners/`):
    - Added control character removal (e.g., `\x07` BELL chars from PDFs) via `re.sub(r"[\x00-\x1f\x7f]", " ", text)`
    - Improved page break regex to handle converted whitespace patterns: `-=_*\s+PAGE N\s+-=_*`
    - `clean_text()` now accepts optional params (`remove_page_breaks`, `collapse_whitespace`) instead of using defaults
- **Formatter prompts updated** (`formatters/prompts.py`):
    - `sections`: changed from flat list `["Section 1"]` to structured objects `[{"level": 2, "title": "..."}]` with heading hierarchy levels
    - `chunks.section`: renamed to `section_path: ["Introduction"]` (array representing full hierarchical path)
- **Formatter writer enhanced** (`formatters/writer.py`):
    - Added `_render_section_path()` helper — maps section depth to markdown header level (H2–H6)
    - Sections list now renders with indentation based on `level` field
    - Output metadata block includes `**Total words:** N | **Chunks:** M` summary line
- **pipeline.py docstring**: clarified full Scheme C pipeline flow (parser → cleaner → formatter → chunker/embedder)

### Added
- **Tests: cleaners module** (`cleaners/tests/test_cleaner.py`) — unit tests for control char removal, page break detection, whitespace normalization, and empty input handling

## [0.2.0] — 2026-06-13

### Added
- **Formatters module** (`formatters/`): LLM-powered text formatting for raw copied content
  - `format_text()` / `format_text_async()`: Clean and structure raw web/text into title, tags, metadata, chunks via local Qwen model
  - `write_to_md()` / `format_md()`: Write structured results to markdown files or return as string
  - `prompts.py`: Configurable system prompts for different source types (`web`, `markdown`, `pdf_clip`)
- **Tests**: Unit tests for formatters module (mocked httpx responses)

### Changed
- **.gitignore**: Added `.doc/` and `output/` to ignore user data directories

## [0.1.0] — 2026-06-13

### Added
- **Parser dispatcher** (`parsers/dispatcher.py`): `PARSERS` registry + `resolve_parser()` with extension aliases (md→markdown, htm→html)
- **Parsers**: PDF (PyMuPDF), DOCX (python-docx), HTML (BeautifulSoup + readability-lxml fallback), Markdown (python-markdown), TXT (multi-encoding fallback: utf-8/gbk/gb2312/latin-1)
- **TextCleaner** (`cleaners/__init__.py`): page-break removal, whitespace collapse, line normalization
- **Chunker** (`chunkers/__init__.py`): configurable max_chars / min_chunk_chars (default 3) / overlap_chars
- **Embedder** (`embedders/bm25.py`): bge-m3 client with OpenAI-compatible `/v1/embeddings` API
- **Pipeline entry points**: `process_file()` and `process_directory()` in `pipeline.py`

### Fixed
- Parser registration: moved from decorators (which can't reference the class being defined) to post-class registration calls
- Extension matching: added fallback for aliases like .md → markdown, .htm → html
- Chunker min_chunk_chars lowered from 30→3 to handle short documents without filtering out chunks

### Notes
- Parser modules use `importlib.import_module()` in `__init__.py` with try/except for missing dependencies (graceful skip)
- bge-m3 service base_url defaults to http://127.0.0.1:8000; configure via Embedder(base_url=...)
