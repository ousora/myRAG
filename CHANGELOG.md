# Changelog — myRAG Pipeline

## [Unreleased]

### Changed (MarkItDown Migration)

- **Unified parser backend** (`parsers/dispatcher.py`): replaced individual parsers with MarkItDownParser (pdf, docx, md, txt) and TrafilaturaParser (html). Registration via module-level `register()` calls instead of importlib.
- **TextCleaner reorganized**: moved from inline class in `pipeline.py` + legacy `cleaners/` to dedicated `parsers/text_cleaner.py` (~120 lines) with YAML config support (`rules_config="custom.yaml"`). Old `cleaners/__init__.py` kept as backward-compat facade.
- **Formatter prompt updated**: removed chunks section from LLM output; added `body` field (raw cleaned text for downstream chunking). Fixed `.format()` brace conflicts in JSON examples using double-brace escaping (`{{"level": 2}}`).
- **Chunker redesigned** (`chunkers/__init__.py`): `chunk(text)` no longer accepts external `section_path`. Now auto-parses markdown headers (##, ###) via `_parse_sections()` regex and assigns semantic context per chunk. Falls back to ["General"] when no sections found.
- **Writer simplified**: body written directly from LLM response — MarkItDown already produces properly formatted markdown with hierarchical headers. Removed duplicate header re-parsing logic.
- **LLM output robustness** (`formatters/__init__.py`): added `re.search(r'\{.*\}', content, re.DOTALL)` to extract first valid JSON object from LLM responses that may include extra text (e.g., thinking process leakage).

### Added

- `[tool.setuptools.packages.find] include = ["myrag*"]` in pyproject.toml for flat-layout editable install support.

### Removed

- Old parser files: `pdf.py`, `docx.py`, `html.py`, `md_parser.py`, `txt.py`.
- Chunks section from formatter prompt and test assertions (chunking is now a downstream concern).

## [0.2.0] — 2026-06-13

### Added
- **Formatters module** (`formatters/`): LLM-powered text formatting for raw copied content
  - `format_text()` / `format_text_async()`: Clean and structure raw web/text into title, tags, metadata via local Qwen model
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
