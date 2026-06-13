# Project: myrag-pipeline — RAG Data Cleanup Pipeline

## Tech Stack
- Python ≥3.10
- PyMuPDF (pdf), python-docx (docx), BeautifulSoup + readability-lxml (html), markdown, PaddleOCR (optional)
- bge-m3 embedding service (local deployment via llama.cpp / vLLM)
- httpx for API calls

## Architecture
```text
parsers/          — File-type-specific document parsers
cleaners/         — TextCleaner: denoising, header/footer cleanup, Unicode normalization
chunkers/         — Chunker: configurable size + overlap segmentation
embedders/        — Embedder: bge-m3 embedding client (OpenAI-compatible API)
pipeline.py       — process_file() / process_directory() entry point
```

## Key Files
- `parsers/dispatcher.py` — PARSERS registry + resolve_parser(), triggered via `import myrag.parsers`
- `parsers/__init__.py` — Loads all parsers using `importlib.import_module()` with try/except for missing dependencies
- `pipeline.py` — process_file(filepath, max_chars=512) → list[dict]; process_directory() → list[dict]

## Conventions
- Parser classes use class methods (no decorators — registered after class definition in each file)
- Each parser module handles ImportError gracefully (pass + skipped by __init__ loop)
- Extension aliases: markdown→md/mkd, html→htm
- Chunker min_chunk_chars defaults to 3 (not 30 — avoid filtering out small chunks from short docs)

## Testing
```bash
cd myrag && python -m pytest tests/
# or manually test with process_file("path/to/file")
```

## Rules
See `.github/copilot-instructions.md` for coding standards and project rules.