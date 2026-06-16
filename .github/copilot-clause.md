# Project: myrag-pipeline — RAG Data Cleanup Pipeline (Scheme C)

## Tech Stack
- Python ≥3.10
- MarkItDown (PDF, DOCX, MD, TXT parsing → clean markdown)
- Trafilatura (HTML page body extraction with Chinese NLP support)
- Qwen3.6-35B-A3B-MoE via llama.cpp at 192.168.191.112:8081 (`--ctx-size 131072`)
- bge-m3 embedding service (local deployment)
- httpx for API calls

## Architecture
```text
parsers/          — MarkItDown / Trafilatura unified parser + TextCleaner
cleaners/         — REMOVED (was backward-compat facade)
chunkers/         — Chunker: configurable size + overlap segmentation
embedders/        — Embedder: bge-m3 embedding client (OpenAI-compatible API)
formatters/       — LLM formatter with clean section_path constraints
pipeline/core.py  — process_file(), process_directory(), process_file_hybrid()
pipeline/ingest.py — _ingest_markdown() for existing .md ingestion
```

## Key Files
- `parsers/dispatcher.py` — resolve_parser() routing: html→Trafilatura, pdf/docx/md/txt→MarkItDown
- `parsers/text_cleaner.py` — TextCleaner with configurable regex rules (YAML config) for noise removal
- `formatters/prompts.py` — System prompt with clean section_path constraints (no article numbers/anchors in text)
- `formatters/writer.py` — write_to_md(): renders chunks with hierarchical H1/H2/H3 headers from section_path
- `pipeline.py` — process_file_with_md(filepath, output_dir) → path to .md file

## Conventions
- Parser routing via dispatcher: html/htm→TrafilaturaParser, pdf/docx/markdown/txt→MarkItDownParser
- TextCleaner removes anchor markers ([导图], [小结]) and article numbers (第一篇, 第二篇) before LLM
- Formatter output contains clean section_path arrays without "第X篇" prefixes — writer.py renders these as ##/### headers
- Chunk size defaults to 512 chars; formatter handles semantic segmentation

## Testing
```bash
cd myrag && python -m pytest tests/
# or manually test with process_file_with_md("path/to/file")
```

## Rules
See `.github/copilot-instructions.md` for coding standards and project rules.
