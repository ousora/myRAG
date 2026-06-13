# myRAG — TODO (Persistent Task Tracker)

## ✅ Completed

### Parsers & Cleaners
- [x] PDF parser (PyMuPDF) — ISO 20022 document verified
- [x] DOCX, HTML, Markdown, TXT parsers with dispatcher + decorator registry
- [x] TextCleaner: control chars (\x07), page breaks, whitespace collapse
- [x] Cleaner tests (5 passed)

### Formatter & Writer
- [x] LLM formatter (`format_text_async`) — max_tokens=8192, timeout=180s
- [x] System prompt with nested `section_path` arrays + metadata sections with level
- [x] Markdown writer with `_render_section_path()` auto H2-H5 indentation
- [x] Formatter tests (10 passed)

### Pipeline & Docs
- [x] `pipeline.py` — Scheme C documented: parser → cleaner → formatter → chunker → embedder
- [x] README.md — architecture + directory structure + usage examples
- [x] CHANGELOG.md, LICENSE (MIT), .gitignore
- [x] `.github/copilot-clause.md`, `.github/copilot-instructions.md`
- [x] `.test/` directory with `scripts/run_pdf_test.py` and `output/`

### Git Setup
- [x] Initialized git, pushed to github.com/ousora/myRAG.git (main)

## 🚧 Pending — Tests (High Priority)
- [ ] **parsers/tests** — unit tests for each parser (rules.md requirement)
- [ ] **chunkers/tests** — chunker logic validation
- [ ] **embedders/tests** — bge-m3 client mock test
- [ ] **pipeline integration test** — end-to-end pipeline verification

## 🚧 Pending — Core Features
- [ ] Configure embedder `base_url` (bge-m3 endpoint)
- [ ] Enable PaddleOCR support in pdf.py (pending dependency install)
- [ ] Vector DB integration: embedders → FAISS/Milvus store

## 🚧 Pending — Configuration
- [ ] `.editorconfig` for consistent indentation/line endings

---
*Created 2026-06-13. Updated after formatter + cleaner fixes.*
