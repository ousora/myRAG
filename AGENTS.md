# myRAG Pipeline — Agent Instructions

## Project Overview

myRAG is a RAG (Retrieval-Augmented Generation) data pipeline that converts raw documents (.pdf, .docx, .html, .md, .txt) into structured markdown and vector embeddings for semantic search. The pipeline: parse → clean → LLM-format → chunk → embed → sqlite-vec storage.

**Tech stack**: Python 3.10+, markdown-it-py (pure Python markdown chunker), bge-m3 embeddings (remote API or local sentence-transformers), sqlite-vec for local vector storage.

## Key Directories

| Path | Purpose |
|------|---------|
| `src/pipeline/core.py` | Core pipeline: `process_file()`, `process_directory()`, `process_file_hybrid()`, `process_file_with_md()` |
| `src/config.py` | Config loader with resolution chain: env var → config.yaml → config.example.yaml |
| `src/parsers/` | MarkItDown (pdf/docx/md/txt) + Trafilatura (html) dispatcher |
| `src/cleaners/` — removed; canonical impl in `parsers/text_cleaner.py` |
| `src/formatters/` | LLM-powered structuring: `prompts.py`, `writer.py`, async formatting |
| `src/chunkers/` | LangChain MarkdownHeaderTextSplitter wrapper with RecursiveCharacterTextSplitter fallback |
| `src/embedders/bge_m3.py` | bge-m3 embedding client (OpenAI-compatible API) |
| `src/storage/sqlite_vec.py` | SQLiteVecStore with FTS5 full-text search |
| `conf/config.yaml` | User endpoints (gitignored); template at `conf/config.example.yaml` |

## Build & Test Commands

```bash
uv sync --extra dev --extra sqlite-vec    # Install deps
uv run pytest -v                           # Run all tests
uv run ruff check .                        # Lint
uv run mypy src/                           # Type check (if available)
```

Tests live alongside source: `src/chunkers/tests/`, `src/formatters/tests/`, `src/storage/tests/`.

## Architecture Decisions

1. **Two-phase pipeline**: Generate `.md` first (`process_file_with_md`), then ingest to vector DB separately (`_ingest_markdown`). This lets users inspect/edit markdown before embedding.
2. **Hybrid A+B indexing**: Chunk-level fine-grained search (A) + document-level coarse-grained context fallback (B). Both stored in sqlite-vec.
3. **Auto-chunking for large docs**: Texts >28K chars are split at paragraph boundaries; each chunk receives last 10 lines of previous output + cumulative summary as context.
4. **Config resolution chain**: `$MYRAG_CONFIG` → `conf/config.yaml` → `conf/config.example.yaml`. All endpoints configurable via YAML.
5. **Facade pattern removed** — `cleaners/` directory deleted; use `parsers.text_cleaner.TextCleaner` directly.

## Conventions

- **Type hints required** on all public functions (args + return types)
- **No `print()`** in production code — use `logging` module at INFO/WARNING/ERROR levels
- **No hardcoded values** — endpoints, chunk sizes, thresholds via config or function defaults
- **Specific exceptions only** — no bare `except:` clauses
- **Files ≤500 lines** — split modules that grow larger
- **English docs**, Chinese only as supplementary notes for technical concepts
- **Update README.md and CHANGELOG.md** after code changes

## Common Patterns

### Adding a new parser
Register in `src/parsers/dispatcher.py` via `register_parser(ext, ParserClass)`. The dispatcher auto-registers at module load time.

### Adding a config field
Add to `Config.__init__()` in `src/config.py` and update `conf/config.example.yaml`.

### Testing a formatter change
Run `uv run pytest src/formatters/tests/ -v`. Mock `httpx.post` for LLM API calls.

### Ingesting a document to vector DB
```python
from pipeline import _ingest_markdown
_ingest_markdown("output/doc.md", store_path="data/myrag.db")
```

## Testing Workflows

When testing formatter/pipeline changes with real documents:

- **Input**: Place raw test documents in `tmp/doc/` (create if needed)
- **Intermediate output** (LLM JSON responses): Written to `tmp/raw/` — useful for debugging LLM behavior
- **Final markdown**: Output written to `tmp/out/`

Example workflow:
```bash
cp /path/to/document.txt tmp/doc/my-doc.txt
uv run python -c "from formatters import format_md; ... # process tmp/doc/my-doc.txt → tmp/out/"
cat tmp/raw/*.txt   # inspect LLM raw output if needed
cat tmp/out/my-doc.md  # review final markdown
```

Run `git status` to verify no unintended files are committed (tmp/ should be gitignored).

## Pitfalls

- The `cleaners/` module is a facade — canonical implementation is in `parsers/text_cleaner.py` with YAML config support
- `format_text_async()` returns a Future, call `.result(timeout=3600)` to get the dict
- Chunker's `section_path` strips H1 (document title) — sections start at H2 level
- sqlite-vec requires `sqlite-vec` extra: `uv sync --extra sqlite-vec`
