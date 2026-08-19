# myRAG — RAG Pipeline

```
.doc/file → parse → clean → format ─┬→ write_to_md() → readable .md (with [[wikilinks]])
                                    └→ chunk → match entities → embed → sqlite-vec
```

> Large texts (>20K chars) are auto-split at paragraph boundaries and processed chunk-by-chunk. Each chunk receives the last 10 lines of previous markdown output + cumulative summary as context for continuity.

## Architecture

```text
Raw file (.pdf/.docx/.html/.md/.txt)
    ↓ parser.parse()              # MarkItDown / Trafilatura → text
    ↓ cleaner.clean()             # TextCleaner: noise removal
    ↓ formatter.format_text()     # LLM → {title, tags, sections, entities, body}
    │                             # Small docs: single-shot. Large docs: auto-chunked
    │
    ├→ write_to_md(result)        # .md file with [[Entity]] wikilinks
    │                              # (entities extracted by LLM, matched to text)
    │
    └→ pipeline.hybrid functions (LLM-powered):
        - process_file_hybrid() — parse → LLM format → chunk → embed (Hybrid A+B)
        - process_file_with_md() — parse → LLM format → write structured .md
        - process_directory_hybrid() — batch directory processing with concurrency
        - rag_query() — retrieve chunks + generate LLM answer

    Internal pipeline steps:
        ↓ chunker.chunk(body)     # markdown-it-py (pure Python, no LangChain)
        ↓ _match_entities()       # tag chunks with entity_names from text match
        ↓ embedder.store_chunks() # bge-m3 → 1024-d (remote API or local CPU)
        ↓ SQLiteVecStore          # sqlite-vec: chunks + entity_names + FTS5
```

## Pipeline Components

### 1. Parser (`src/parsers/`)

**MarkItDown** (pdf, docx, md, txt) + **Trafilatura** (html). Single `resolve_parser()` dispatcher with lazy import — parsers load on first use (fail-fast in `__init__`). TrafilaturaParser handles HTML encoding with UTF-8 → GBK fallback.

Missing dependencies raise `ParserNotFoundError` from [`myrag.exceptions`](src/myrag/exceptions.py) for structured error handling:

```python
from myrag.exceptions import ParserNotFoundError

try:
    parser = resolve_parser("report.pdf")
except ParserNotFoundError as e:
    print(f"Parser not available: {e}")
```

### 2. TextCleaner (`src/parsers/text_cleaner.py`)

Control chars, page breaks, whitespace. Optional YAML config for custom regex rules.

```python
from parsers.text_cleaner import TextCleaner
cleaned = TextCleaner().clean(raw_text)
```

### 3. Formatter (`src/formatters/`)

LLM-powered: extracts title, tags, section hierarchy. **Auto-chunks large texts** (>20K chars) at paragraph boundaries — each chunk gets the last 10 lines of previous markdown output + cumulative summary as context for continuity.

**Offline mode** (`use_llm=False`): skips the LLM formatter entirely and writes the parser's raw cleaned output as the .md body (title from the first `# ` heading, or "Untitled"). A deterministic normalizer ([`src/parsers/markdown_normalizer.py`](src/parsers/markdown_normalizer.py)) runs first to structure the output: promotes standalone short lines to `##` headings, normalizes list markers, formats bare URLs as `[url](url)`, repairs unclosed bold/italic, and aligns table columns. No model call, no network — deterministic. CLI: `myrag md input.pdf --no-llm`.

**JSON Schema enforcement**: `call_llm()` accepts a `schema=` parameter to send JSON Schema via `response_format`, letting llama.cpp / OpenAI servers enforce output structure natively (schemas in [constants.py](src/formatters/constants.py)).

**Tag quality**: Extracts proper nouns and domain-specific multi-word phrases; filters generic single words ("banking", "system").

**Output validation**: `validate_format_output()` and `try_fix_common_issues()` from `formatters.prompts` for post-processing without re-calling LLM. Import as: `from formatters.prompts import validate_format_output, try_fix_common_issues`.

```python
from formatters import format_text_async, format_text_with_system, call_llm, write_to_md

# Standard formatting (async)
future = format_text_async(cleaned, source_type="pdf")
result = future.result(timeout=3600)

# Custom system prompt
result = format_text_with_system(cleaned, source_type="pdf", system_prompt=custom_prompt)

md_path = write_to_md(result, "output/")    # readable markdown
```

### 4. Chunker (`src/chunkers/`)

Pure Python markdown splitting via `markdown-it-py` (no LangChain dependency). Splits on `##`/`###` boundaries with hierarchical metadata tracking. **Each heading updates the section metadata context** — consecutive headings without body text share one text section but each gets its own metadata. Oversized sections get recursive character split with sentence-aware boundaries (Chinese `。！？` + English `.!?`). Plain text without headers auto-detected.

```python
from chunkers import Chunker
chunks = Chunker(chunk_size=512, chunk_overlap=64).chunk(markdown_text)
# Each chunk: {"text": "...", "section_path": ["Services", "HVPS"], "metadata": {...}}
```

**Input validation**: `Chunker(chunk_size=-1)` raises `ValueError`. **Character-level fallback**: when sentence splitting still produces oversized chunks (e.g., URLs, base64 blobs), a `_split_by_char()` greedy splitter with overlap kicks in.

### 5. Embedder + Storage

bge-m3 embeddings → sqlite-vec database with FTS5 full-text index + entity_names column.

**Dual embedding mode** — set `embedding.mode` in config to switch:
- `"remote"` (default): calls HTTP API at `embedding.base_url` (vLLM / Ollama compatible)
- `"local"`: uses sentence-transformers (`uv sync --extra local-embeddings`), CPU inference, no network dependency

**Hash fallback** (`embedding.hash_fallback: true`): when the endpoint is unreachable, embedders fall back to a deterministic SHA-256 → 1024-d vector. Same text → same vector, so retrieval is stable across runs, but vectors carry no semantic signal. Offline/dev only — not a replacement for a real model.

**Dimension validation**: Both remote and local backends validate embedding dimension on every call. Mismatched dimensions raise `EmbeddingError` with context about expected vs actual size.

**HTTP retry + resource cleanup**: Remote embedder retries transient failures (429/502/503/504) with exponential backoff. Use as a context manager for deterministic connection cleanup: `with Embedder() as e: ...`.

**CJK-aware token estimation**: Local embedder uses multi-language heuristic (`len//2` → character-class-aware counting) so batch sizing stays accurate for Chinese/Japanese/Korean text that SentencePiece tokenizes at ~1 char/token.

**Entity search** — `entity_names` column stores entity mentions per chunk for cross-doc entity lookup:

```python
# Query by entity name (uses wildcard LIKE matching on JSON array)
db.conn.execute(
    "SELECT text FROM chunks WHERE entity_names LIKE ?",
    ['%"GPT-4"%']
).fetchall()

# Build + query
from embedders import Embedder, create_embedder  # Factory function respects config.mode
from storage.sqlite_vec import SQLiteVecStore

db = SQLiteVecStore("data/myrag.db")
e = create_embedder()  # or: from myrag.exceptions import EmbeddingError
hits = db.search_chunks(e.embed("your question"), k=5)
```

**Hybrid search** — `search_chunks()` performs vector similarity search; for combined vector + FTS5 full-text use `hybrid_search()`, with results fused using Reciprocal Rank Fusion (RRF) for fair ranking of both signals. **Section filter** uses exact JSON array element matching via `json_each`: `db.search_chunks(..., section_filter=["Services"])` matches chunks whose `section_path` contains the literal string "Services" (no wildcard false-positives). **Empty query fallback**: when `query_text=""` but `query_vector` is provided, returns pure vector results (avoids FTS5 MATCH '' syntax error).

## Quick Start

### Install

```bash
# Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev --extra sqlite-vec
# For local bge-m3 embedding (CPU inference, offline): uv sync --extra local-embeddings
cp conf/config.example.yaml conf/config.yaml
# Edit conf/config.yaml with your endpoints
```

### CLI

```bash
# 1. Generate .md only (inspect or edit before storage)
python -m src md input.pdf --output-dir output/

# 1b. Generate .md WITHOUT calling the LLM (offline / parser output only)
python -m src md input.pdf --output-dir output/ --no-llm

# 2. Ingest an existing .md into sqlite-vec (no LLM call)
python -m src ingest output/doc.md --store data/doc.db

# 3. Generate .md and auto-ingest (two-step, transparent)
python -m src process input.pdf --store data/doc.db

# 3b. Same, but skip the LLM formatter
python -m src process input.pdf --store data/doc.db --no-llm

# Hybrid mode: generate .md + ingest in one command with LLM formatting
python -m src hybrid input.pdf --store data/doc.db

# Traditional CLI (no LLM, no sqlite-vec)
python -m src process-file input.txt --chunk-size 512
```

## Directory Structure

```text
myrag/
├── src/
│   ├── __init__.py           # Package init
│   ├── config.py             # Config loader: get_config()
│   ├── pipeline/             # Pipeline modules (split to stay under 500 lines)
│   │   ├── __init__.py       # Package init
│   │   ├── core.py           # Core functions: process_file, process_directory, process_file_hybrid, rag_query
│   │   ├── cli.py            # CLI entry point with argparse subcommands
│   │   ├── ingest.py         # _ingest_markdown function
│   │   ├── markdown_utils.py # Markdown rendering + reference section stripping
│   │   └── utils.py          # Parser resolution + doc summary helpers
│   ├── parsers/              # MarkItDown + Trafilatura dispatcher
│   │   ├── dispatcher.py
│   │   └── text_cleaner.py
│   ├── formatters/           # LLM formatter + prompts + markdown writer + wikilinks
│   │   ├── __init__.py
│   │   ├── constants.py      # JSON schemas for response_format (incl. entities)
│   │   ├── prompts.py
│   │   ├── writer.py
│   │   ├── tags.py           # Tag extraction from body content
│   │   └── cache.py          # Process-wide LRU formatting cache
│   ├── chunkers/             # Pure Python markdown-it-py chunker (no LangChain)
│   ├── embedders/            # bge-m3: remote HTTP API + local sentence-transformers
│   │   ├── __init__.py
│   │   ├── bge_m3.py         # Unified Embedder with mode dispatch + dimension validation
│   │   └── local_bge.py      # LocalEmbedder via sentence-transformers
│   ├── myrag/                # Shared utilities
│   │   ├── __init__.py       # Package init + exception re-exports
│   │   └── exceptions.py     # Typed exception hierarchy (ParserNotFoundError, EmbeddingError, etc.)
│   └── storage/              # SQLiteVecStore with FTS5 full-text search
│       ├── __init__.py
│       ├── sqlite_vec.py     # Main store class (80 lines)
│       ├── schema.py         # Table creation + schema definitions (94 lines)
│       ├── inserts.py        # Upsert operations (234 lines)
│       └── search.py         # Search + hybrid RRF (313 lines)
├── conf/
│   ├── config.yaml           # Your endpoints (gitignored)
│   └── config.example.yaml   # Template (committed)
├── output/                   # Generated markdown files
├── data/                     # sqlite-vec databases (gitignored)
├── logs/                     # Pipeline logs (gitignored)
├── pyproject.toml
├── uv.lock
└── README.md
```

## Configuration

Single file: `conf/config.yaml` (gitignored). Template at `conf/config.example.yaml`.

```yaml
llm:
  endpoint: "http://your-llm:8081/v1/chat/completions"
  model: "your-model-name"
  temperature: 0.3                # Default; set to 0.0 for deterministic entity extraction
  max_tokens: 16384
  timeout: 300

embedding:
  mode: "remote"                  # "remote" (HTTP API) or "local" (sentence-transformers)

  # When mode == "remote":
  base_url: "http://your-embedder:11435"
  model: "bge-m3"

  # When mode == "local":
  # local_model: "BAAI/bge-m3"    # Auto-downloaded on first use (~1.1GB)
```

Resolution chain: `$MYRAG_CONFIG` → `conf/config.yaml` → `conf/config.example.yaml`.

```python
from config import get_config
cfg = get_config()
print(cfg.llm_endpoint)  # from your config file

# Hot-reload config after modifying conf/config.yaml without restart:
# get_config(reset=True)  # clears cache and reloads
```

## Testing

```bash
cd myrag
uv run pytest -v
# 185 tests: chunkers 12 + formatters 35 + storage 36 + integration 9 + config 9 + parsers 12 + embedders 11 + test_formatter 13 + pipeline 16 + test_rerank 10 + test_directory_hybrid + test_rag_query + test_local_embedder
```

### Linting

```bash
uv run ruff check .
uv run ruff check . --fix   # auto-fix most issues
```

### Type Checking

```bash
uv run mypy src/       # ~355 errors — internal module typing incomplete; add src/py.typed marker
```

## Testing

```bash
cd myrag
uv run pytest -v
# 253 tests: chunkers 12 + formatters 46 + storage 46 + integration 9 + config 9 + parsers 24 + embedders 29 + test_formatter 13 + pipeline 36 + test_rerank 20 + test_directory_hybrid + test_rag_query + test_local_embedder + test_text_cleaner 24 + test_tags 17 + test_cache 11 + test_ingest 5 + test_utils 16
```

`get_config()` validates required fields (LLM endpoint, model name) on every call. Invalid configs raise `ValueError` with descriptive messages before any pipeline work begins. Debug logging of LLM responses controlled by `debug_log_llm_responses: true` in config (gated by `logging.debug`).

## Architecture Notes

- **Config resolution chain**: `$MYRAG_CONFIG` → `conf/config.yaml` → `conf/config.example.yaml`. All endpoints configurable via YAML.
- **Facade pattern** — `TextCleaner` and `Chunker` classes in `pipeline.core` are thin facades that delegate to `parsers.text_cleaner.TextCleaner` and `chunkers.Chunker` respectively. The canonical implementations live in their own modules with full feature support (YAML config, markdown-it-py chunking).
- **Module splitting**: Large files (>500 lines) split into focused modules: `pipeline/markdown_utils.py`, `pipeline/utils.py`, `formatters/tags.py`, `formatters/cache.py`, `storage/schema.py`, `storage/inserts.py`, `storage/search.py`. Each module has clear single responsibility. All modules now use consistent module-level imports.
- **Type checking**: Strict mypy configuration enabled (`disallow_untyped_defs = true`). All public functions have type hints. Run `uv run mypy src/` to verify.
- **Pre-commit hooks**: ruff check/format + mypy run automatically before each commit. Configure with `pre-commit install`.
