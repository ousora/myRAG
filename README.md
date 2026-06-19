# myRAG — RAG Pipeline

```
.doc/file → parse → clean → format ─┬→ write_to_md() → readable .md (with [[wikilinks]])
                                    └→ chunk → match entities → embed → sqlite-vec
```

> Large texts (>28K chars) are auto-split at paragraph boundaries and processed chunk-by-chunk. Each chunk receives the last 10 lines of previous markdown output + cumulative summary as context for continuity.

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
    └→ _render_markdown_with_sections(result)
        ↓ chunker.chunk(body)     # markdown-it-py (pure Python, no LangChain)
        ↓ _match_entities()       # tag chunks with entity_names from text match
        ↓ embedder.store_chunks() # bge-m3 → 1024-d (remote API or local CPU)
        ↓ SQLiteVecStore          # sqlite-vec: chunks + entity_names + FTS5
```

## Pipeline Components

### 1. Parser (`src/parsers/`)

**MarkItDown** (pdf, docx, md, txt) + **Trafilatura** (html). Single `resolve_parser()` dispatcher.

- Lazy import: parsers are loaded on first use (fail-fast in `__init__`).
- TrafilaturaParser handles HTML encoding with UTF-8 → GBK fallback.

```python
from parsers.dispatcher import resolve_parser
parser = resolve_parser("report.pdf")
text = parser.parse("report.pdf")
```

### 2. TextCleaner (`src/parsers/text_cleaner.py`)

Control chars, page breaks, whitespace. Optional YAML config for custom regex rules.

```python
from parsers.text_cleaner import TextCleaner
cleaned = TextCleaner().clean(raw_text)
```

### 3. Formatter (`src/formatters/`)

LLM-powered: extracts title, tags, section hierarchy. **Auto-chunks large texts** (>28K chars) at paragraph boundaries — each chunk gets the last 10 lines of previous markdown output + cumulative summary as context for continuity.

**JSON Schema enforcement**: `call_llm()` accepts a `schema=` parameter to send JSON Schema via `response_format`, letting llama.cpp / OpenAI servers enforce output structure natively (schemas in [constants.py](src/formatters/constants.py)).

**Tag quality**: Extracts proper nouns and domain-specific multi-word phrases; filters generic single words ("banking", "system").

**Output validation**: `validate_format_output()` + `try_fix_common_issues()` for post-processing without re-calling LLM.

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

Pure Python markdown splitting via `markdown-it-py` (no LangChain dependency). Splits on `##`/`###` boundaries with hierarchical metadata tracking. Consecutive headings with no body text between are merged into one section. Oversized sections get recursive character split with sentence-aware boundaries (Chinese `。！？` + English `.!?`). Plain text without headers auto-detected.

```python
from chunkers import Chunker
chunks = Chunker(chunk_size=512, chunk_overlap=64).chunk(markdown_text)
# Each chunk: {"text": "...", "section_path": ["Services", "HVPS"], "metadata": {...}}
```

### 5. Embedder + Storage

bge-m3 embeddings → sqlite-vec database with FTS5 full-text index + entity_names column.

**Dual embedding mode** — set `embedding.mode` in config to switch:
- `"remote"` (default): calls HTTP API at `embedding.base_url` (vLLM / Ollama compatible)
- `"local"`: uses sentence-transformers (`uv sync --extra local-embeddings`), CPU inference, no network dependency

**Entity search** — `entity_names` column stores entity mentions per chunk for cross-doc entity lookup:

```python
# Query by entity name
db.conn.execute(
    "SELECT text FROM chunks WHERE entity_names LIKE ?",
    ['%"GPT-4"%']
).fetchall()

# Build + query
from embedders import Embedder
from storage.sqlite_vec import SQLiteVecStore

db = SQLiteVecStore("data/myrag.db")
e = Embedder()
hits = db.search_chunks(e.embed("your question"), k=5)
```

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
python -m pipeline md input.pdf --output-dir output/

# 2. Ingest an existing .md into sqlite-vec (no LLM call)
python -m pipeline ingest output/doc.md --store data/doc.db

# 3. Generate .md and auto-ingest (two-step, transparent)
python -m pipeline process input.pdf --store data/doc.db

# Traditional (no LLM, no storage)
python -m pipeline process-file input.txt --chunk-size 512
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
│   │   └── ingest.py         # _ingest_markdown function
│   ├── parsers/              # MarkItDown + Trafilatura dispatcher
│   │   ├── dispatcher.py
│   │   └── text_cleaner.py
│   ├── formatters/           # LLM formatter + prompts + markdown writer + wikilinks
│   │   ├── __init__.py
│   │   ├── constants.py      # JSON schemas for response_format (incl. entities)
│   │   ├── prompts.py
│   │   └── writer.py
│   ├── chunkers/             # Pure Python markdown-it-py chunker (no LangChain)
│   ├── embedders/            # bge-m3: remote HTTP API + local sentence-transformers
│   │   ├── __init__.py
│   │   ├── bge_m3.py         # Unified Embedder with mode dispatch
│   │   └── local_bge.py      # LocalEmbedder via sentence-transformers
│   └── storage/              # SQLiteVecStore
│       └── sqlite_vec.py
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
  temperature: 0.0                # 0 for deterministic entity extraction
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
```

## Testing

```bash
cd /home/colinvan/workspace/myrag
uv run pytest -v
# 39 tests: chunkers 8 + formatters 9 + storage 13 + integration 9
```
