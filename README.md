# myRAG — RAG Pipeline

```
.doc/file → parse → clean → format ─┬→ write_to_md()  → 可读 .md
                                    └→ chunk → embed → sqlite-vec
```

> 大文档自动按段落分片处理（>28K chars），每片保留前文收尾 + 累计摘要作为上下文，保证连续性。

## Architecture

```text
Raw file (.pdf/.docx/.html/.md/.txt)
    ↓ parser.parse()              # MarkItDown / Trafilatura → text
    ↓ cleaner.clean()             # TextCleaner: noise removal
    ↓ formatter.format_text()     # LLM → {title, tags, sections, body}
    │                             # 小文档一次调用，大文档自动分片
    ├→ write_to_md(result)        # 可读 .md 文件
    └→ _render_markdown_with_sections(result)
        ↓ chunker.chunk(md)       # LangChain MarkdownHeaderTextSplitter
        ↓ embedder.store_chunks() # bge-m3 → 1024-d vectors
        ↓ SQLiteVecStore          # sqlite-vec: chunks + FTS5 + documents
```

## Pipeline Components

### 1. Parser (`src/parsers/`)

**MarkItDown** (pdf, docx, md, txt) + **Trafilatura** (html). Single `resolve_parser()` dispatcher.

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

```python
from formatters import format_text_async, write_to_md

future = format_text_async(cleaned, source_type="pdf")
result = future.result(timeout=3600)
md_path = write_to_md(result, "output/")    # readable markdown
```

### 4. Chunker (`src/chunkers/`)

LangChain `MarkdownHeaderTextSplitter` splits on `##`/`###` boundaries. Oversized sections get `RecursiveCharacterTextSplitter` fallback. Plain text without headers auto-detected.

```python
from chunkers import Chunker
chunks = Chunker(chunk_size=512, chunk_overlap=64).chunk(markdown_text)
# Each chunk: {"text": "...", "section_path": ["Services", "HVPS"], "metadata": {...}}
```

### 5. Embedder + Storage

bge-m3 embeddings → sqlite-vec database with FTS5 full-text index.

```python
# One-shot: ingest + embed + store
from pipeline import process_file_hybrid

result = process_file_hybrid(
    "report.pdf", doc_id="doc_001",
    chunk_size=512,
    store_path="data/myrag.db",   # persist to sqlite-vec
)

# Query
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
cp conf/config.example.yaml conf/config.yaml
# Edit conf/config.yaml with your endpoints
```

### CLI

```bash
# Generate readable markdown
python -m pipeline md input.pdf --output-dir output/

# Full ingest: format → chunk → embed → sqlite-vec
python -m pipeline hybrid input.pdf --store data/myrag.db

# Traditional (no LLM, no storage)
python -m pipeline process-file input.txt --chunk-size 512
```

## Directory Structure

```text
myrag/
├── src/
│   ├── __init__.py           # Package init
│   ├── config.py             # Config loader: get_config()
│   ├── pipeline.py           # process_file / process_file_hybrid / process_file_with_md
│   ├── parsers/              # MarkItDown + Trafilatura dispatcher
│   │   ├── dispatcher.py
│   │   └── text_cleaner.py
│   ├── cleaners/             # Backward-compat facade
│   ├── formatters/           # LLM formatter + prompts + markdown writer
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── writer.py
│   ├── chunkers/             # LangChain MarkdownHeaderTextSplitter wrapper
│   ├── embedders/            # bge-m3 embedding client (OpenAI-compatible API)
│   └── storage/              # SQLiteVecStore
│       └── sqlite_vec.py
├── conf/
│   ├── config.yaml           # Your endpoints (gitignored)
│   └── config.example.yaml   # Template (committed)
├── output/                   # Generated markdown files
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
  temperature: 0.3
  max_tokens: 16384            # Chunked mode uses this per-chunk
  timeout: 300                 # seconds for HTTP request

embedding:
  base_url: "http://your-embedder:11435"
  model: "bge-m3"
  timeout: 60
```

Resolution: `$MYRAG_CONFIG` → `conf/config.yaml` → `conf/config.example.yaml`.

```python
from config import get_config
cfg = get_config()
print(cfg.llm_endpoint)  # from your config file
```

## Testing

```bash
cd /home/colinvan/workspace/myrag
uv run pytest -v
# 22 tests: chunkers 8 + formatters 9 + cleaners 5
```
