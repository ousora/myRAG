# myRAG — RAG Pipeline

```
.doc/file → parse → clean → format ─┬→ write_to_md()  → 可读 .md
                                    └→ chunk → embed → sqlite-vec
```

## Architecture

```text
Raw file (.pdf/.docx/.html/.md/.txt)
    ↓ parser.parse()              # MarkItDown / Trafilatura → text
    ↓ cleaner.clean()             # TextCleaner: noise removal
    ↓ formatter.format_text()     # LLM → {title, tags, sections, body}
    ├→ write_to_md(result)        # 可读 .md 文件
    └→ _render_markdown_with_sections(result)
        ↓ chunker.chunk(md)       # LangChain MarkdownHeaderTextSplitter
        ↓ embedder.store_chunks() # bge-m3 → 1024-d vectors
        ↓ SQLiteVecStore          # sqlite-vec: chunks + FTS5 + documents
```

## Pipeline Components

### 1. Parser (`parsers/`)

**MarkItDown** (pdf, docx, md, txt) + **Trafilatura** (html). Single `resolve_parser()` dispatcher.

```python
from myrag.parsers.dispatcher import resolve_parser
parser = resolve_parser("report.pdf")
text = parser.parse("report.pdf")
```

### 2. TextCleaner (`parsers/text_cleaner.py`)

Control chars, page breaks, whitespace. Optional YAML config for custom regex rules.

```python
from myrag.parsers.text_cleaner import TextCleaner
cleaned = TextCleaner().clean(raw_text)
```

### 3. Formatter (`formatters/`)

LLM-powered: extracts title, tags, section hierarchy. Configurable via `conf/config.yaml`.

```python
from myrag.formatters import format_text_async, write_to_md

future = format_text_async(cleaned, source_type="pdf")
result = future.result(timeout=300)
md_path = write_to_md(result, "output/")    # readable markdown
```

### 4. Chunker (`chunkers/`)

LangChain `MarkdownHeaderTextSplitter` splits on `##`/`###` boundaries. Oversized sections get `RecursiveCharacterTextSplitter` fallback. Plain text without headers auto-detected.

```python
from myrag.chunkers import Chunker
chunks = Chunker(chunk_size=512, chunk_overlap=64).chunk(markdown_text)
# Each chunk: {"text": "...", "section_path": ["Services", "HVPS"], "metadata": {...}}
```

### 5. Embedder + Storage

bge-m3 embeddings → sqlite-vec database with FTS5 full-text index.

```python
# One-shot: ingest + embed + store
from myrag.pipeline import process_file_hybrid

result = process_file_hybrid(
    "report.pdf", doc_id="doc_001",
    chunk_size=512,
    store_path="data/myrag.db",   # persist to sqlite-vec
)

# Query
from myrag.embedders import Embedder
from myrag.storage.sqlite_vec import SQLiteVecStore

db = SQLiteVecStore("data/myrag.db")
e = Embedder()
hits = db.search_chunks(e.embed("your question"), k=5)
```

## Quick Start

### Install

```bash
pip install -e ".[dev,sqlite-vec]"
cp conf/config.example.yaml conf/config.yaml
# Edit conf/config.yaml with your endpoints
```

### CLI

```bash
# Generate readable markdown
python -m myrag.pipeline md input.pdf --output-dir output/

# Full ingest: format → chunk → embed → sqlite-vec
python -m myrag.pipeline hybrid input.pdf --store data/myrag.db

# Traditional (no LLM, no storage)
python -m myrag.pipeline process-file input.txt --chunk-size 512
```

## Directory Structure

```text
myrag/
├── config.py              # Config loader: get_config()
├── pipeline.py            # process_file / process_file_hybrid / process_file_with_md
├── parsers/               # MarkItDown + Trafilatura dispatcher
│   ├── dispatcher.py
│   └── text_cleaner.py
├── cleaners/              # Backward-compat facade
├── formatters/            # LLM formatter + prompts + markdown writer
│   ├── __init__.py
│   ├── prompts.py
│   └── writer.py
├── chunkers/              # LangChain MarkdownHeaderTextSplitter wrapper
├── embedders/             # bge-m3 embedding client (OpenAI-compatible API)
├── storage/               # SQLiteVecStore
│   └── sqlite_vec.py
├── conf/
│   ├── config.yaml        # Your endpoints (gitignored)
│   └── config.example.yaml # Template (committed)
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

## Configuration

Single file: `conf/config.yaml` (gitignored). Template at `conf/config.example.yaml`.

```yaml
llm:
  endpoint: "http://your-llm:8081/v1/chat/completions"
  model: "your-model-name"
  temperature: 0.3
  max_tokens: 8192
  timeout: 180

embedding:
  base_url: "http://your-embedder:11435"
  model: "bge-m3"
  timeout: 60
```

Resolution: `$MYRAG_CONFIG` → `conf/config.yaml` → `conf/config.example.yaml`.

```python
from myrag.config import get_config
cfg = get_config()
print(cfg.llm_endpoint)  # from your config file
```

## Testing

```bash
cd /home/colinvan/workspace
PYTHONPATH=. myrag/.venv/bin/python -m pytest myrag/chunkers/tests/ myrag/formatters/tests/ myrag/cleaners/tests/ -v
# 22 tests: chunkers 8 + formatters 9 + cleaners 5
```
