# myRAG — RAG Data Cleanup Pipeline (Scheme C)

## Architecture

```text
Raw file (.pdf/.docx/.html/.md/.txt)
    ↓ parser.parse()            # MarkItDown / Trafilatura → Clean Markdown
    ↓ cleaner.clean()           # TextCleaner: noise removal, whitespace normalization
    ↓ formatter.format_text()   # LLM semantic structuring → title/tags/sections[]
        → write_to_md(result)   [human-readable .md with hierarchical headers]
    ↓ chunker.chunk(text)       # Auto-parse headers, split into embedding-ready chunks
    ↓ embedder.store_chunks()   [optional: vector DB indexing — Hybrid A+B]
```

## Pipeline Components

### 1. Parser (`parsers/`) — Unified Text Extraction

Uses **MarkItDown** for PDF, DOCX, Markdown, TXT and **Trafilatura** for HTML. Both convert to clean markdown/text format via `resolve_parser()`.

```python
from myrag.parsers.dispatcher import resolve_parser
parser = resolve_parser("report.pdf")
raw_text = parser.parse("/path/to/report.pdf")
```

Supported: `pdf`, `docx`, `md`/`mkd`, `txt`, `html`/`htm`.

### 2. TextCleaner (`parsers/text_cleaner.py`) — Deterministic Cleaning

Removes control chars, page breaks, normalizes whitespace. Supports optional YAML config for user-defined regex rules.

```python
from myrag.parsers.text_cleaner import TextCleaner
cleaned = TextCleaner().clean(raw_text)
```

### 3. Formatter (`formatters/`) — LLM Semantic Structuring

Calls local LLM (Qwen MoE at `192.168.191.112:8081`) for title/tags/section extraction. Outputs structured JSON with `body` field for downstream chunking.

```python
from myrag.formatters import format_text_async, write_to_md

future = format_text_async(cleaned, source_type="pdf")
result = future.result(timeout=300)
md_path = write_to_md(result, "output/")
```

### 4. Chunker (`chunkers/`) — Header-Enriched Splitting

Auto-parses markdown headers from formatter output, assigns semantic context per chunk, and optionally prepends headers for better vector retrieval.

```python
from myrag.chunkers import Chunker
chunks = Chunker(max_chars=512).chunk(cleaned_text)
# Each chunk: {"text": "## Section\n\ncontent...", "section_path": ["Section"]}
```

### 5. Embedder (`embedders/`) — Vector Indexing (Optional)

OpenAI-compatible bge-m3 embedding client.

```python
from myrag.embedders import Embedder
e = Embedder(base_url="http://your-server:11435")
vectors = e.embed(["text1", "text2"])
```

## Directory Structure

```text
myrag/
├── parsers/              # MarkItDown + Trafilatura unified backend
│   ├── dispatcher.py           # resolve_parser() routing
│   └── text_cleaner.py         # TextCleaner with YAML config support
├── cleaners/             # Backward-compat facade → parsers.text_cleaner
├── chunkers/             # Header-enriched chunking with auto section detection
├── embedders/            # bge-m3 embedding client (OpenAI-compatible API)
├── formatters/           # LLM text formatter + markdown writer
│   ├── __init__.py             # format_text_async()
│   ├── prompts.py              # System prompt template
│   └── writer.py               # write_to_md() with H2-H5 header rendering
├── storage/              # SQLite-vec vector store (standalone, not yet integrated)
│   └── sqlite_vec.py
├── pipeline.py           # process_file() / process_file_hybrid() / process_file_with_md()
├── pyproject.toml        # Project config + dependencies
├── CHANGELOG.md          # Version history
└── README.md             # This file
```

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Process a Single File

```python
from myrag.pipeline import process_file_with_md

path = process_file_with_md("report.pdf", output_dir="output/")
```

### CLI Usage

```bash
PYTHONPATH=/home/colinvan/workspace python -m myrag.pipeline md input.pdf
PYTHONPATH=/home/colinvan/workspace python -m myrag.pipeline hybrid input.pdf --doc-id mydoc_001
```

## Configuration

All endpoints and model settings live in **`conf/config.yaml`** (gitignored — safe for local IPs).

A template with safe defaults is provided at `conf/config.example.yaml` (committed).

```yaml
# config/config.yaml
llm:
  endpoint: "http://192.168.191.112:8081/v1/chat/completions"
  model: "Qwen/QwQ-32B-A3B-..."
  temperature: 0.3
  max_tokens: 8192
  timeout: 180

embedding:
  base_url: "http://192.168.191.112:11435"
  model: "bge-m3"
  timeout: 60
```

**Resolution order** (first wins):
1. `$MYRAG_CONFIG` environment variable
2. `conf/config.yaml` (your instance)
3. `conf/config.example.yaml` (safe defaults)

**Runtime override**:
```python
from myrag.config import get_config
cfg = get_config()
print(cfg.llm_endpoint)   # from config file

# Embedder with explicit endpoint (ignores config)
e = Embedder(base_url="http://other-server:11435")
```

## Testing

```bash
cd /home/colinvan/workspace && PYTHONPATH=/home/colinvan/workspace python3 -m pytest myrag/formatters/tests/ myrag/cleaners/tests/ -v
```
