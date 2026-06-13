# myRAG — RAG Data Cleanup Pipeline (Scheme C)

## Architecture

```text
Raw file (.pdf/.docx/.html/.md/.txt)
    ↓ parser.parse()            # MarkItDown / Trafilatura → Clean Markdown
    ↓ cleaner.clean()           # TextCleaner: noise removal, anchor cleanup
    ↓ formatter.format_text()   # LLM semantic structuring → title/tags/sections[]
        → write_to_md(result)   [human-readable .md with hierarchical headers]
    ↓ chunker.chunk(section_path=...)  [downstream: split into embedding-ready chunks]
    ↓ embedder.store_chunks()             [optional vector DB pipeline]
```

## Pipeline Components (Scheme C)

### 1. Parser (`parsers/`) — Unified Text Extraction

Uses **MarkItDown** for PDF, DOCX, Markdown, and TXT files; **Trafilatura** for HTML pages. Both convert directly to clean markdown/text format.

**Usage:**
```python
from myrag.parsers.dispatcher import resolve_parser
parser = resolve_parser("report.pdf")
raw_text = parser.parse("/path/to/report.pdf")
```

Supported extensions: `pdf`, `docx`, `md` / `mkd`, `txt`, `html` / `htm`.

### 2. TextCleaner (`parsers/text_cleaner.py`) — Deterministic Cleaning

Removes noise artifacts via configurable regex rules (default + user overrides in YAML config): extra whitespace, leading markdown headers, repeated paragraphs. Runs before LLM to reduce token cost and improve output quality.

```python
from myrag.parsers.text_cleaner import TextCleaner
cleaned = TextCleaner.clean(raw_text)  # or: cleaned = TextCleaner.clean(raw_text, rules_config="custom_rules.yaml")
```

### 3. Formatter (`formatters/`) — LLM Semantic Structuring

Calls local LLM endpoint (Qwen3.6-35B-A3B-MoE via llama.cpp at `192.168.191.112:8081`) to produce structured output with title, tags, and metadata.sections[]. The LLM does NOT output chunks — chunking is handled by the downstream Chunker module.

**Usage:**
```python
from myrag.formatters import format_text_async, write_to_md

future = format_text_async(cleaned, source_type="pdf")
result = future.result(timeout=300)  # LLM response may take time
write_to_md(result, "output/")       # Writes .md file with proper headers
```

### 4. Chunker (`chunkers/`) — Fine-Grained Splitting (Downstream)

Splits the formatter's full text into smaller chunks for embedding. Automatically parses section headers from markdown to assign correct semantic context per chunk.

```python
from myrag.chunkers import Chunker
chunks = Chunker(max_chars=256).chunk(full_markdown_text)  # auto-detects ## sections
```

### 5. Embedder (`embedders/`) — Vector Indexing (Optional)

Calls local bge-m3 embedding service for vector DB storage (FAISS/Milvus).

## Directory Structure

```text
myrag/
├── parsers/          # Unified parser via MarkItDown / Trafilatura
│   ├── dispatcher.py     # resolve_parser() routing logic
│   └── text_cleaner.py   # TextCleaner: noise removal & normalization
├── cleaners/         # Backward compat facade — re-exports TextCleaner + clean_text() from parsers.text_cleaner
├── chunkers/         # Chunking module (max_chars, overlap)
├── embedders/        # bge-m3 embedding client (OpenAI-compatible API)
├── formatters/       # LLM-based text formatter + markdown writer
│   ├── __init__.py     # format_text() / format_text_async()
│   ├── prompts.py      # System prompt template (clean section_path constraints)
│   └── writer.py       # write_to_md() — renders chunks with hierarchical headers
├── pipeline.py       # process_file_with_md() + process_directory() entry points
├── pyproject.toml    # Project config (setuptools build, dev deps)
├── CHANGELOG.md      # Version history
├── LICENSE           # MIT License
└── README.md         # This file

.doc/                 # Raw input documents (PDFs, TXTs, etc.)
.output/              # Formatted output (.md files)
```

## Quick Start

### Process a Single File to Markdown
```python
from myrag.pipeline import process_file_with_md

path = process_file_with_md("report.pdf", output_dir="output/")
# Writes: output/report.md with proper H1/H2/H3 headers
```

### CLI Usage
```bash
cd /home/colinvan/workspace && PYTHONPATH=/home/colinvan/workspace myrag/.venv/bin/python -m myrag.pipeline md ./myrag/.doc/cnaps.txt --output-dir ./myrag/output/
```

### LLM-Formatted Output (Programmatic)
```python
from myrag.formatters import format_text_async, write_to_md

with open("cnaps.txt") as f:
    raw = f.read()[:15000]  # Truncate for testing

future = format_text_async(raw, source_type="txt")
result = future.result(timeout=300)
write_to_md(result, "output/")       # Writes output/title.md with headers
```

## Configuration

- **Embedding Service**: `embedders/bge_m3.py` — set base_url for local bge-m3 endpoint
- **LLM Endpoint**: `formatters/__init__.py` ENDPOINT + MODEL variables (defaults to qwenpaw service at 192.168.191.112:8081)
- **Chunk Size**: Default 512 chars, configurable via Chunker(max_chars=...)

## Testing

```bash
cd myrag && PYTHONPATH=/home/colinvan/workspace python3 -m pytest formatters/tests/ cleaners/tests/ -v
# Expected: passes with mock LLM responses
```
