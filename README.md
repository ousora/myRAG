# myrag-pipeline — RAG Data Cleanup Pipeline

## Architecture

```text
myrag/
├── parsers/          # File-type-specific document parsers
│   ├── dispatcher.py     # PARSERS registry + resolve_parser()
│   ├── pdf.py            # PyMuPDF parser
│   ├── docx.py           # python-docx parser
│   ├── html.py           # BeautifulSoup + readability-lxml
│   ├── md_parser.py      # Python markdown module
│   └── txt.py            # Plain text (multi-encoding fallback)
├── cleaners/         # Text cleaning
│   └── __init__.py     # TextCleaner — denoising, normalization
├── chunkers/         # Text segmentation
│   └── __init__.py     # Chunker — size + overlap control
├── embedders/        # Embedding client (bge-m3)
│   ├── bm25.py         # Embedder class
│   └── __init__.py
├── formatters/       # Text formatting via LLM
│   ├── prompts.py          # System prompt templates
│   ├── writer.py           # Markdown file writer + formatter
│   ├── __init__.py     # format_text() / format_text_async()
│   └── tests/            # Unit tests
├── pipeline.py       # Entry point: process_file / process_directory
└── pyproject.toml    # Dependencies and build config
```

## Quick Start

### Install
```bash
cd myrag && pip install -e ".[dev]"
```

### Parse a single file
```python
from myrag.pipeline import process_file

chunks = process_file("path/to/report.pdf")
# → [{"text": "...", "metadata": {"source": "..."}}, ...]
```

### Process an entire directory
```python
from myrag.pipeline import process_directory

all_chunks = process_directory("./docs", max_chars=512)
```

### Embedding (bge-m3)
```python
from myrag.embedders import Embedder

e = Embedder(base_url="http://192.168.191.112:8081")
emb = e.embed("Hello world")  # → list[float]
```

### Format raw text (LLM-powered)
The `formatters` module cleans and structures raw copied text into knowledge chunks using a local LLM.

```python
from myrag.formatters import format_text, write_to_md

raw = """...copied text from webpage..."""
result = format_text(raw, source_type="web")
# → {"title": "...", "tags": [...], "metadata": {...}, "chunks": [...]}

write_to_md(result, "./output")  # saves as output/<title>.md
```

## CLI
```bash
python -m myrag.pipeline ./docs --max-chars 512 > output.json
```
