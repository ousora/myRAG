# myRAG — RAG Data Cleanup Pipeline (Scheme C)

## Architecture

```text
Raw file (.pdf/.docx/.txt)
    ↓ parser.parse_file()     # Extract text from binary formats
    ↓ cleaner.clean_text()    # Regex-based noise removal (ads, nav bars, etc.)
    ↓ formatter.format_text()  # LLM semantic structuring into title/tags/chunks
        → write_to_md(result, output_dir/)   [human-readable .md]
    ↓ chunker.chunk() or embedder.embed()     [optional vector DB pipeline]
```

## Pipeline Components (Scheme C)

### 1. Parser (`parsers/`) — Text Extraction
Extracts raw text from binary formats: PDF (PyMuPDF), DOCX (python-docx), HTML (BeautifulSoup+readability), Markdown, TXT (with encoding detection).

**Usage:**
```python
from myrag.parsers import resolve_parser
parser = resolve_parser("report.pdf")
raw_text = parser.parse_file("/path/to/report.pdf")
```

### 2. Cleaner (`cleaners/`) — Deterministic Noise Removal
Regex-based cleaning: page breaks, extra whitespace, navigation bars. Runs **before** the LLM formatter to reduce token cost.

```python
from myrag.cleaners import TextCleaner
cleaned = TextCleaner().clean(raw_text)
```

### 3. Formatter (`formatters/`) — LLM Semantic Structuring
Calls local bge-m3 endpoint to produce structured output: title, tags, metadata, section_path arrays, and chunks (512 chars each).

**Usage:**
```python
from myrag.formatters import format_text_async, write_to_md

future = format_text_async(cleaned, source_type="web")
result = future.result(timeout=300)  # LLM response may take time
write_to_md(result, "output/")       # Writes .md file
```

### 4. Chunker (`chunkers/`) — Fine-Grained Splitting (Optional)
Splits formatter output into smaller chunks for embedding. Typically not needed — formatter already produces ~512-char semantic segments.

```python
from myrag.chunkers import Chunker
chunks = Chunker(max_chars=256).chunk(formatted_text)
```

### 5. Embedder (`embedders/`) — Vector Indexing (Optional)
Calls local bge-m3 embedding service for vector DB storage (FAISS/Milvus).

 ## Directory Structure

  ```text
  myrag/
  ├── parsers/          # File-type-specific document parsers
  │   ├── dispatcher.py     # PARSERS registry + resolve_parser()
  │   ├── pdf.py            # PyMuPDF parser
  │   ├── docx.py           # python-docx parser
  │   ├── html.py           # BeautifulSoup + readability parser
  │   ├── md_parser.py      # markdown library parser
  │   └── txt.py            # Plain text with encoding detection
  ├── cleaners/         # TextCleaner for noise removal & normalization
  ├── chunkers/         # Chunking module (max_chars, overlap)
  ├── embedders/        # bge-m3 embedding client (OpenAI-compatible API)
  ├── formatters/       # LLM-based text formatter + markdown writer
  │   ├── __init__.py       # format_text() / format_text_async()
  │   ├── prompts.py        # System prompt template
  │   └── writer.py         # write_to_md() / format_md()
  ├── pipeline.py     # process_file() + process_directory() entry points
  ├── pyproject.toml  # Project config (setuptools build, dev deps)
  ├── CHANGELOG.md    # Version history
  ├── LICENSE         # MIT License
  └── README.md       # This file

  .test/                  # Test artifacts (not tracked in git)
  ├── output/             # Unit test results & formatted outputs (.json, .md)
  └── scripts/            # Standalone test scripts for manual runs
      └── run_pdf_test.py

  .doc/                   # Raw input documents (PDFs, TXTs, etc.)
```

## Quick Start

### Parse a Single File (Traditional RAG)
```python
from myrag.pipeline import process_file

chunks = process_file("report.pdf")  # Returns list[dict] with text + metadata
print(json.dumps(chunks, indent=2))
```

### Process Directory
```python
from myrag.pipeline import process_directory

all_chunks = process_directory("./docs", max_chars=512)
# Processes all supported files in ./docs/ recursively
```

### LLM-Formatted Output (Scheme C)
```python
from myrag.formatters import format_text_async, write_to_md

with open("cnaps.txt") as f:
    raw = f.read()[:15000]  # Truncate for testing

future = format_text_async(raw, source_type="web")
result = future.result(timeout=300)  # LLM response may take up to 2 minutes
write_to_md(result, "output/")       # Writes output/title.md
```

## Configuration

- **Embedding Service**: `embedders/bge_m3.py` — set base_url for local bge-m3 endpoint
- **LLM Endpoint**: `formatters/__init__.py` ENDPOINT + MODEL variables (defaults to your qwenpaw service)
- **Chunk Size**: Default 512 chars, configurable via Chunker(max_chars=...)

## Testing

```bash
cd myrag && python3 -m pytest formatters/tests/ -v
# Expected: 10 passed
```
