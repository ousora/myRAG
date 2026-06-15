# Ingest Document Skill

Run the two-phase pipeline: generate structured markdown from a document, then ingest it into the vector DB.

## When to use

- User says "ingest", "process", "add to RAG", or "embed document"
- Converting a raw file (.pdf/.docx/.html) for semantic search

## Steps

### Phase 1: Generate Markdown

```python
from pipeline import process_file_with_md

md_path = process_file_with_md(
    filepath="path/to/document.pdf",
    output_dir="./output/",
)
```

This calls the LLM formatter and writes a structured `.md` file. The user should inspect it before Phase 2.

**CLI alternative:**
```bash
uv run python -m pipeline md input.pdf --output-dir output/
```

### Phase 2: Ingest to Vector DB

```python
from pipeline import _ingest_markdown

_ingest_markdown(
    md_path="output/document.md",
    store_path="data/myrag.db",
)
```

**CLI alternative:**
```bash
uv run python -m pipeline ingest output/document.md --store data/myrag.db
```

### One-Step Alternative (LLM + Embed in one call)

For fully automated processing without inspecting markdown:

```python
from pipeline import process_file_hybrid

result = process_file_hybrid(
    filepath="path/to/document.pdf",
    doc_id="unique_doc_id",
    store_path="data/myrag.db",
)
# Returns: {"chunks": [...], "document": {...}, "format_result": {...}, "md_path": None, "db_path": "..."}
```

**CLI alternative:**
```bash
uv run python -m pipeline process input.pdf --store data/myrag.db
```

## Querying After Ingest

```python
from storage.sqlite_vec import SQLiteVecStore
from embedders import Embedder

db = SQLiteVecStore("data/myrag.db")
e = Embedder()
hits = db.search_chunks(e.embed("search query"), k=5)
```

## Notes

- `store_path` must end in `.db` — the directory is created automatically
- Each document needs a unique `doc_id` to avoid overwriting chunks
- The two-phase approach (inspect .md before ingest) is recommended for production use
- For traditional RAG (no LLM), use `process_file()` which returns chunk dicts directly
