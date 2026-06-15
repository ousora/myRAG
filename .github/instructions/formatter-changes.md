# Modifying the LLM Formatter

## Overview

The formatter (`src/formatters/`) uses an LLM to extract structured data (title, tags, sections, body) from raw text. It supports two modes: single-shot (small docs) and chunked (large docs >28K chars).

## Key Files

| File | Purpose |
|------|---------|
| `src/formatters/__init__.py` | Public API: `format_text()`, `format_text_async()`, auto-chunking logic |
| `src/formatters/prompts.py` | LLM system prompts: `SYSTEM_PROMPT`, `CHUNKED_SYSTEM_PROMPT`, getter functions |
| `src/formatters/writer.py` | Markdown file writer: `write_to_md()`, section header rendering |

## Prompt Structure

### Single-Shot (`SYSTEM_PROMPT`)
- Expects valid JSON output with keys: `title`, `tags`, `metadata`, `body`
- `body` must contain ALL content — never summarize
- `metadata.sections` defines the document hierarchy (level 2 = major sections)
- Chrome removal rules are embedded in the prompt (navigation, footers, TOC items)

### Chunked (`CHUNKED_SYSTEM_PROMPT`)
- Receives: current chunk text + last 10 lines of previous output + cumulative summary
- Produces: markdown content for this chunk + one-sentence summary for continuity
- Summary is appended to the cumulative context for the next chunk

## Making Changes

### Modifying Prompt Content

Edit `src/formatters/prompts.py`. The prompts are raw strings — update the text directly.

```python
# Before changing, check what the prompt expects:
# - SYSTEM_PROMPT expects JSON with title/tags/metadata/body keys
# - CHUNKED_SYSTEM_PROMPT expects markdown content + summary
```

**Important:** After changing prompts, update the example in the prompt itself to match the new format.

### Changing Chunking Threshold

In `src/formatters/__init__.py`:

```python
_CHUNK_THRESHOLD_CHARS = 28000  # Change this value
```

This controls when auto-chunking triggers. Default is ~28K chars (~7000 tokens). Adjust based on your LLM's context window.

### Modifying JSON Output Schema

If you change what the formatter returns (e.g., add new fields), update:
1. The prompt in `prompts.py` — describe the new field
2. `writer.py` — handle the new field in `write_to_md()`
3. `pipeline.py` — any code that consumes the result dict
4. Tests in `src/formatters/tests/test_formatter.py`

### Testing Prompt Changes

```bash
# Run formatter tests (mocks LLM API)
uv run pytest src/formatters/tests/ -v

# Manual test with real LLM
uv run python -c "
from formatters import format_text_async
text = open('test/sample.txt').read()
future = format_text_async(text, source_type='txt')
result = future.result(timeout=300)
print(result['title'])
"
```

## Common Pitfalls

- **`format_text_async()` returns a Future** — always call `.result(timeout=N)` to get the dict
- **The LLM may return markdown fences** (` ```json `) — the code strips them with regex, but prompts should specify "JSON only, no fences"
- **Chunked mode summaries must be one sentence** — longer summaries break context continuity in subsequent chunks
- **`write_to_md()` uses `title` for filename** — sanitize titles (non-alphanumeric → `_`) to avoid filesystem issues
- **Section levels start at 2** — H1 is reserved for the document title; sections use H2/H3
