# Adding a New Parser to myRAG

## Overview

The parser dispatcher in `src/parsers/dispatcher.py` routes files to the right extractor based on file extension. It uses MarkItDown (pdf/docx/md/txt) and Trafilatura (html). Add new parsers by registering them with the dispatcher.

## Steps

### 1. Create the Parser Class

In `src/parsers/`, create a new file (e.g., `epub_parser.py`) or add to an existing one:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class TextParser(Protocol):
    def parse(self, filepath: str) -> str: ...

class MyNewParser(TextParser):
    """Extract text from .xyz files using some library."""

    def __init__(self) -> None:
        # Initialize any dependencies here
        pass

    def parse(self, filepath: str) -> str:
        # Return cleaned text content
        ...
```

**Requirements:**
- Must implement `parse(self, filepath: str) -> str` matching the `TextParser` protocol
- Use specific `try/except` blocks — no bare `except:`
- Log at INFO level on success, WARNING/ERROR on failure
- Handle empty files and missing dependencies gracefully

### 2. Register with the Dispatcher

In `src/parsers/dispatcher.py`, add registration at module load time (bottom of file):

```python
# Import your parser
from .epub_parser import MyNewParser

# Register for the extension
register_parser("xyz", MyNewParser)
# Optional: register aliases
# register_parser("myformat", MyNewParser)  # also maps to xyz via alias
```

The `register_parser()` function handles extension → parser mapping and common aliases (markdown→md/mkd, html→htm).

### 3. Add Dependencies

If the parser needs new packages:
1. Add to `[project.optional-dependencies]` in `pyproject.toml` under a descriptive extra name
2. Document the extra in README.md Quick Start section

### 4. Write Tests

Create `src/parsers/tests/test_<name>.py`:

```python
from parsers.epub_parser import MyNewParser

class TestMyNewParser:
    def test_parses_valid_file(self):
        parser = MyNewParser()
        result = parser.parse("test/fixtures/sample.xyz")
        assert len(result) > 0

    def test_empty_file_returns_empty_string(self):
        parser = MyNewParser()
        result = parser.parse("test/fixtures/empty.xyz")
        assert result == ""
```

### 5. Update Documentation

Update these files:
- `README.md` — add the new extension to the supported formats list
- `conf/config.example.yaml` — if the parser needs new config fields
- `.github/AGENTS.md` — add entry to Key Directories table if it's a new submodule

## Common Pitfalls

- **Don't create a separate module** unless the parser has complex logic (>50 lines). Simple parsers can be inline in `dispatcher.py`.
- **Always use `register_parser()`** — never modify the PARSERS dict directly. This ensures aliases work correctly.
- **MarkItDown already handles** .md, .txt, .pdf, .docx. Only add a new parser if it handles something MarkItDown doesn't (e.g., .epub, .rst, .csv).
- **Trafilatura handles** .html/.htm. Use it for any HTML-based format that needs content extraction.
