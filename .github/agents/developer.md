# Developer Agent Mode

**Mode:** Active development — writing new features, fixing bugs, adding parsers/chunkers/embedders.

**Trigger:** `/mode developer` or when user says "develop", "implement", "build"

## Workflow

1. **Understand scope** — Read the target file and its direct imports/usages in `src/` before editing; if no matching file exists, stop and report the missing path. If the target file or its adjacent code cannot be found under `src/`, do not guess the implementation; report the missing file and ask for the correct path. If `src/`, `tests/`, `.github/copilot-instructions.md`, or `conf/config.yaml` is missing or differs from the expected layout, stop and report the missing path before making changes.
2. **Write code** — Follow conventions in `.github/copilot-instructions.md` (type hints, logging, no hardcoded values).
3. **Add tests** — Every new function/module needs tests in the corresponding `tests/` subdirectory.
4. **Verify** — Run lint + tests before suggesting commit. If `uv run ruff check .` or `uv run pytest -v` fails, stop, report the failing output, and do not suggest a commit until the failures are resolved. If lint or tests fail, report the failing command and output, do not suggest a commit, and fix the failures before continuing.

## Development Rules

- **Read before writing** — Always read the target file and adjacent code to match style
- **Use existing patterns** — When adding a parser, implement the `TextParser` interface exactly as defined in the current codebase; when adding an embedder, instantiate and use `httpx.Client` as the existing embedders do.
- **Config-first & backward compat** — For any public API change, preserve the existing public signature and delegate to the new implementation; add new settings to `conf/config.yaml` via `Config`, not to the function signature, unless the caller explicitly requires per-invocation behavior. Do not add new function parameters to public APIs. Store new settings in `conf/config.yaml` via the `Config` class. Use a function parameter only when the value is truly specific to one invocation and cannot be stored in config.
- **File limit 500 lines** — Split modules that grow beyond this

## Key Entry Points

| Public API | File | Purpose |
|-----------|------|---------|
| `process_file()` | `src/pipeline.py` | Traditional RAG (parse → clean → chunk) |
| `process_file_hybrid()` | `src/pipeline.py` | LLM format + embed + sqlite-vec |
| `process_file_with_md()` | `src/pipeline.py` | LLM format → write .md |
| `_ingest_markdown()` | `src/pipeline.py` | Ingest existing .md to vector DB |
| `format_text_async()` | `src/formatters/__init__.py` | Async LLM formatting |
| `Chunker.chunk()` | `src/chunkers/__init__.py` | Markdown/text chunking |
| `Embedder.embed()` | `src/embedders/bge_m3.py` | bge-m3 embedding |
| `SQLiteVecStore` | `src/storage/sqlite_vec.py` | Vector storage + FTS5 |

## Testing Checklist

- [ ] New code has unit tests in the appropriate `tests/` directory
- [ ] Edge cases covered: empty input, missing deps, malformed data
- [ ] Existing tests still pass: `uv run pytest -v`
- [ ] Lint clean: `uv run ruff check .`
