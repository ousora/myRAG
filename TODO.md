# myRAG — TODO

## Backlog

### P1 — Important

- [ ] **CLI search subcommand**
  - `python -m pipeline search "question" --db data/doc.db`

---

## Completed (summary)

Core pipeline: multi-format parser → TextCleaner → LLM formatter → markdown writer → chunker → bge-m3 embedder → sqlite-vec storage.

- `rag_query()` — hybrid search (vector + FTS5 RRF) + MMR re-ranking + LLM answer generation
- `process_directory_hybrid()` — batch processing with concurrent LLM/embedding
- `process_file_with_md()` / `--no-llm` — offline deterministic markdown generation
- Hash-based pseudo-embedding fallback for offline/dev use
- `TextCleaner` and `Chunker` facade classes in `pipeline.core`
- `__main__.py` entry point, `[project.scripts]` for `myrag` CLI
- Config hot reload, validation, and CWD-independent paths
- All code audit fixes (ruff/mypy), type annotations, context managers, ThreadPoolExecutor shutdown
- 270+ tests passing

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.
