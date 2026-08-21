# myRAG — TODO

## Backlog

### P1 — Important

- [ ] **Migrate chunks table to sqlite-vec `vec0` virtual table**
  - Current searches are brute-force `ORDER BY vec_distance_cosine(...)` full scans; a `vec0` KNN virtual table (`WHERE embedding MATCH ? AND k = ?`) would give indexed ANN retrieval.
  - Needs a migration path for existing DBs (user_version bump) and metadata-column filters for `source_doc_id`/section queries.

### Done (2026-08-21)

- [x] **CLI query subcommand** — `python -m src query "question" --store data/doc.db`

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
