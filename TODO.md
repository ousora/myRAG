# myRAG — TODO

## Backlog

### Done (2026-08-23)

- [x] **Fix Chinese full-text search (FTS5 v3 schema)** — unicode61 merges CJK runs into one token, so single-char queries never matched. Chunks now carry a pre-segmented `text_seg` column (`segment_for_fts()` in `myrag/cjk.py`); `chunks_fts` + sync triggers recreated against it; automatic migration with rebuild on open.
- [x] **Keyword-only retrieval mode** — `embedding.mode: "none"`: fully offline retrieval via FTS5 + BM25, no embedding endpoint at all; `rag_query()` skips the embedder entirely.
- [x] **Hash-fallback no longer pollutes ranking** — while `hash_fallback: true`, retrieval runs keyword-only instead of fusing byte-pattern noise vectors into RRF. Re-ingest required after switching to a real model.
- [x] **Local batch encoding uses the cache** — only cache-missed texts are encoded, so re-ingesting unchanged chunks costs zero CPU inference. OOM handling simplified to recursive bisection.
- [x] **Section-path context in chunk embeddings** — embed input is `"A > B\n<body>"`; stored text unchanged. Existing stores need re-ingest to benefit.

### Done (2026-08-22)

- [x] **Migrate chunks table to sqlite-vec `vec0` KNN virtual table** — `chunks_vec` ANN mirror with cosine metric + METADATA filters; automatic `user_version` 0→1→2 migration with idempotent backfill; brute-force fallback for section-filtered queries and vec0 failures. (v0.8.0)
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
- 315 tests passing

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.
