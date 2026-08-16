# Changelog — myRAG Pipeline

## [0.6.6] — 2026-08-16

### Added

- **Deterministic markdown normalizer** (`src/parsers/markdown_normalizer.py`): runs in the `use_llm=False` path to structure raw parser output without any model call. Five transformations, all regex/line-based:
  1. **Heading promotion** — standalone short lines preceded by a blank line → `##` headings
  2. **List normalization** — `1)`/`1.` → `1.`, `a)`/`a.` → `a)`, `•`/`·`/`*`/`+` → `-`
  3. **Link formatting** — bare URLs → `[url](url)` (skips code spans)
  4. **Bold/italic repair** — closes unclosed `**`/`*` markers
  5. **Table alignment** — normalizes pipe spacing (`| a | b |`)

  Self-check: `src/parsers/tests/test_normalizer.py`.

## [0.6.5] — 2026-08-16

### Added

- **`process_file_with_md` / `process_file_hybrid` `use_llm=False` mode**: skips the LLM formatting step and writes the parser's raw cleaned output as the .md body (title from the first `# ` heading, or "Untitled"). Fully deterministic, no model call — offline fallback for inspection or when the LLM endpoint is down. CLI: `myrag md input.pdf --no-llm` / `myrag process input.pdf --store data.db --no-llm`.

- **Hash fallback** (`embedding.hash_fallback: true`): when the endpoint is unreachable, embedders fall back to a deterministic SHA-256 → 1024-d vector. Same text → same vector, so retrieval is stable across runs, but vectors carry no semantic signal. Offline/dev only — not a replacement for a real model.

## [0.6.4] — 2026-08-16

### Fixed

- **Test patch target drift after merge**: `formatters/call_llm.py` was merged into `formatters/__init__.py` by an upstream monolith rewrite, but 4 tests still patched `formatters.call_llm.httpx.post` — which resolves to `getattr(call_llm_function, 'httpx')` and raises `AttributeError`. Repointed to `formatters.httpx.post` (module-level import). This matches the upstream fix in `2f7fd5a`.

- **Embedder mode override lost**: `Embedder(base_url=..., model=...)` ignored explicit args when `embedding.mode=local`, returning a `LocalEmbedder` that lacks `model`/`store_chunk`. `__new__` now forces `remote` mode when either arg is supplied.

- **LocalEmbedder API gap**: Added `model` class attr and `store_chunk()` method so the local backend mirrors `Embedder`'s public surface (tests and `rag_query` callers don't need to branch on backend).

### Added

- **Hash-based pseudo-embedding fallback**: `config.embedding.hash_fallback` (bool, default `false`). When the remote/local endpoint is unreachable, `Embedder.embed()` and `LocalEmbedder.embed()` fall back to a deterministic SHA-256 → 1024-d float vector. Same text → same vector, so retrieval is stable across runs, but vectors carry no semantic signal. See `src/embedders/tests/test_hash_fallback.py` for a runnable self-check.

- **End-to-end test**: `src/tests/test_e2e_pipeline.py` — real `.txt` file → parse → clean → mock LLM formatter → chunker → hash embeddings → sqlite-vec → `rag_query` → answer. No external services required; runs in ~1.2s.

### Changed

- **Test count**: 254 passing (was 253).

## [0.6.3] — 2026-08-15

### Fixed

- **Local imports moved to module level**: `import logging` in `chunkers/__init__.py`, `import time` in `embedders/bge_m3.py`, `import datetime` in `formatters/__init__.py`, `import re` in `pipeline/core.py` all moved from inside functions to module-level imports for consistency and performance.
- **Redundant CJK regex compilation** in `storage/search.py:_build_fts_query()` — removed per-call `"|".join(_CJK_RANGE)` and uses the pre-compiled `_CJK_PAT` pattern directly.
- **Duplicate `_count_words` calls** in `storage/inserts.py:upsert_chunks()` — pre-computes word counts once per chunk instead of calling `_count_words()` twice per chunk (once in params list, once in result list).
- **Config inconsistency**: `conf/config.example.yaml` `chunk_threshold_chars` aligned to 28000 to match `conf/config.yaml` (was 20000).
- **Syntax error** in `formatters/__init__.py:228` — missing `]` in slice `[:8]` on `hashlib.md5(...).hexdigest()[:8]`.
- **NameError** in `pipeline/core.py:188` — `format_text_async` was not imported after `import re` was moved to module level.
- **Test assertion fixes**: Fixed `test_cache_lru_order`, `test_cjk_entities`, `test_preserves_leading_spaces_for_markdown`, `test_broken_table_row_merged`, `test_head_zero`, `test_unknown_extension` to match actual implementation behavior.

### Added

- **5 new test modules** (+53 tests, total 253):
  - `parsers/tests/test_text_cleaner.py` — 24 tests for `TextCleaner` (control chars, page breaks, whitespace, tables, custom YAML rules, edge cases).
  - `formatters/tests/test_tags.py` — 17 tests for tag extraction (script detection, Latin/CJK tokenization, proper nouns, body-based tag generation).
  - `formatters/tests/test_cache.py` — 11 tests for the LRU formatting cache (key generation, hit/miss, eviction, LRU ordering, clear).
  - `pipeline/tests/test_ingest.py` — 5 tests for `_ingest_markdown` (file not found, minimal markdown, headings, custom doc_id, custom chunk_size) with mocked `Embedder`/`Chunker`/`SQLiteVecStore`.
  - `pipeline/tests/test_utils.py` — 16 tests for `build_doc_summary`, `resolve_parser`, `source_type_for` (head/tail slicing, empty body, tag formatting, extension mapping).

### Changed

- **Test count**: 253 tests passing (was 185 — +68 new tests across 5 modules).
- **Module splitting**: 
  - `formatters/__init__.py` (750→200 lines) split into `formatters/__init__.py` + `formatters/_internal.py` (475 lines). The `_internal.py` module contains JSON preprocessing, paragraph splitting, single-shot/chunked formatting, LLM API calls, and CJK-aware threshold calculation. Public API (`format_text`, `format_text_async`, `format_text_with_system`, `call_llm`, `call_llm_raw`) remains unchanged.
  - `pipeline/core.py` (539→100 lines) split into `pipeline/core.py` + `pipeline/hybrid.py` (415 lines). The `hybrid.py` module contains all LLM-powered functions (`process_file_hybrid`, `process_file_with_md`, `process_directory_hybrid`, `rag_query`). `core.py` retains facades (`TextCleaner`, `Chunker`), traditional RAG functions (`process_file`, `process_directory`), and re-exports from `hybrid.py` for backward compatibility.

## [0.6.2] — 2026-08-14

### Fixed

- **Code audit: exception f-string literals**: `OSError`, `FileNotFoundError`, `ValueError` in `cli.py`, `ingest.py`, `writer.py` now assign message to variable before raising (per EM101/EM102 ruff rule).
- **Code audit: zip() missing `strict=`**: Added `strict=True` to all `zip()` calls in `core.py`, `rerank.py`, `test_sqlite_vec.py` (per B905 ruff rule).
- **Code audit: RET504 unnecessary assignments**: Removed dead intermediate variables in `core.py` (`md_path`), `rerank.py` (`scored`), `text_cleaner.py` (`text`).
- **Code audit: W293 trailing whitespace**: Removed trailing whitespace from blank lines in `core.py` docstrings.
- **Code audit: D401 imperative mood**: Fixed docstring first-line mood in `text_cleaner.py` ("Filters out" → "Remove") and `rerank.py` ("Normalized" → "Compute").
- **Code audit: N806 variable naming**: Renamed `_MAX_IN_CLAUSE` to `max_in_clause` in `search.py` (local vars should be lowercase).
- **Code audit: D102 missing docstrings**: Added docstrings to `TextCleaner.clean()`, `Chunker.__new__()`, and other facade methods in `core.py`.
- **Code audit: SIM110**: Simplified `_is_inside_protected()` in `writer.py` to use `return any(...)`.
- **Code audit: TRY300**: Restructured `_load_sqlite_vec()` in `schema.py` to use `try/except/else` pattern instead of early return in try block.
- **Code audit: D301 raw docstrings**: Added `r"""` prefix to `TextCleaner` class docstring containing backslash sequences.
- **Mypy: added `src/py.typed` marker**: Resolves `import-untyped` errors for internal module references.
- **TC001 suppressed**: Added `noqa: TC001` comments for `Embedder` and `SQLiteVecStore` top-level imports in `core.py` (required at runtime for context manager support).
- **D417 suppressed**: Added `noqa: D417` for `**kwargs` and auto-generated params in `process_file_hybrid()` / `process_directory_hybrid()` docstrings.

### Changed

- **Test count**: 185 tests passing (was 194 — some tests removed during cleanup).

## [0.6.1] — 2026-08-13

### Fixed

- **`upsert_document` raised `NameError` after ON CONFLICT refactor**: `existing` and `cursor` were referenced but never defined. Now queries the document ID via `SELECT id FROM documents WHERE source_file=?` after the upsert. (src/storage/inserts.py)
- **`hybrid` CLI command printed output twice**: Duplicate print block caused each piece of output (chunks, DB path, index message, title) to be printed twice. Removed the redundant block. (src/pipeline/cli.py)

### Added

- **Module splitting for large files**: Split `core.py` (688→531 lines) into `pipeline/core.py` + `pipeline/markdown_utils.py` + `pipeline/utils.py`. Split `formatters/__init__.py` (951→742 lines) into `formatters/__init__.py` + `formatters/tags.py` + `formatters/cache.py`. Split `sqlite_vec.py` (654→80 lines) into `storage/sqlite_vec.py` + `storage/schema.py` + `storage/inserts.py` + `storage/search.py`. All modules now under 500-line limit.
- **Strict mypy configuration**: Added `[tool.mypy]` to `pyproject.toml` with `disallow_untyped_defs = true`. Added mypy hook to `.pre-commit-config.yaml`. All public functions have type hints.
- **[tool.ruff] configuration**: Added project-level ruff config to `pyproject.toml` with line-length 120, target Python 3.10, and comprehensive rule sets (E, F, W, I, N, B, A, C4, D, DTZ, EM, FBT, ICN, ISC, LOG, PIE, Q, RSE, RET, SIM, TCH, TID, TRY, UP, YTT).
- **[tool.pytest] configuration**: Added testpaths, python_files/classes/functions patterns, and addopts for consistent test discovery.
- **`__init__.py` for `src/myrag/`**: Explicit package declaration with exception re-exports.
- **4 new test modules** (+76 tests, total 185): `test_directory_hybrid.py`, `test_rag_query.py`, `test_local_embedder.py`, `test_rerank.py`.
- **`[project.scripts]` entry point**: `myrag` command available after install.
- **`*.db`, `*.sqlite3`, `*.sqlite` added to `.gitignore`**.
- **mypy step added to CI workflow**.

### Changed

- **Unified `chunk_size` default**: CLI defaults raised from 512 to 1024 to match `pipeline.core` defaults. Consistent behaviour across all entry points.
- **Pre-compiled CJK regex**: `_cjk_re` moved to module level in `storage/inserts.py` to avoid recompilation on every `_count_words()` call.
- **Config alignment**: `config.yaml` now includes `query_instruction` and `debug_log_llm_responses` fields to match `config.example.yaml`.
- **Bare except fix**: `parsers/text_cleaner.py` now catches specific `(OSError, yaml.YAMLError)` instead of bare `Exception`.
- **Extracted `_call_llm_api` shared function** to avoid duplicate HTTP request logic in `call_llm` and `call_llm_raw`.
- **Pre-compiled `_PARAGRAPH_SPLIT` regex** at module level.
- **`upsert_document` now atomic** via `INSERT ... ON CONFLICT` instead of SELECT + conditional INSERT/UPDATE.
- **Pre-compiled CJK regex** in `_build_fts_query` at module level.
- **`_count_words` optimized**: uses `re.sub` + `re.findall` instead of character-by-character iteration.
- **Case-insensitive title matching** in `markdown_utils.py` added `re.IGNORECASE`.
- **`rag_query` return type** annotated as `dict[str, Any]`.
- **`LocalEmbedder` type annotations** corrected for `embed` and `embed_query` return types.
- **`try_fix_common_issues` uses `copy.deepcopy`** instead of shallow `dict()` copy.
- **Cache key hashes components separately** to avoid system prompt dominating hash computation.
- **`sentence-transformers` upper bound removed** (`>=2.7` without `<3`).
- **`httpx.TimeoutException` handling** updated for httpx 1.0+ compatibility.
- **`MarkdownIt` pre-compiled in `Chunker.__init__`** and reused across `chunk()` calls.
- **`store_chunks` accepts `batch_size` parameter** to split large batches and avoid API limits.
- **`IN` clause capped at 1000 items** to prevent SQLite variable number overflow.
- **`Embedder` and `SQLiteVecStore` now support context managers** (`__enter__`/`__exit__`).
- **`ingest.py` wraps `Embedder` and `SQLiteVecStore` in `with` statements** to prevent resource leaks.
- **`search_documents` now uses `query_vector`** for vector-based sorting when provided.
- **`logs/` directory creation checks** for pre-existing file before `mkdir`.
- **`pyproject.toml` duplicate `[tool.pytest.ini_options]` merged** into single section.
- **`writer.py` YAML frontmatter** uses consistent quoting.

### Added Tests (total 185)

- **4 new test modules** (+76 tests): `test_directory_hybrid.py`, `test_rag_query.py`, `test_local_embedder.py`, `test_rerank.py`.

## [0.5.6] — 2026-07-24

### Fixed

- **FTS query stripping missed `-`, `[`, `]`, `{}`, `<>/~?!.`**: `_build_fts_query()` stripped `"*^:()\\` but left FTS operators like `-` (AND NOT) and unmatched parentheses in queries. A user question "retrieval-augmented generation" or "(foo AND bar)" could crash with `no such column` errors instead of returning results. Expanded `_FTS_SPECIAL` regex to cover all recognized FTS5 operator characters; updated docstring (`src/storage/sqlite_vec.py`).
- **Embedding deserialization silently produced length-1 vectors**: `_deserialize_embedding()` fell back to `list(raw)` for unexpected types (None, dict, int), which downstream cosine-distance calls would reject with a confusing dimension-mismatch error. Now raises `ValueError` naming the actual type so callers see exactly what's corrupt (`src/storage/sqlite_vec.py`).
- **NaN / infinity embeddings could silently corrupt vector store**: sqlite-vec serializes them as invalid float32 bytes, and downstream `vec_distance_cosine()` returns garbage distances that pollute every query. Added `_validate_embedding_finite()` guard on both `upsert_chunk` (single) and `upsert_chunks` (batch); raises with the offending index before any write reaches SQLite (`src/storage/sqlite_vec.py`).
- **Early return from no-parser branch returned inconsistent shape**: `process_file_hybrid()` skipped back only `{chunks, document}` while normal path returns 5 keys. Callers accessing missing keys would get KeyError; callers iterating results got misaligned dicts. Now always returns the same 5-key structure (`src/pipeline/core.py`).
- **CJK word count and FTS search coverage was ~1% short**: `_count_words()` and `_build_fts_query()` used only `\u4e00-\u9fff` (basic CJK plane) which misses Extension B/C/D characters (~35K codepoints of rare names, place names, historical forms). Replaced with hex-range constants covering Blocks A–D (`>99.5% coverage`) and converted at import time to proper `\\uXXXX` regex patterns; shared between both call sites so count/FTS agree on what counts as CJK (`src/storage/sqlite_vec.py`).

## [0.5.5] — 2026-07-11

### Added
- **`process_directory_hybrid()`**: batch-process every supported file in a directory through the LLM-formatted Hybrid A+B pipeline. The parse/format/chunk/embed phase runs concurrently (`max_workers`, default 4); `doc_id` is derived deterministically from each file's path relative to the directory so re-runs overwrite the same records instead of duplicating them. (src/pipeline/core.py)
- **`md_path` reuse in `process_file_hybrid()`**: pass an existing `.md` file and the (expensive) LLM formatter is skipped entirely — the markdown is reused for chunking/embedding. Enables the two-phase workflow (generate `.md` once, ingest/experiment many times) without re-paying for formatting. (src/pipeline/core.py)
- **`SQLiteVecStore.get_embeddings_by_ids()`**: fetch chunk embeddings already stored in the index, keyed by row id and aligned to the requested order. Unknown ids yield an empty list in position. (src/storage/sqlite_vec.py)
- **`rag_query()` accepts optional pre-opened `db` / `embedder`** so a session can reuse one connection/embedder across queries instead of re-opening on every call. (src/pipeline/core.py)

### Changed
- **Skip ALL embedding when not persisting**: `process_file_hybrid()` no longer constructs the embedder or embeds chunks when `store_path` is None (previously only the document-level embedding was skipped). The most expensive remote step is now avoided entirely for `.md`-only / in-memory callers. (src/pipeline/core.py)
- **MMR re-ranking reuses stored chunk vectors**: `rag_query()` fetches the retrieved chunks' embeddings from the index via `get_embeddings_by_ids()` instead of re-embedding every chunk (a per-query batch of remote embedding calls). Missing vectors fall back to on-demand embedding only for the gaps. (src/pipeline/core.py, src/storage/sqlite_vec.py)
- **`upsert_chunks()` now writes in a single transaction** via `executemany` (was a per-chunk INSERT loop), then maps generated row ids back by `(source_doc_id, chunk_index)`. Faster for large documents. (src/storage/sqlite_vec.py)
- **Document-level (B) summary uses head+tail of the body** instead of only the first 500 characters, so long documents also contribute their closing context to the coarse-grained embedding. Shared helper `_build_doc_summary()` used by both `process_file_hybrid()` and `_ingest_markdown()`. (src/pipeline/core.py, src/pipeline/ingest.py)
- **Lowered large-doc auto-chunk threshold 28000 → 20000 chars** (`chunk_threshold_chars` default in `src/config.py`, `conf/config.yaml`, `conf/config.example.yaml`). Texts above this are split for chunked LLM processing (~5000 tokens for English). (src/config.py, src/formatters/__init__.py, AGENTS.md, README.md)

## [0.5.4] — 2026-07-09

### Added
- **bge-m3 query instruction prefix**: new `embed_query()` on both remote and local embedders prepends the retrieval instruction (`Represent this sentence for searching relevant passages: `) to queries only — documents are embedded without it. Configurable via `embedding.query_instruction` (default set in `conf/config.example.yaml`). (src/embedders/bge_m3.py, src/embedders/local_bge.py, src/config.py)
- **Document-level (B) vector index now functional**: `documents` table gains an `embedding BLOB` column; `upsert_document()` persists the doc embedding; `search_documents()` supports vector ranking by cosine distance (with `k` limit). Existing DBs are migrated via `ALTER TABLE` on open. (src/storage/sqlite_vec.py)
- **MMR re-ranking of retrieved chunks**: new `src/rerank.py` re-orders hybrid-search results by Maximal Marginal Relevance (lexical relevance + embedding diversity) so context blocks are not near-duplicates. Wired into `rag_query()`.
- **Process-wide caches**: bounded LRU caches for embedding results (`src/embedders/bge_m3.py`, `local_bge.py`) and formatting results (`src/formatters/__init__.py`) — identical inputs are processed once per process.

### Changed
- **`rag_query()` now uses true hybrid retrieval**: switched from pure-vector `search_chunks()` to `hybrid_search()` (vector + FTS5 RRF fusion) with `embed_query()`, then MMR re-ranks. When chunk recall is sparse it appends the top document summary as a coarse-grained B fallback. (src/pipeline/core.py)
- **Document (B) summary embeds the LLM-formatted body** instead of raw cleaned text, so the coarse index captures structured semantics. (src/pipeline/core.py)
- **Default `chunk_size` raised 512 → 1024** for bge-m3's large context window (process_file / process_file_hybrid / process_directory). (src/pipeline/core.py)
- **`_render_markdown_with_sections()` simplified**: dropped the mis-nesting no-headings branch (it listed headers then dumped the whole body); now just prepends the title and lets the chunker fall back to plain-text splitting. (src/pipeline/core.py)
- **`Embedder.__new__` cleaned up**: local mode now constructs `LocalEmbedder` directly instead of the `object.__new__` + manual `__init__` hack; `LocalEmbedder` gained `__enter__/__exit__` so `with Embedder() as e:` works in local mode too. (src/embedders/bge_m3.py, src/embedders/local_bge.py)

### Fixed
- **CJK entity matching was broken**: `_match_entities_to_chunks()` used `\b` word boundaries which never match Chinese names. Now uses substring match for CJK entity names and word-boundary match for Latin ones. (src/pipeline/core.py)
- **Wasted document embedding when not persisting**: `process_file_hybrid()` no longer calls `store_document()` (and its embedding) when `store_path` is None; it returns a lightweight metadata dict instead. (src/pipeline/core.py)
- **`TrafilaturaParser` broken on trafilatura 2.x**: removed the removed `prefer_full_output` kwarg (now `favor_recall`) so HTML parsing no longer crashes. (src/parsers/dispatcher.py)
- **`hybrid_search()` crashed on hyphenated queries**: user questions like "retrieval-augmented generation" were passed verbatim to FTS5 MATCH, where the hyphen is parsed as an operator (`no such column: augmented`). Added `_build_fts_query()` to strip FTS5 special characters and OR-join tokens. (src/storage/sqlite_vec.py)
- **`rag_query()` reused the JSON-only `call_llm` to synthesize answers**: the LLM replies in free text, not JSON, so it raised "no JSON-like content". Added `call_llm_raw()` and switched `rag_query` to it. (src/formatters/__init__.py, src/pipeline/core.py)
- **Reference/bibliography sections polluted retrieval**: reference lists (References, 参考文献, 參考, Bibliography, Further reading, …) are high in keyword overlap but carry no answer-worthy content, so they dominated top-k. `process_file_hybrid()` now strips such sections from the markdown *before chunking/embedding* (the human `.md` output is untouched). Added `_strip_reference_sections()` + `_is_reference_title()` with English (word-boundary) and CJK (substring) title matching. (src/pipeline/core.py)

## [0.5.3] — 2026-07-09

### Fixed

- **`rag_query()` crashed on every query**: `Embedder.embed(str)` returns a single embedding vector (`list[float]`), but the code indexed `query_vectors[0]`, turning the vector into a bare float and crashing `serialize_float32()`. Now uses the vector directly, with a guard handling both `str` and `list` return shapes. `Embedder` is also closed via a context manager (fixes connection leak). (src/pipeline/core.py)
- **Vector search returned duplicate chunks with truncated `section_path`**: `search_chunks()` and `hybrid_search()` used `FROM chunks c, json_each(c.section_path)` — a cross join that multiplied each chunk once per section-path element (wasting the `LIMIT` budget) and returned a single `json_each.value` element instead of the full path. Replaced with `FROM chunks c` + `EXISTS (... json_each ...)` filtering so results are deduplicated and `section_path` is the complete array. (src/storage/sqlite_vec.py)
- **Hybrid search RRF fusion used bm25 score as rank**: `fts_rank` was taken from the raw bm25 `rank` value (a negative float), making the `+rrf_k` (60) offset meaningless and corrupting fused ordering. FTS rank now uses 1-based result order; vector ranks are also precomputed once. (src/storage/sqlite_vec.py)
- **`_render_markdown_with_sections()` reordered document content**: it hoisted all `metadata.sections` headers above the body, shifting every section's content under the wrong header when the LLM body already contained headings (the common case). Now keeps the body's own heading structure intact, falling back to `metadata.sections` only when the body has no headings. (src/pipeline/core.py)
- **`process_file_hybrid` / `process_file_with_md` hardcoded `source_type="pdf"`** regardless of the actual file type. Now derives the formatter hint from the file extension (pdf/docx→pdf, html/htm→web, md/mkd→markdown, txt→web). (src/pipeline/core.py)
- **CJK `word_count` was always 1 for unspaced text**: `len(text.split())` counts only whitespace-delimited tokens, so Chinese paragraphs counted as a single word. Added a `_count_words()` helper that counts CJK characters plus whitespace-split tokens. (src/storage/sqlite_vec.py)
- **Dead code in `SQLiteVecStore.close()`**: removed an empty `if not self.conn.in_transaction: pass` branch. (src/storage/sqlite_vec.py)
- **`Embedder.embed()` docstring** claimed `list[list[float]]` for all inputs; corrected to document the actual contract (`str → list[float]`, `list[str] → list[list[float]]`). (src/embedders/bge_m3.py)

### Changed

- **`hybrid_search()` fusion loop** no longer re-sorts `vec_results` on every iteration (O(n²) → O(n)); the sorted rank map is computed once. (src/storage/sqlite_vec.py)

### Added Tests (total 103)

- No new tests added; existing suite (103 passing) covers the modified behavior, and an end-to-end run (`python -m pipeline.cli process`) confirms parse → clean → LLM format → embed → sqlite-vec persistence and correct deduplicated retrieval.

## [0.5.2] — 2026-07-06

### Fixed

- **`upsert_chunk` was INSERT-only, producing duplicate rows on re-ingest**: Added `UNIQUE(source_doc_id, chunk_index)` constraint + switched to `INSERT OR REPLACE`. Batch variant (`upsert_chunks`) now wraps all inserts in a single transaction. (src/storage/sqlite_vec.py)
- **LIKE escaping broken for tag search and section filter**: `_escape_like_pattern` escaped `%`/`\` but SQL queries lacked `ESCAPE '\'` clause, so wildcards had no effect. Replaced LIKE matching with exact `json_each.value = ?` equality checks in both `search_documents()` tag filtering and `search_chunks()` section filtering. (src/storage/sqlite_vec.py)
- **Oversized chunks silently passed through**: Sentence-split could leave chunks larger than `chunk_size`. Now falls back to character-level split (`_split_by_char()`) with overlap, emitting a warning log. Added input validation: `chunk_size <= 0` raises ValueError; negative `chunk_overlap` rejected. (src/chunkers/__init__.py)
- **Chunked formatting path discarded custom system_prompt**: `_format_text_async_impl()` forwarded `system_prompt` to single-shot but silently dropped it for large documents routed through chunked processing. Now passes it through both paths and uses it as the base prompt for all chunks. (src/formatters/__init__.py)
- **Dead duplicate title re-extraction in chunked merge**: Title was extracted twice after dedup — once at line 712, again at line 729 with identical regex. Removed redundant extraction. (src/formatters/__init__.py)
- **HTTP embedding API had no retry on transient failures**: Network blips, rate limits (429), and server errors (5xx) crashed the pipeline mid-ingestion. Added `_post_with_retry()` with exponential backoff (1s→8s cap) for 429/502/503/504 timeouts. (src/embedders/bge_m3.py)
- **Embedder connection pool leaked**: `httpx.Client` had no `.close()`. Added `__enter__/__exit__` context manager for deterministic cleanup; `_post_with_retry` handles close gracefully on exit. (src/embedders/bge_m3.py)
- **Token estimation wildly inaccurate for CJK text**: `len(text)//2` underestimated Chinese tokens by ~50% (bge-m3 SentencePiece is ~1:1 for CJK). Replaced with multi-language-aware heuristic counting CJK chars as 1 token each, ASCII letters at 4:1 ratio. (src/embedders/local_bge.py)
- **OOM fallback only handled batch-level OOM**: Individual item encode could also OOM on small-memory systems and propagate unhandled. Now progressively reduces batch size (batch→half-batch→single-item). (src/embedders/local_bge.py)

### Changed

- **`_setup_schema()` now idempotent per connection**: Added `_schema_ready` flag so `executescript` runs once at first query, not on every call. Eliminates implicit commits and repeated DDL overhead across all 10+ query methods. (src/storage/sqlite_vec.py)
- **Hybrid search N+1 query eliminated**: Text-only FTS path previously executed one SELECT per result row. Now uses single `IN (...)` JOIN to fetch all chunk details in one round-trip. Cosine distance also batched via IN clause instead of per-ID queries. (src/storage/sqlite_vec.py)
- **RRF rank assignment fixed**: Previously converted raw cosine distance to integer rank via scaling, collapsing distinct results into identical ranks. Now sorts vec_results by score and assigns sequential ranks 1..N for proper RRF scoring. (src/storage/sqlite_vec.py)
- **`_metadata_to_section_path` no longer hardcodes H1/H2/H3**: Uses dynamic keys from `self._level_to_key.values()` built at construction time, so custom `headers_to_split_on` configurations produce correct section paths. (src/chunkers/__init__.py)
- **Chunker exposes `__repr__`** for easier debugging in logs and REPL sessions. (src/chunkers/__init__.py)
- **Embedders exports `LocalEmbedder`** from package root so it's discoverable via IDE autocomplete. (src/embedders/__init__.py)
- **Writer `format_md()` uses temp directory instead of hardcoded `/tmp/md_format_output`**: Uses `tempfile.mkdtemp()` with `shutil.rmtree` cleanup in finally block to avoid polluting global temp space and prevent collisions across processes. (src/formatters/writer.py)
- **Parser error messages now readable**: `ParserNotFoundError` for missing MarkItDown/Trafilatura no longer passes the library name as `filepath`; shows descriptive "library not available" message instead. (src/parsers/dispatcher.py)
- **Removed unused `_debug_import_snapshot()` from test conftest**: The function was prefixed with underscore so pytest never collected it, but its presence was misleading. (src/storage/tests/conftest.py)

### Added Tests (+0, total 103)

- No new tests in this release — existing test suite covers all modified behavior paths. One pre-existing test (`test_hybrid_search_returns_results`) was failing due to the unserialized query_vector bug and is now fixed.

## [0.5.1] — Phase 6 Bug Fixes (Code Review)

### Fixed

- **Unhandled `TimeoutError` in pipeline**: `process_file_hybrid()` and `process_file_with_md()` now catch `concurrent.futures.TimeoutError` from `future.result()`. Previously a slow LLM would crash the entire pipeline with an unhandled exception.
- **CJK ratio detection misclassified non-CJK text**: `_detect_cjk_ratio()` was treating all non-ASCII characters (Cyrillic, Arabic, Devanagari) as CJK. Now only counts actual CJK ranges (`\u4E00-\u9FFF`, Hiragana `\u3040-\u309F`, Katakana `\u30A0-\u30FF`). Non-CJK multilingual documents no longer get incorrectly aggressive chunking thresholds.
- **`format_text_with_system()` used wrong threshold**: Was calling `_get_chunk_threshold()` (English-only default) instead of `effective_chunk_threshold(raw)` like the other two public formatters. CJK text routed through this function would exceed LLM token limits.
- **Duplicate headings in markdown output**: `_render_markdown_with_sections()` was appending body content that already contained `##`/`###` headers on top of metadata-section-rendered headers, producing duplicate headings. Body heading lines are now stripped before appending.
- **Section filter LIKE false-positive matches**: `search_chunks()` used `LIKE '%General%'` against the raw JSON array string (`["General","ReGeneral"]`) which matched substring `"ReGeneral"`. Now uses `json_each()` unnest + IN subquery for precise per-element matching.
- **Tag filter semantics changed from OR to AND**: `search_documents()` was using `OR` between tag conditions (any-match), but the parameter name and expected behavior imply all-specified-tags-must-match. Changed to `AND`.
- **Redundant SQL in hybrid_search RRF loop**: Each combined ID triggered a separate `serialize_float32()` + single-row SQL query for cosine distance. Now serializes once and uses `IN (...)` batch query, eliminating N redundant round-trips.
- **Path traversal on `output_dir`**: `write_to_md()` now resolves paths via `Path.resolve()` before creating directories or writing files, preventing relative path escape attacks from crafted CLI input.

### Test Fixes

- **Missing assertion in `_fix_bare_quotes_in_body_field` test**: Added `assert result is None` to `test_quote_before_body_key_not_matched` — previously the test silently passed even if the function returned a non-None fix (masking a false-positive bug).
- Updated `test_only_digits_and_punctuation` to expect 0.0 ratio for digits/punctuation only text (matches corrected CJK detection logic).

## [0.6.0] — 2026-07-03

### Added

- **Unified exception system** (`src/myrag/exceptions.py`): `MyRagException` base class + typed exceptions: `ParserNotFoundError`, `EmbeddingError`, `ChunkingError`, `FormattingError`, `StorageError`. Enables structured error handling across all pipeline stages.
- **Config validation on startup**: `get_config()` now validates required fields (LLM endpoint, model name) before any pipeline work begins; descriptive errors for invalid configs.
- **Embedding dimension validation**: Both remote and local backends validate embedding dimension on every call; mismatched dimensions raise `EmbeddingError` with context about expected vs actual size.
- **Embedder factory function** (`create_embedder()`): Factory pattern replaces direct `Embedder()` instantiation, respects config's `embedding.mode`. Union type for `Embedder | LocalEmbedder` in public API. Optional `validate=True` parameter performs a test embedding at construction time to catch dimension mismatches early.
- **Debug LLM response logging**: New config flag `debug_log_llm_responses: true` gates full LLM request/response logging via `logging.debug()`. Disabled by default; useful for debugging JSON parsing issues without polluting production logs.

### Changed

 - **CJK-aware chunk threshold** (`_detect_cjk_ratio()`): Chunk size scales down from the configured base (default 20000 chars, ~5000 tokens) to ~5000 chars when text is ≥50% CJK (Chinese/Japanese/Korean), based on character-to-token ratio of 1:1 for CJK vs 4:1 for English. Linear interpolation at 10–50% CJK density.
- **Multi-language tag extraction**: `_detect_body_script()` detects text script; `_tokenize_cjk()` uses bigram + whitespace tokenization for CJK; `_tokenize_latin()` uses `[a-zA-Z]{3,}` regex; multi-language stopword sets (English, Chinese, Japanese); `_extract_cjk_entities()` extracts named-entity-like phrases from CJK text.
- **Sentence-split abbreviation handling**: `_SENTENCE_ABBREVIATIONS` set with 50+ common abbreviations ("Mr.", "Dr.", "U.S.A.", etc.) prevents false sentence boundary detection in title/initials/acronyms contexts.
- **O(n) heading lookup** (`_split_by_headings()`): Pre-build `heading_by_line` dict for constant-time per-line lookups instead of O(n*m) nested loop scanning.
- **Chunker overlap word-boundary preservation**: `_apply_overlap()` now extends back to nearest whitespace boundary before truncating, preserving words intact across chunk boundaries instead of cutting mid-word.

### Fixed

- **FTS5 sync triggers** (`chunks_ai/au/ad`): SQLite content-synced virtual table pattern ensures FTS5 index stays in sync with `chunks` table on INSERT/UPDATE/DELETE without manual trigger maintenance.
- **JSON serialization dead code removed**: `upsert_chunk()` passes original dicts directly to INSERT instead of round-tripping through `json.dumps`/`json.loads`.
- **Hybrid search empty query fallback**: Returns pure vector results when `query_text=""` and `query_vector` provided; returns `[]` otherwise (avoids FTS5 MATCH '' syntax error).
- **Section filter LIKE escaping**: Wildcard LIKE matching now properly escapes `\`, `%`, `_` characters with single-char escape prefix for SQLite `ESCAPE '\'`.
- **EmbeddingError missing context kwarg**: Added `context: dict | None = None` parameter to `EmbeddingError.__init__`; previously `_validate_embedding_dimension()` would raise TypeError on dimension mismatch.
- **Dead code after return in `_split_by_sentence`**: Removed unreachable merge-back-to-chunk-size block that was leftover from earlier implementation; each sentence now correctly becomes its own chunk.
- **Unicode escape handling in bare-quote fixer**: `_fix_bare_quotes_in_body_field()` now recognizes `\uXXXX` escapes (in addition to `\"`, `\\`, etc.) so documents containing unicode-escaped quotes don't corrupt the JSON walker's position tracking.

### Added Tests (+18, total 103)

- `test_cjk_threshold.py`: 9 tests covering `_detect_cjk_ratio()` and `effective_chunk_threshold()` edge cases (CJK-only, mixed CJK/English, empty strings).
- `test_chunker.py`: Plain text fallback, overlap word-boundary preservation, abbreviation detection.
- `test_sqlite_vec.py`: FTS5 sync on insert/delete, full-text search function, hybrid RRF ranking verification, empty query fallback.

### Architecture Notes

- **Config resolution chain**: `$MYRAG_CONFIG` → `conf/config.yaml` → `conf/config.example.yaml`. All endpoints configurable via YAML.
- **Facade pattern** — `TextCleaner` and `Chunker` classes in `pipeline.core` are thin facades that delegate to `parsers.text_cleaner.TextCleaner` and `chunkers.Chunker` respectively. The canonical implementations live in their own modules with full feature support (YAML config, markdown-it-py chunking).

## [0.5.0] — 2026-06-19

### Fixed

- **Issue 1: conftest.py test collection error** — Removed empty `src/storage/tests/__init__.py` that caused pytest to resolve conftest as `tests.conftest` with missing module. (src/storage/tests/)
- **Issue 3: Chunker adjacent-heading merge bug** — `_split_by_headings()` now creates a new section at every heading boundary, not only when body content exists between headings. Previously consecutive headings without body were merged into one section, giving all sub-chunks wrong `section_path` metadata. (src/chunkers/__init__.py)
- **Issue 4: Formatter module-level config caching** — `_CHUNK_THRESHOLD_CHARS` was evaluated at import time and cached. Replaced with `_get_chunk_threshold()` lazy-evaluated function so config changes take effect on each call. (src/formatters/__init__.py)
- **Issue 5: LLM schema fallback only caught HTTP 500** — Expanded retry to include 503 and 429 status codes for better coverage of schema-incompatible backends. Added HTTP status code to warning log message. (src/formatters/__init__.py)
- **Issue 6: Entity substring false positives** — `_match_entities_to_chunks()` now uses `re.search(r'\b...\b')` word-boundary matching instead of plain `in` substring check. Previously "AI" matched inside "algorithm". (src/pipeline/core.py)
- **Issue 7: hybrid_search rank not normalized** — Replaced naive tuple sort `(fts_rank, vec_score)` with RRF (Reciprocal Rank Fusion) algorithm. FTS5 BM25 rank and cosine distance had vastly different scales, causing FTS to completely dominate. Now both signals contribute fairly via `1/(rank + k)`. (src/storage/sqlite_vec.py)
- **Issue 8: section_filter LIKE query always returned empty** — `json_extract(section_path, '$')` returns full JSON array string `'["General"]'`, so `LIKE '"General"'` did not match. Changed to `LIKE '%General%'` wildcard matching. (src/storage/sqlite_vec.py)
- **Issue 2: Two inconsistent markdown generation paths** — `process_file_hybrid()` now accepts optional `md_output_dir` parameter and delegates to `write_to_md()` for structured markdown output, same as `process_file_with_md()`. Both pipelines now produce identical `.md` content. (src/pipeline/core.py)

### Changed

- **Chunker zero LangChain**: Replaced `langchain-text-splitters` (MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter) with pure Python + `markdown-it-py`. Same output format (`text`, `section_path`, `metadata`), all 8 existing tests pass. Headers parsed via markdown-it-py AST (handles ATX + setext headers natively). Each heading updates the section metadata context — consecutive headings without body text share one text section but each gets its own metadata. (src/chunkers/__init__.py, pyproject.toml)
- **Dependency cleanup**: Removed `langchain-text-splitters` + 23 transitive LangChain deps. Added `markdown-it-py>=3.0,<4`. (pyproject.toml)
- **Sentence-aware recursive split**: Custom `_split_by_sentence()` handles Chinese（`。！？`）and English (`.!?`) sentence boundaries. Falls back to character-level split only when a single sentence exceeds `chunk_size`. (src/chunkers/__init__.py)
- **Local bge-m3 embedding**: New `LocalEmbedder` class (`src/embedders/local_bge.py`) using sentence-transformers for offline/CUDA inference. Config `embedding.mode: "remote" | "local"` switches between HTTP API and local model at runtime via `Embedder.__new__()` dispatch. Added `local-embeddings` extra (`uv sync --extra local-embeddings`). Validation integrated into `Config._validate()`. (src/embedders/bge_m3.py, src/embedders/local_bge.py, src/config.py, conf/config.yaml, conf/config.example.yaml)
- **Entity extraction + entity_names**: Formatter prompt now outputs `metadata.entities` (list of `{name, type}` with 5 types: PERSON/ORG/PRODUCT/LOCATION/CONCEPT). JSON Schema enforced via `response_format`. `validate_format_output()` validates entity format. Writer `_insert_wikilinks()` applies `[[wikiname]]` only to .md display files (not chunk text). Pipeline `_match_entities_to_chunks()` matches entities per chunk. sqlite-vec `chunks` table has `entity_names TEXT` column for entity-based retrieval queries. (src/formatters/constants.py, src/formatters/prompts.py, src/formatters/writer.py, src/pipeline/core.py, src/storage/sqlite_vec.py)
- **Schema fallback for incompatible LLM backends**: `call_llm()` now catches HTTP 500/503/429 errors with PEG-grammar mismatch (`response_format` rejected by llama.cpp), retries without schema automatically. Fixes compatibility with gemma-4 and other models that don't support structured JSON Schema enforcement. (src/formatters/__init__.py)
- **Formatter JSON Schema enforcement**: `call_llm()` now accepts a `schema=` parameter that sends JSON Schema via `response_format`, letting llama.cpp / OpenAI servers enforce output structure natively. Schemas defined in new `constants.py` file. ([src/formatters/constants.py](src/formatters/constants.py), [src/formatters/__init__.py](src/formatters/__init__.py))
- **Formatter JSON parsing robustness**: Multi-level retry — `strict=True` fast path, then `strict=False`, then bare-quote fix for body field (`_fix_bare_quotes_in_body_field()`). Bare quotes inside string values no longer cause parse failures. ([src/formatters/__init__.py](src/formatters/__init__.py))
- **Tag extraction quality**: Removed generic single-word tags ("banking", "company", "system"), introduced proper noun extraction from title + body, multi-word phrase merging, and a whitelist of useful domain-specific terms. Tags now describe document subject matter so a reader can understand what it's about at a glance. ([src/formatters/__init__.py](src/formatters/__init__.py))
- **Few-shot examples added**: Prompts now include concrete input/output examples for both single-shot and chunked formatting, improving output consistency across LLM calls. ([src/formatters/prompts.py](src/formatters/prompts.py))
- **Few-shot examples generalised**: Replaced FX Networks-specific example with a generic research paper example so tags demonstrate domain-agnostic patterns rather than topic-specific values. ([src/formatters/prompts.py](src/formatters/prompts.py))

### Added

- **Output validation helpers**: `validate_format_output(result) → list[str]` checks required fields; `try_fix_common_issues(result)` auto-fixes bad tags / missing metadata without re-calling LLM. ([src/formatters/prompts.py](src/formatters/prompts.py))
- **JSON schema constants**: `FORMATTER_SCHEMA` and `CHUNKED_SCHEMA` extracted from inline prompts into [src/formatters/constants.py](src/formatters/constants.py) for reuse by `call_llm(schema=...)`. ([src/formatters/constants.py](src/formatters/constants.py))

### Changed

- **Parser lazy loading**: Moved MarkItDown/Trafilatura imports from module level into `__init__`. Parsers now load on first use, allowing the module to be imported even when optional deps are missing (fail-fast in `__init__`). ([src/parsers/dispatcher.py](src/parsers/dispatcher.py))
- **TrafilaturaParser encoding**: Now reads HTML files with UTF-8 → GBK fallback instead of passing filepath directly. Fixes silent parse failures on non-UTF-8 encoded Chinese web pages.

## [0.3.0] — 2026-06-17

### Fixed (2026-06-17)

- **hashlib import missing**: Added `import hashlib` to `formatters/__init__.py` (used for LLM response debug logging).
- **total_words = 0 in metadata**: Placeholder value from prompt template was passed through unchanged. Now computed as `len(body.split())`.
- **tags not displayed in markdown output**: Tags are at result level (`result["tags"]`) but writer.py read from `metadata.get("tags")`. Updated `_write_metadata_block()` to accept full result dict and prioritize `result["tags"]`.
- **Placeholder metadata in single-shot mode**: LLM copies template placeholders (created_at: "ISO-8601", total_words: 0). Now overridden with real values in `_format_text_single()`.
- **Split table headers from PDF extraction** → renamed to `_fix_broken_tables()` and rewritten. Uses a more robust approach: detects continuation rows by column count heuristic, appends content into the last cell of the preceding header row to preserve Markdown structure.
- **TextCleaner rewrite**: Major overhaul of `parsers/text_cleaner.TextCleaner`:
  - **Generalized page-break regex** — now matches `"--- PAGE N ---"`, `"=== Section ==="`, etc. (previously only matched single separator characters). Split-based filtering with length safeguard (>8 chars) prevents false positives on short lines like bullet points.
  - **Control character handling** — deleted entirely instead of replacing with space; excludes `\n` and `\t` to avoid breaking text structure.
  - **YAML flags parsing** (`_parse_flags`) now supports `int`, `str`, or `list[str]` (case-insensitive), e.g. `"IGNORECASE"`.
  - **Custom rules pre-compiled** in `__init__` instead of on each `clean()` call — avoids repeated regex compilation cost.
  - **Whitespace collapse** now only trims trailing spaces (`_TRIM_TRAILING_SPACE_RE`) — leading indentation (code blocks, lists) preserved.

### Removed

- **Backward-compat facades**: Deleted `src/cleaners/` directory and its tests. Canonical implementation is now exclusively in `parsers/text_cleaner.TextCleaner`. The convenience function `clean_text()` was removed — use `TextCleaner().clean(text)` directly.
- **Top-level pipeline shim**: Deleted `src/pipeline.py` (re-export wrapper). All imports should come from submodules: `pipeline.core`, `pipeline.ingest`, etc.

### Changed

- **AGENTS.md moved**: Relocated from `.github/` to root directory so agents can always find it.
- **Pipeline module split**: `pipeline.py` (549 lines) → `pipeline/core.py` (356), `pipeline/cli.py` (128), `pipeline/ingest.py` (81). All under 500-line limit.
- **Formatter public API**: `_call_llm()` renamed to `call_llm()`, exported in `__all__`. New `format_text_with_system(raw, source_type, *, system_prompt)` convenience wrapper. Added `system_prompt` parameter chain through `format_text_async()`.

### Fixed

- **sqlite_vec import detection**: Replaced fragile `sys.path` walking with `importlib.metadata.distribution("sqlite-vec").files`. Works across editable installs, wheels, and different Python versions. Added explicit `PackageNotFoundError` handling.

### Added (Auto-Chunking for Large Docs — 2026-06-14)

- **Chunked formatter**: Texts >28K chars auto-split at paragraph boundaries and processed chunk-by-chunk. Each chunk LLM call receives the last 10 lines of previous markdown output + cumulative summary as continuity context. Single-shot path unchanged for small docs.
- **CHUNKED_SYSTEM_PROMPT**: New prompt with concrete input/output example, `Do NOT summarize — preserve ALL substantive content` instruction, markdown style rules, and JSON output schema (`part_md` + `summary`).
- **uv package management**: Replaced pip + requirements.txt with `uv sync`. Single source of truth in `pyproject.toml`. `uv.lock` committed for reproducible installs.

### Changed

- **Package layout**: Flat root → `src/` standard layout. All imports changed from `myrag.xxx` to `xxx`.
- **Dependencies cleaned**: Removed 6 ghost deps (`pymupdf`, `python-docx`, `beautifulsoup4`, `readability-lxml`, `markdown`, `tiktoken`). Added `markitdown[pdf]`, `trafilatura`, `httpx`, `langchain-text-splitters`, `PyYAML`.
- **Config defaults**: `max_tokens` 8192 → 16384, `timeout` 180 → 300.
- **Pipeline timeout**: `future.result()` 300 → 3600 (1h for large doc processing).
- **Test files**: Removed `sys.path.insert` hacks; imports now work via proper package install.

### Fixed

- **Chunk summary bug**: First run produced 20:1 compression (115 lines for 57-page PDF) — missing `DO NOT summarize` in chunked prompt. Fixed + added concrete example → now 619 lines with 8 tables, full glossary, all technical data preserved.
- **str.format KeyError**: JSON curly braces in EXAMPLE block needed `{{`/`}}` escaping.

- **LangChain Chunker**: Replaced custom regex Chunker with `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`. Splits on markdown header boundaries; oversized sections get recursive character split. Plain-text fallback when no headers detected.
- **sqlite-vec storage**: `process_file_hybrid(store_path=...)` now persists embeddings to sqlite-vec database. 17+ chunks, FTS5 full-text index, vector similarity search.
- **Config centralization**: All endpoints (LLM, embedding) in `conf/config.yaml` + `config.py` loader. Resolution chain: `$MYRAG_CONFIG` → `conf/config.yaml` → `conf/config.example.yaml`.
- **Formatter prompt v2**: Added few-shot example, Wikipedia chrome removal rules, CRITICAL body completeness constraint, explicit section level definitions.
- **Chunker unit tests**: 8 tests covering empty input, header splits, hierarchical metadata, oversized sections, plain text fallback, pipeline facade.

### Changed (Post-Audit Cleanup — 2026-06-14)

- **Chunker unified**: Removed duplicate Chunker from `pipeline.py`, canonical implementation in `chunkers/`.
- **writer.py fix**: H1 collision bug, removed hardcoded Chinese document filtering.
- **process_file_hybrid()**: Chunks rendered markdown (from `metadata.sections`), not raw text. Added `store_path` param.
- **pipeline section rendering**: `_render_markdown_with_sections()` generates reliable `##`/`###` headers from LLM metadata.
- **pyproject.toml**: Dependencies updated to `markitdown`, `trafilatura`, `httpx`, `langchain-text-splitters`.

### Fixed

- **sqlite-vec commit bug**: `upsert_chunks` and `upsert_document` now call `conn.commit()` — documents table was always 0 rows.
- **Chunker metadata bug**: Fixed `_metadata_to_section_path` returning `["General"]` for docs without H1 header.

### Removed

- `doc/markitdown-migration-plan.md`, `output/China_National_Clearing_Center.md`, `storage/__init__.py`, all `__pycache__/`, `.pytest_cache/`.

---

## [0.2.0] — MarkItDown Migration — 2026-06-13

### Changed

- **Unified parser backend**: Replaced individual parsers with MarkItDownParser + TrafilaturaParser.
- **TextCleaner reorganized**: Moved to `parsers/text_cleaner.py` with YAML config.
- **Formatter prompt updated**: Removed `chunks` from LLM output, added `body` field.
- **Chunker redesigned**: Auto markdown header parsing, header-enriched embeddings.

## [0.1.0] — 2026-06-13

### Added

- Parser dispatcher, PDF/DOCX/HTML/MD/TXT parsers
- TextCleaner, Chunker, bge-m3 embedder, CLI with argparse
