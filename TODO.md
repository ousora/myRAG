# myRAG — TODO (Persistent Task Tracker)

## ✅ Completed (2026-06-14 Session)

### Core Architecture
- [x] LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter chunking
- [x] sqlite-vec vector storage integrated (process_file_hybrid store_path=...)
- [x] All endpoints centralized in conf/config.yaml + config.py loader
- [x] Formatter prompt v2 with few-shot example + body completeness constraint
- [x] Plain-text fallback when no markdown headers detected

### Pipeline Fixes
- [x] process_file_hybrid() chunks rendered markdown (from metadata.sections)
- [x] sqlite-vec commit bug fixed (documents table was always 0 rows)
- [x] Chunker H1 metadata bug fixed (docs without H1 returned ["General"])
- [x] writer.py H1 collision + hardcoded Chinese filter removed

### Tests
- [x] Chunker: 8 unit tests (header split, metadata, oversized, plain-text fallback)
- [x] All 22 tests pass (chunkers 8 + formatters 9 + cleaners 5)

---

## 🚧 Pending

### Tests
- [ ] parser integration test — end-to-end from file to chunks
- [ ] embedder mock test — verify bge-m3 client without real endpoint

### Features
- [ ] Hybrid search API — expose SQLiteVecStore.hybrid_search() as CLI/HTTP endpoint
- [ ] Reranker integration — add cross-encoder reranking on top of vector search
- [ ] Batch processing — process_directory() with sqlite-vec storage
