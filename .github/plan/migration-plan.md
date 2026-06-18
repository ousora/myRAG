# myRAG Migration Plan: Zero LangChain + Entity Graph + Local bge-m3

> **Goal**: Remove `langchain-text-splitters` dependency, build wikilink entity graph for chunks, and support one-click switching of bge-m3 embedding (remote API ↔ local sentence-transformers).

---

## I. Change Checklist

| # | Item | Files | Complexity |
|---|------|-------|------------|
| A | Rewrite chunker: replace LangChain splitters with pure Python | `src/chunkers/__init__.py` | ⭐⭐ |
| B | bge-m3 embedding client supports remote/local dual mode | `src/embedders/bge_m3.py`, `conf/config.yaml` | ⭐⭐⭐ |
| C | Add entity extraction to formatter, output wikilink-formatted markdown | `src/formatters/writer.py`, prompt templates | ⭐⭐ |
| D | Add entities + relations tables in sqlite-vec (optional) | `src/storage/sqlite_vec.py` | ⭐⭐ |

---

## II. A — Rewrite Chunker (Remove LangChain)

### Current State

Current `chunkers/__init__.py` depends on:

```python
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,   # → header-aware split
    RecursiveCharacterTextSplitter, # → fallback for oversized chunks
)
```

Output format (backward compatible):
```python
{"text": "## Section\n\ncontent...", "section_path": ["H2 Title"], "metadata": {"H1": "...", "H2": "..."}}
```

### Implementation Plan

**MarkdownHeaderTextSplitter → Lightweight Markdown AST Parser:**

Recommended: `markdown-it-py` (~30KB, pure Python) or `mistune`:
```python
from markdown_it import MarkdownIt
md = MarkdownIt()
tokens = md.parse(text)
# tokens contain heading nodes with level and content — no regex edge cases to worry about
```

Alternative — pure regex (must cover edge cases):
```python
pattern = r'^#{1,6}\s+.*?(?=\n#{1,6}|\Z)'  # ATX headers only
# Must handle: code blocks containing #, setext headers, consecutive blank lines
```

> ⚠️ **Recommend markdown-it-py** — Regex-based approaches struggle with edge cases like `#` inside code blocks, consecutive blank lines, etc., making bugs harder to debug.

**RecursiveCharacterTextSplitter → Custom `recursive_split` + Sentence Boundary Detection:**

```python
def recursive_split(text: str, chunk_size: int = 512, overlap: int = 64):
    """Fallback splitter for oversized chunks."""
    # 1. Try splitting on paragraph boundaries first (highest semantic fidelity)
    # 2. Fallback to sentence boundary: use regex or NLTK split
    # 3. Only if no valid split found, fall back to character-level
    
    separators = ["\n\n", "\n", r'(?<=[。！？])', '. ', ' ']
```

### Key Design Decisions

- **Output format must NOT change** — `Chunker.chunk()` returns a list of dicts with THREE fields:
  ```python
  {"text": "...", "section_path": ["H2 Title"], "metadata": {"H1": "Doc Title", "H2": "Section"}}
  ```
  - `metadata` dict has LangChain-style keys (`{"H1": ..., "H2": ...}`) — used by formatter and downstream logic.
  - `section_path` is a flat list derived from metadata (e.g., `[chunk["metadata"]["H2"]]`).
  - Downstream consumers: `embedders/bge_m3.py` reads `"text"` + adds `"embedding"`; `storage/sqlite_vec.py` reads `"text"`, `"section_path"` (stored as JSON string in DB), and optionally `"metadata"`.
  
- **Preserve plain-text fallback logic** — when a document has no markdown headers, fall back to pure text split (current behavior).

- **`chunkers/tests/test_chunker.py` MUST pass** — this is the regression guarantee. Tests verify:
  - Empty input → empty output
  - Simple header splits produce ≥2 chunks
  - `section_path` contains expected headers
  - Hierarchical metadata (`H1`, `H2`, `H3`) preserved correctly
  - Oversized sections split into sub-chunks
  - Pipeline facade compatibility (`pipeline.Chunker(chunk_size=...)`)

### Acceptance Criteria

1. `uv run pytest src/chunkers/tests/ -v` passes all tests
2. Remove `"langchain-text-splitters>=0.3"` from `pyproject.toml`, add `markdown-it-py>=3.0,<4` (pinned version)
3. `uv sync && uv run pytest` is fully green (no other LangChain dependencies)

### ⚠️ Setext Header Support

Markdown also supports `=====` (H1) / `-----` (H2) style setext headers. `markdown-it-py` parses them by default — verify that chunk extraction correctly handles their hierarchy and text. Add test cases covering setext header scenarios.

---

## III. B — bge-m3 Embedding Dual-Mode Switch

### Current State

Current implementation (remote API only):
```python
# embedders/bge_m3.py
class Embedder:
    def __init__(self, *, base_url="", model=""):
        self.client = httpx.Client(base_url=base_url)
    
    def embed(self, text):
        resp = self.client.post("/v1/embeddings", json={"model": ..., "input": [text]})
        return data["data"][0]["embedding"]  # → list[float], 1024-d
```

config.yaml:
```yaml
embedding:
  base_url: "http://localhost:11435"
  model: "bge-m3"
```

### Implementation Plan

**New `src/embedders/local_bge.py`:**

```python
"""Local bge-m3 embedding via sentence-transformers (no server needed)."""

class LocalEmbedder:
    def __init__(self, *, model_name="BAAI/bge-m3", device=None, batch_size=32, max_tokens_per_batch=None):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=device or "cpu")
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch or 512 * 32  # default: ~8K tokens per batch
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate — bge-m3 tokenizer is similar to GPT-like.
        
        Rule of thumb: Chinese char ≈ 0.5 token, English word ≈ 1 token.
        For exact count, use the model's tokenizer (requires transformers dependency).
        """
        # Simple heuristic for quick estimation
        return len(text) // 2
    
    def _adaptive_batch_size(self, texts: list[str]) -> int:
        """Dynamically reduce batch size if total tokens exceed limit.
        
        bge-m3 with 1024-dim float embeddings can OOM on CPU with large batches.
        This function prevents that by splitting when memory pressure is detected.
        """
        estimated_total_tokens = sum(self._estimate_tokens(t) for t in texts)
        if estimated_total_tokens > self.max_tokens_per_batch:
            # Scale down proportionally
            scale = self.max_tokens_per_batch / max(estimated_total_tokens, 1)
            return max(4, int(self.batch_size * scale))
        return self.batch_size
    
    def embed(self, text: str | list[str]) -> list[list[float]]:
        if isinstance(text, str):
            return [self._model.encode(text).tolist()]
        
        # Batch encoding with memory protection
        all_embeddings = []
        effective_bs = self._adaptive_batch_size(text)
        
        for i in range(0, len(text), effective_bs):
            batch = text[i:i + effective_bs]
            
            try:
                embeddings = self._model.encode(batch)  # → (batch_n, 1024) numpy
            except RuntimeError as e:
                # OOM fallback: reduce to single-item batches
                if "out of memory" in str(e).lower():
                    logger.warning("OOM on batch %d-%d, falling back to individual encoding", i, i + effective_bs)
                    for chunk_text in batch:
                        all_embeddings.append(self._model.encode(chunk_text).tolist())
                    continue
                raise
            
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
    
    def store_chunks(self, chunks: list[dict], *, doc_id: str) -> list[dict]:
        """Add embeddings to a list of chunks (matching current Embedder API).
        
        Current embedders/embedders.bge_m3.py signature:
            store_chunks(chunks, doc_id) → [chunk_with_embedding]
        
        Each input chunk must have: {"text", "section_path"} (no embedding key).
        Returns the same chunks augmented with an "embedding" key.
        """
        texts = [ch["text"] for ch in chunks]
        embeddings = self.embed(texts)
        result = []
        for ch, emb in zip(chunks, embeddings):
            out = dict(ch)
            out["embedding"] = emb
            result.append(out)
        return result
    
    def store_document(self, *, title: str, tags: list[str], text_summary: str, 
                       source_file: str, total_chunks: int,
                       embedding: list[float] | None = None) -> dict:
        """Store document-level record (B index), matching current Embedder API.
        
        Note: The embedding param is optional — may be used by downstream for doc-level search.
        """
        return {
            "id": id(self) & 0x7FFFFFFF,  # simple surrogate ID
            "title": title,
            "tags": tags,
            "text_summary": text_summary[:500],
            "source_file": source_file,
            "total_chunks": total_chunks,
            "embedding": embedding,
        }
```

> ⚠️ **API compatibility note** — Current `embedders/bge_m3.py` exposes `.store_chunks(chunks, doc_id)` and `.store_document(...)`. LocalEmbedder must implement these exact signatures. The Embedder's store_chunks returns chunks WITH `"embedding"` keys added; downstream code (pipeline.core → storage.sqlite_vec.upsert_chunks) expects this format.

### Config Example:

```yaml
embedding:
  mode: "local"                     # ← "remote" or "local" (one-click switch)
  
  # When mode == "remote":
  base_url: "http://localhost:11435"
  model: "bge-m3"
  
  # When mode == "local":
  local_model: "BAAI/bge-m3"        # HF model name, auto-downloads on first use
  batch_size: 32                    # preferred batch size (default)
  max_tokens_per_batch: 8192        # soft limit — will auto-reduce if exceeded
  force_gpu: false                  # true = prefer CUDA; default auto-detects torch.cuda.is_available()
```

Default `max_tokens_per_batch=8192` (~500 chars × 32 chunks), safe for bge-m3 on CPU. Increase this value and monitor memory usage if higher throughput is needed; decrease it if OOM occurs.

### ⚠️ Vector Space Consistency (P0 — CRITICAL)

Remote API and locally downloaded models from HuggingFace Hub may differ in version (even with the same name), resulting in incompatible vector spaces. **After switching modes, ALL documents must be re-indexed** — mixing old and new vectors severely degrades retrieval quality.

Add explicit warning in config and code:
```python
class Embedder:
    def __init__(self, *, base_url="", model=""):
        from config import get_config
        cfg = get_config()
        self.mode = cfg.embedding_mode
```

### ⚠️ Config Mutual Exclusion and Environment Variable Override

Config validation using flat attributes:
```python
def validate_embedding_config(cfg):
    mode = cfg.embedding_mode  # ← flat attribute, not cfg.get("embedding", {}).get("mode")
    if mode == "remote":
        assert cfg.embedding_base_url, "base_url required when mode=remote"
    elif mode == "local":
        assert cfg.embedding_local_model, "local_model required when mode=local"
    else:
        raise ValueError(f"Unknown embedding mode: {mode}")
```

> ⚠️ **Config attribute path note** — The Config class uses flat attributes (e.g. `cfg.embedding_base_url`), not nested objects. New fields added by this plan:
> - `cfg.embedding_mode` (str) — `"remote"` | `"local"`
> - `cfg.embedding_local_model` (str, optional) — HF model name when mode=local

Support environment variable overrides for deployment flexibility:
- `$EMBEDDING_MODE` → overrides `cfg.embedding_mode`
- `$HF_HOME` → standard HF cache directory (already supported by sentence-transformers)

### ⚠️ Vector Space Consistency Warning (P0 — CRITICAL)

The vector space consistency warning should be placed in the **pipeline layer** (`core.py`) where `store_path` is available, not in the Embedder class itself. The Embedder doesn't know about DB paths.

In `process_file()` or `_ingest_markdown()`:
```python
if cfg.embedding_mode == "local" and Path(store_path).exists():
    logger.warning(
        "Switched to local embedding mode. Existing vectors were computed with a different model version.\n"
        "All documents must be re-indexed for consistent retrieval."
    )
# User should confirm before proceeding; optionally block until --force-reindex flag is set.
```

Remote API and locally downloaded models from HuggingFace Hub may differ in version (even with the same name), resulting in incompatible vector spaces. **After switching modes, ALL documents must be re-indexed** — mixing old and new vectors severely degrades retrieval quality.

### ⚠️ GPU Auto-Detection and Progress Bar

sentence-transformers auto-detects CUDA, but explicit config override is recommended. Enable progress bar on first load:

```python
from sentence_transformers import SentenceTransformer, logging as st_logging
st_logging.set_verbosity_error()  # suppress HF download logs
model = SentenceTransformer(model_name, device=device or "cpu", show_progress_bar=True)
```

### ⚠️ bge-m3 Single Text Length Limit

bge-m3 max sequence length is 8192 tokens. If a single chunk exceeds this limit, truncate it:
```python
MAX_CHUNK_TOKENS = 7680  # leave headroom for special tokens
if estimated_tokens > MAX_CHUNK_TOKENS:
    logger.warning("Chunk exceeds max length (%d tokens), truncating to %d", 
                   estimated_tokens, MAX_CHUNK_TOKENS)
```

### Config Design

```yaml
# conf/config.yaml — add embedding.mode
embedding:
  mode: "remote"            # ← "remote" or "local" (one-click switch)
  
  # When mode == "remote":
  base_url: "http://localhost:11435"
  model: "bge-m3"
  
  # When mode == "local":
  local_model: "BAAI/bge-m3"   # HF model name, auto-downloads on first use
```

### Dependency Changes

`pyproject.toml`:
- **Remove** `langchain-text-splitters` → ❌ (due to change A)
- **Add** `sentence-transformers>=2.7,<3` (optional via extras when mode=local)

```toml
[project.optional-dependencies]
dev = ["pytest", ...]
sqlite-vec = ["sqlite-vec"]
local-embeddings = ["sentence-transformers>=2.7,<3"]  # ← pinned major version, avoid breaking changes
```

Users install on demand: `uv sync --extra local-embeddings`.

### ⚠️ Implicit Dependency Check (P0)

After removing `langchain-text-splitters`, globally verify no other LangChain imports remain:
```bash
rg 'import langchain|from langchain' src/ pyproject.toml
# Expected: ZERO results. If any found, they must be removed or refactored.
```

### Acceptance Criteria

1. `mode=remote`: behavior identical to current API call path (no changes)
2. `mode=local`: same input returns same 1024-dim float list
3. First use auto-downloads model (~1.5GB, with progress indicator)
4. CPU inference performance acceptable — bge-m3 embeds one text in ~1–3s on CPU (depends on hardware)

### ⚠️ Model Version Locking (P2)

Remote API and locally downloaded HF Hub models may differ in version even with the same name, causing vector space inconsistency. Recommended:

```yaml
embedding:
  local_model_revision: "refs/pr/7"   # ← optional: specify exact revision/hash
```

Or use `sentence-transformers` fingerprint verification:
```python
model = SentenceTransformer(model_name)
fingerprint = model.model_hash()  # verify against known-good fingerprint on first load
```

---

## IV. C — Entity Extraction + Wikilink Graph (Approach B: Explicit Links)

### Design Objective

Inspired by LLM Wiki's interlinked markdown KB concept — the formatter outputs `.md` with automatic `[[entity_name]]` insertion, and relations are queried via fts5 reverse lookup.

**Core idea**: The formatter prompt returns an `entities` field containing extracted entities and their types. writer.py inserts wikilinks at corresponding positions in the body text when writing .md files.

### Prompt Changes

Add to existing LLM formatter prompt output schema:

```json
{
    "title": "...",
    "tags": ["tag1", "tag2"],
    "metadata": {
        "sections": [...],
        "entities": [                    // ← NEW
            {"name": "GPT-4", "type": "PRODUCT"},
            {"name": "中国", "type": "LOCATION"}
        ]
    },
    "body": "# Title\n\n正文内容..."
}
```

Prompt constraint example:
> Extract key entities (people, organizations, products, locations, concepts) from the document. For each entity, provide a `name` and a `type`. The name must match exactly how it appears in the text.

> ⚠️ **Prompt Details (P1)** — must add these critical constraints for output stability:
> - Use `temperature=0` (deterministic output)
> - Restrict entity type set to `PERSON | ORG | PRODUCT | LOCATION | CONCEPT`, no free-form new types
> - Entity name **must** exactly match original text (no normalization or synonym expansion)
> - Include 2–3 few-shot examples for consistency

### Writer.py Changes

In `_write_body_with_sections()`, replace entity mentions with wikilinks:

```python
def _insert_wikilinks(body: str, entities: list[dict]) -> str:
    """Replace entity mentions with [[wikiname]] format.
    
    CRITICAL: Skip code blocks (```, ``) and existing links []() to avoid corruption.
    Use longest-match-first strategy to prevent short entities from overwriting long ones.
    
    ⚠️ OFFSET BUG FIX: After each replace, string offsets shift. 
    Strategy: collect all replacements first, then apply from back-to-front.
    """
    # 1. Extract all "protected" ranges (code blocks + inline code + existing links)
    protected_ranges = _extract_protected_ranges(body)
    
    # 2. Collect ALL replacements (entity_name → [[wikiname]]) as list of (start, end, replacement)
    replacements = []
    for e in sorted(entities, key=lambda x: -len(x["name"])):  # longest first
        name = re.escape(e["name"])
        # Find all matches with their positions
        for match in re.finditer(name, body):
            pos_start, pos_end = match.start(), match.end()
            if not _is_inside_protected(pos_start, protected_ranges):
                replacement = f'[[{e["name"]}]]'
                # Check this position hasn't been modified by a longer entity already matched
                if not any(ps <= pos_start < pe for ps, pe in replacements):
                    replacements.append((pos_start, pos_end, replacement))
    
    # 3. Apply replacements FROM BACK TO FRONT to preserve earlier positions
    for start, end, replacement in sorted(replacements, key=lambda x: -x[0]):
        body = body[:start] + replacement + body[end:]
    
    return body


def _extract_protected_ranges(text: str) -> list[tuple[int, int]]:
    """Find all protected regions where wikilink insertion is unsafe."""
    protected = []
    
    # Code blocks: ```...``` (supports language tag, indented code)
    for m in re.finditer(r'`{1,6}[^`]*`{1,6}', text):
        protected.append((m.start(), m.end()))
    
    # Inline code: `...` (single backtick pair, no nested backticks)
    for m in re.finditer(r'(?<!`)`(?!`)([^`]*)`(?!`)', text):
        protected.append((m.start(), m.end()))
    
    # Existing wikilinks: [[...]]
    for m in re.finditer(r'\[\[.*?\]\]', text):
        protected.append((m.start(), m.end()))
    
    # Existing links: [text](url)
    for m in re.finditer(r'\[[^\]]*\]\([^)]*\)', text):
        protected.append((m.start(), m.end()))
    
    return sorted(protected)


def _is_inside_protected(position: int, protected_ranges: list[tuple[int, int]]) -> bool:
    """Check if position falls within any protected range."""
    for start, end in protected_ranges:
        if start <= position < end:
            return True
    return False
```

Key constraints:
- **Skip code blocks** — entity names inside Markdown code blocks (```\n...\n```) and inline code (`...`) must NOT be replaced.
- **Skip existing links** — content in `[text](url)` format should not get wrapped with another wikilink layer.
- **Longest match first** — replace "AI Agent" before "AI", preventing short entities from overwriting long ones.
- **Back-to-front replacement** — collect all replacement positions then apply in descending offset order to avoid string offset shift bugs.

### Unit Test Requirements (P1)

Must verify the following scenarios:
| Scenario | Expected Behavior |
|----------|-------------------|
| `"GPT-4 is OpenAI's product"` + entities=[GPT-4, OpenAI] | `[["GPT-4"]] is [["OpenAI"]]'s product` |
| ``Code: `# Header` is not a title`` | Original preserved, no wikilink inserted |
| `"Entity [[already linked]] content"` | Keep existing [[ ]] format unchanged |
| entities=["中国科技政策", "中国"] | Long match first, result correctly nested or only long entity replaced |

### Entity Extraction Evaluation Set (P1)

Entity extraction entirely depends on LLM prompt quality — no evaluation mechanism currently. Recommended to build a small-scale Golden Dataset:

```python
# tests/test_entity_extraction.py — pseudocode illustration
golden = [
    ("GPT-4 is OpenAI's product released in 2023...", {"GPT-4": "PRODUCT", "OpenAI": "ORG"}),
    ("China tech policy supports AI development.", {"中国": "LOCATION", "科技政策": "CONCEPT"}),
    # ... 50–100 examples covering edge cases
]

def evaluate_extraction():
    for text, expected in golden:
        result = extract_entities(text)  # your LLM call
        # Strict mode: match on BOTH name AND type (recommended)
        # Use set of tuples for deterministic matching: {(name, type), ...}
        result_set = {tuple(sorted(e)) if isinstance(e, (list, tuple)) else e for e in result}
        expected_set = {tuple(sorted(e)) if isinstance(e, (list, tuple)) else e for e in expected}
        precision = len(result_set & expected_set) / max(len(result_set), 1)
        recall = len(result_set & expected_set) / max(len(expected_set), 1)
        assert precision > 0.85 and recall > 0.85
```

> ⚠️ **Evaluation strict mode** — recommend requiring entity NAME + TYPE to both match. Use set of tuples `set[(name, type)]` for deterministic comparison (avoid order-dependent matching). Note `strict=True` in the Golden Dataset spec.

Recommended thresholds: Precision ≥ 0.9, Recall ≥ 0.85. Prompts below these thresholds should not be merged.

### Golden Dataset Versioning (P2)

When a local file-based dataset is modified across iterations, evaluation results become irreproducible. Recommended version management approach:

**Approach A — Git LFS:**
```bash
# Store dataset in git, large files via LFS
git lfs track "*.jsonl"
git add tests/data/golden_entities_v1.jsonl
```

**Approach B — Separate Dataset Repository (recommended):**
```
myrag-entities/              # Separate repo for evaluation data
├── golden/
│   ├── v1/                  # versioned snapshots
│   │   └── entities.jsonl
│   └── CHANGELOG.md         # What changed, why
└── README.md                # Schema + usage instructions
```

**Evaluation script records dataset hash:**
```python
import hashlib
def load_golden(path="tests/data/golden_entities_v1.jsonl"):
    content = Path(path).read_bytes()
    _dataset_hash = hashlib.sha256(content).hexdigest()[:12]
    
    with open(path, "r") as f:
        return json.load(f)
```

CI outputs dataset hash: `EVAL_DATASET_HASH=a3f7c9d1e2b4 — golden v1 (2025-06)`. This allows tracing back to exact historical evaluation results even after dataset updates.

**Change management**: Modifications to the Golden Dataset (add/delete/modify entries) must go through PR review + CHANGELOG.md — direct overwrites of existing version entries are prohibited.

---

## V. D — sqlite-vec Relationship Tables (Optional Enhancement)

### Why Last?

Approach C (wikilink in markdown + fts5) covers 80% of relationship tree needs at minimal cost. The relation table is an advanced feature:

```sql
-- Entity index table (proposed addition)
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    type TEXT,           -- PERSON / ORG / PRODUCT / LOCATION / CONCEPT
    source_doc_ids TEXT  -- JSON array of doc IDs mentioning this entity
);

-- Relation table (entity-to-entity connections, proposed addition)
CREATE TABLE relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_entity_id INTEGER REFERENCES entities(id),
    to_entity_id INTEGER REFERENCES entities(id),
    relation_type TEXT NOT NULL,   -- "co_occur", "defined_by", "related_to"...
    confidence REAL DEFAULT 0.8
);

-- FTS5 index (for entity search)
CREATE VIRTUAL TABLE entities_fts USING fts5(name, type, content=entities);

-- ⚠️ Relation table indexes (critical for performance)
CREATE INDEX idx_relations_from ON relations(from_entity_id);
CREATE INDEX idx_relations_to ON relations(to_entity_id);
CREATE INDEX idx_relations_type ON relations(relation_type);

-- Many-to-many join table (replaces source_doc_ids JSON string queries)
CREATE TABLE doc_entities (
    doc_id TEXT NOT NULL,           -- matches chunks.source_doc_id (TEXT type!)
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    PRIMARY KEY (doc_id, entity_id)
);
```

> ⚠️ **Actual sqlite-vec schema** — The existing `chunks` table stores `section_path` as a JSON string (`json.dumps([H2_Title])`). Downstream code parses it with `json.loads(raw)` in `_parse_section_path()`. When adding new tables, ensure foreign keys use compatible types:
> - `source_doc_id` is TEXT (not INTEGER) — match this for any cross-table joins
> - `section_path` column stores JSON array strings like `["Section A"]`, not Python lists

### Use Cases

- **Entity linking**: User asks "Who developed GPT-4?" → query `relations WHERE from_entity='GPT-4' AND relation_type='developed_by'`
- **Multi-hop traversal**: 中国 → 科技政策 → AI → GPT-4 (chained retrieval)

### Current Phase Recommendation

Do A+C first, validate results, then decide on D. Entity extraction quality determines the value of the relationship tree — if the entity extraction prompt is poor, building tables is pointless.

### ⚠️ Do NOT Add B Index (Document-Level Coarse-Grained)

The current pipeline has a B index: `store_document()` creates an embedding of the document-level summary for coarse-grained search. **We should not add this** during migration because:

1. **Marginal benefit**: The B index just wraps up chunk-level results. Searching "which documents are relevant" vs "which chunks are relevant" gives nearly the same answer after deduplication by `source_doc_id`.
2. **Added complexity**: Three search paths (chunk vector + doc vector + entity graph) means three queries to merge and rank. More query logic = more bugs.
3. **Storage cost**: Extra embedding (~1KB per document × N documents). For 10K docs that's ~10MB — not huge but unnecessary.
4. **No clear win condition**: Without a concrete use case (e.g., "I need to browse by topic categories"), it's just optimization without validation.

**When would you add B index later?** When you have >50K chunks and want fast "list all relevant docs" queries without chunk-level deduplication overhead. For now, stick with chunk-level search + optional entity graph.

---

## VI. Implementation Order and Dependencies

```
Phase 1 (independent) ──→ Phase 2 (depends on C) ──→ Phase 3 (optional)
    A                        B + C                         D
   chunker                  embedder+formatter             relations table
   ~30 min                 ~1 hour                      optional
```

| Phase | Change | Risk | Rollback Plan |
|-------|--------|------|---------------|
| **Phase 1** | A: Rewrite chunker | Low — pure Python, test-covered | `git revert` one file |
| **Phase 2a** | B: Local embedder | Medium — model download, perf diff | Config toggle back to remote |
| **Phase 2b** | C: Entity extraction | Medium — prompt quality determines outcome | Prompt tweak, no schema change |
| **Phase 3** | D: Relations table | Low — pure DDL, backward-compatible | Drop tables |

### ⚠️ E2E Regression Required After Phase 1 (P0)

After A is complete, **do not proceed directly to Phase 2**. First run full pipeline regression test:
```bash
# End-to-end: ingest a real document through the full pipeline
cd myrag && uv run python -m tests.e2e_regression --all-docs

# Verify downstream consumers still work (embedder + storage)
uv run pytest src/embedders/tests/ src/storage/tests/ -v
```

If E2E fails, the chunker output format has breaking changes — rollback to Phase A. This is the critical gate preventing cascade failure.

---

## VII. Complete Change File List

```
src/chunkers/__init__.py          ← Rewrite (remove LangChain import)
src/embedders/bge_m3.py           ← Add LocalEmbedder + config switch
conf/config.yaml                  ← Add embedding.mode field
conf/config.example.yaml          ← Sync update example config
src/formatters/writer.py          ← _insert_wikilinks() function
pyproject.toml                    ← Remove langchain-text-splitters, add local-embeddings extra
src/storage/sqlite_vec.py         ← Optional: entities/relations table init (Phase 3)
```

---

## VIII. Pitfalls and Warnings

### 8.1 Chunker API Compatibility ⚠️ CRITICAL

The return format of `Chunker.chunk()` is the pipeline contract. Change A **must** preserve this signature unchanged:
```python
def chunk(self, text: str) -> list[dict]:
    # Each dict must have: {"text", "section_path", "metadata"}
```

Any field name change cascades to the embedder and storage modules.

### 8.2 bge-m3 Local Performance ⚠️

CPU-only inference — embedding one chunk (~500 chars):
- **bge-small** (~70MB): ~0.3s/text — recommended for daily use
- **bge-base** (~400MB): ~1s/text
- **bge-large (m3)** (~1.5GB): ~2–5s/text

If you have many documents (>1000 chunks), local embedding may become a bottleneck. Recommendation:
- Small projects: bge-small + local mode
- Large projects: continue with remote API or add batch processing

### 8.3 Wikilink Markdown Rendering Impact ⚠️

Most markdown renderers (GitHub, Obsidian, VS Code) display `[[wikiname]]` as plain text — no impact on reading. But some tools attempt to parse them as links:
- **Obsidian** → auto-creates wiki links (positive effect, exactly what LLM Wiki designs for)
- **GitHub Markdown** → displays literally (harmless)
- **Custom renderers** → may need CSS/JS support

### 8.4 sentence-transformers First Download ⚠️

`SentenceTransformer("BAAI/bge-m3")` auto-downloads from HuggingFace Hub (~1.5GB) on first use. Recommended:
- Document this behavior in README
- Provide `--force-download` parameter for manual triggering
- Consider pre-installing into Docker image or CI cache (see below)

### 8.5 Docker / CI Cache Strategy (P2)

bge-m3 model is ~1.5GB — not pre-baked in Dockerfile/CI means repeated downloads on every build/deploy.

**Dockerfile pre-heat:**
```dockerfile
# Pre-download model during build (avoids runtime download)
RUN pip install sentence-transformers && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

**CI HuggingFace Hub cache volume:**
```yaml
# GitHub Actions example
- uses: actions/cache@v4
  with:
    path: ~/.cache/huggingface
    key: hf-cache-${{ runner.os }}-${{ hashFiles('pyproject.toml') }}
    restore-keys: |
      hf-cache-${{ runner.os }}-
```

**Better approach — export model as layer in CI:**
```bash
# Build once, push model to container registry or shared NFS
docker run --rm -v $(pwd)/models:/app/models huggingface-cli download BAAI/bge-m3 --local-dir models/
```

### 8.6 Rollback Plans

| Phase | Code rollback | Data rollback |
|-------|---------------|---------------|
| **A (chunker)** | `git revert` one file | No impact — pure refactor, doesn't change storage format |
| **B (embedder)** | Config toggle → back to `mode: remote` | Existing local vectors can still be used; new data written with remote mode |
| **2b (wikilink)** | Prompt rollback + writer revert | Wikilinks are in .md files — cleanable via `sed -i 's/\[\[//g' output/*.md`, but better to regenerate from version control |
| **D (relations table)** | DROP TABLE entities, relations; remove migration script | ALTER TABLE vec_chunks DROP COLUMN entity_ids |

---

## IX. Summary

| Change | Benefit | Cost |
|--------|---------|------|
| A: Zero LangChain | Eliminates dependency hell, cleaner debugging | ~30 min, test-covered |
| B: Local bge-m3 | No network dependency, offline usable | Config toggle + model download (~1.5GB) |
| C: Wikilink graph | Entity search, cross-doc relations | Prompt quality is the bottleneck |

**Recommendation start with Phase 1 (A)** — smallest change, lowest risk, immediate payoff visible. Phases B+C of Phase 2 can proceed in parallel; D on demand.

---

## X. Documentation and Release Requirements (P1)

Every phase completion requires updating user documentation:

| Phase | README Addition | CHANGELOG.md |
|-------|-----------------|---------------|
| **B** | `local-embeddings` install instructions, first-time model download notice (~1.5GB), CPU performance benchmarks | Add `embedding.mode` config field docs |
| **C** | Wikilink format description (impact on Obsidian/GitHub renderers) | Add entity extraction prompt change notes |
| **D** | Relations table query examples | N/A (optional feature, not in CHANGELOG) |

Before release, users must be notified: **After enabling wikilinks, output markdown files remain compatible with standard markdown renderers. However, Obsidian and similar tools will parse them as wiki links.**

---

## XI. Sparse Mode Considerations (P2)

If BM25/sparse search is introduced later:

### ⚠️ Tokenizer Version Locking

When using `jieba` or other Chinese tokenizers, different versions produce different dictionaries and segmentation results, leading to inconsistent sparse vector spaces even with identical min_df/max_features settings. **Tokenize version must be pinned**:

```yaml
# conf/config.yaml — sparse search (future)
sparse:
  tokenizer: "jieba"
  tokenizer_version: "0.42.1"  # ← pinned version, ensure reproducible tokenization
  min_df: 3
  max_features: 50000
```

- Include `tokenizer_version`, `min_df`, AND `max_features` in fingerprint calculation input (same mechanism as chunker hash and model fingerprint)
- If ANY of tokenizer/tokenizer_version/min_df/max_features changes, rebuild all sparse indexes and record CHANGELOG
- Consider using `jieba-fast` (optimized fork) instead of native jieba ~20% faster tokenization

---

*Plan created on request. Review before execution.*
