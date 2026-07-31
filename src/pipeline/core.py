"""RAG data cleanup pipeline — parse → clean → format → chunk → embed (Hybrid A+B).

Pipeline flow:
    Raw file (.pdf/.docx/.txt)
        ↓ parser.parse_file()     # Unified backend: MarkItDown + Trafilatura
        ↓ cleaner.clean_text()    # Deterministic regex-based noise removal
        ↓ formatter.format_text()  # LLM semantic structuring → title/tags/chunks
            → write_to_md(result, output_dir/)   [human-readable .md]
        ↓ chunker.chunk(section_path=...)     # Physical splitting with section headers
        ↓ embedder.store_chunks() / store_document()  # Hybrid A+B indexing

Hybrid Retrieval (A + B):
    A: Chunk-level index — fine-grained search, direct answer generation
       [chunk] → bge-m3 embedding → sqlite-vec vector DB
    
    B: Document-level index — coarse-grained context fallback
       [doc_summary] → bge-m3 embedding → sqlite-vec vector DB

Usage (traditional RAG):
    from pipeline.core import process_file, process_directory
    
    chunks = process_file("path/to/report.pdf")  # traditional chunking only

Usage (LLM-formatted + Hybrid A+B):
    from pipeline.core import process_file_hybrid
    
    result = process_file_hybrid(
        filepath="path/to/document.pdf",
        doc_id="doc_001"
    )
    # Returns: {
    #   "chunks": list[dict]  (A - fine-grained, ready for sqlite-vec)
    #   "document": dict      (B - coarse-grained, ready for sqlite-vec)
    # }

Usage (LLM-formatted + Markdown output):
    from pipeline.core import process_file_with_md
    
    md_path = process_file_with_md(
        filepath="path/to/document.pdf",
        output_dir="./output/",
    )
"""


from __future__ import annotations

import concurrent.futures
import httpx
import logging
from pathlib import Path

from config import get_config_lazy as _get_config
from embedders import Embedder  # noqa: F401 — used in rag_query type hints
from storage.sqlite_vec import SQLiteVecStore  # noqa: F401 — used in rag_query type hints

# Trigger parser registration at module load time
import parsers  # noqa: F401 — loads dispatcher (MarkItDown + Trafilatura)

from . import markdown_utils, utils

logger = logging.getLogger(__name__)


def _build_doc_summary(title: str, tags: list[str], body: str, *, head: int = 800, tail: int = 400) -> str:
    """Build a document-level (B index) summary from head + tail of the body.
    
    Delegates to utils.build_doc_summary for the actual implementation.
    """
    return utils.build_doc_summary(title, tags, body, head=head, tail=tail)


class TextCleaner:
    """Facade — delegates to parsers.text_cleaner.TextCleaner.
    
    Kept as a class in pipeline.py for backward compatibility with existing callers.
    The actual implementation lives in parsers/text_cleaner.py (YAML config support).
    """

    def __init__(self, *, remove_page_breaks=True, collapse_whitespace=True, rules_config="clean_rules.yaml"):
        from parsers.text_cleaner import TextCleaner as _RealCleaner  # noqa: F811
        self._cleaner = _RealCleaner(
            remove_page_breaks=remove_page_breaks,
            collapse_whitespace=collapse_whitespace,
            rules_config=rules_config,
        )

    def clean(self, text: str) -> str:
        return self._cleaner.clean(text)


class Chunker:
    """Facade — delegates to chunkers module for all chunking logic.

    Kept in pipeline.py for backward compatibility with existing callers.
    The canonical implementation lives in chunkers/__init__.py.
    """
    from chunkers import Chunker as _RealChunker

    def __new__(cls, **kwargs):
        return cls._RealChunker(**kwargs)


def process_file(filepath: str, *, remove_page_breaks=True, collapse_whitespace=True, rules_config="conf/clean_rules.yaml", chunk_size=1024) -> list[dict]:
    """Parse a single file and return structured chunks (traditional RAG).

    Pipeline: parser → cleaner → chunker → output dict list.
    
    For LLM-formatted output with hybrid A+B indexing, use process_file_hybrid().
    
    Returns list of dicts: [{"text": ..., "metadata": {...}}, ...]
    """
    parser = utils.resolve_parser(filepath)
    if parser is None:
        logger.warning("Skipped %s — no parser found", filepath)
        return []

    raw_text = parser.parse(filepath)
    cleaned = TextCleaner(remove_page_breaks=remove_page_breaks, collapse_whitespace=collapse_whitespace, rules_config=rules_config).clean(raw_text)
    chunks = Chunker(chunk_size=chunk_size).chunk(cleaned)

    result = [
        {"text": chunk["text"], "metadata": {"source": filepath}}
        for chunk in chunks
    ]
    logger.info("  → %d chunks from %s", len(result), Path(filepath).name)
    return result


def process_file_hybrid(filepath: str, *, doc_id="doc_0", remove_page_breaks=True, 
                        collapse_whitespace=True, rules_config="conf/clean_rules.yaml", chunk_size=1024, store_path=None, md_output_dir=None, md_path=None):
    """Parse file with LLM formatter → chunker → embedder → sqlite-vec (Hybrid A+B).

    Args:
        filepath: Path to the document file.
        doc_id: Unique identifier for this document in the index.
        chunk_size: Max characters per chunk.
        store_path: Optional path to sqlite-vec database. If provided, chunk
                    vectors are persisted for later retrieval.
        md_output_dir: Optional directory to write structured markdown (via write_to_md).
        md_path: Optional path to an EXISTING .md file. When given and the file
                 exists, the LLM formatter is skipped entirely and the .md is
                 reused (two-phase: generate once, ingest many times).

    Returns dict with:
        chunks  — list of dicts with embedding data (A - fine-grained)
        document — single dict with summary + embedding (B - coarse-grained)
        db_path  — path to sqlite-vec DB if store_path was provided, else None
        md_path  — path to generated/reused .md file, else None
    """
    from formatters import format_text_async, write_to_md
    import re as _re

    cfg = _get_config()

    md_out_path = None
    cleaned = ""
    if md_path and Path(md_path).is_file():
        # Reuse an existing .md — skip the (expensive) LLM formatter entirely.
        logger.info("Reusing existing markdown: %s", md_path)
        content = Path(md_path).read_text(encoding="utf-8")
        title_match = _re.search(r'^#\s+(.+)$', content, _re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Untitled"
        result = {"title": title, "body": content, "tags": [], "metadata": {"entities": []}}
        md_out_path = md_path
    else:
        # 1. Parse & Clean
        parser = utils.resolve_parser(filepath)
        if parser is None:
            logger.warning("Skipped %s — no parser found", filepath)
            return {
                "chunks": [],
                "document": {},
                "format_result": {"title": "", "tags": [], "body": ""},
                "md_path": None,
                "db_path": None,
            }

        raw_text = parser.parse(filepath)
        cleaned = TextCleaner(remove_page_breaks=remove_page_breaks, collapse_whitespace=collapse_whitespace, rules_config=rules_config).clean(raw_text)

        # 2. LLM Format (async)
        future = format_text_async(cleaned, source_type=utils.source_type_for(filepath))
        try:
            result = future.result(timeout=cfg.format_timeout)
        except concurrent.futures.TimeoutError:
            logger.warning("LLM formatting timed out after %ds for %s", cfg.format_timeout, filepath)
            return {
                "chunks": [],
                "document": {},
                "format_result": {"title": "", "tags": [], "body": ""},
                "md_path": None,
                "db_path": None,
            }

        # Write structured markdown if output_dir provided (same path as process_file_with_md)
        if md_output_dir:
            md_out_path = write_to_md(result, md_output_dir)
            logger.info("  → Markdown written to %s", md_out_path)

    # 3. Render markdown with headers from metadata.sections, then chunk
    formatted_md = markdown_utils.render_markdown_with_sections(result)
    # Drop reference/bibliography sections so they don't pollute retrieval.
    formatted_md = markdown_utils.strip_reference_sections(formatted_md)
    chunker = Chunker(chunk_size=chunk_size)
    all_chunks = chunker.chunk(formatted_md)

    # Match document-level entities to individual chunks (Phase C)
    entities = result.get("metadata", {}).get("entities", [])
    all_chunks = markdown_utils.match_entities_to_chunks(all_chunks, entities)

    # Build summary text before the try block so it's always available in the except handler.
    title = result.get("title", "Untitled")
    tags = result.get("tags", [])
    # Embed the LLM-formatted body (not the raw cleaned text) so the coarse-grained
    # B index captures the document's structured semantics. Use a head+tail slice
    # so long documents still contribute their closing context to the B summary.
    body_for_summary = result.get("body", "") or cleaned
    summary_text = _build_doc_summary(title, tags, body_for_summary)

    # 4. Embed + optionally persist to sqlite-vec
    db_path = None

    if store_path:
        # Only construct the embedder and pay for the (remote, expensive)
        # embedding calls when we actually persist. Otherwise we return
        # lightweight metadata and skip all embedding work.
        try:
            from embedders import Embedder
            from storage.sqlite_vec import SQLiteVecStore

            e = Embedder()
            stored_chunks = e.store_chunks(all_chunks, doc_id=doc_id)

            # Document-level index (B) — only embed when we persist it.
            stored_doc = e.store_document(
                title=title,
                tags=tags,
                text_summary=summary_text,
                source_file=filepath,
                total_chunks=len(stored_chunks),
            )
            doc_embedding = stored_doc.get("embedding")

            db = SQLiteVecStore(store_path)

            # Store chunks with embeddings
            db.upsert_chunks(stored_chunks, doc_id=doc_id)

            # Store document-level record
            db.upsert_document(
                title=title,
                tags=tags,
                text_summary=summary_text,
                source_file=filepath,
                total_chunks=len(stored_chunks),
                embedding=doc_embedding,
            )
            db.close()

            db_path = store_path
            logger.info("  → Persisted %d chunks + 1 doc to %s", len(stored_chunks), store_path)
        except (httpx.HTTPError, RuntimeError) as exc:
            logger.warning("Embedding/storage failed (%s): %s", type(exc).__name__, exc)
            stored_chunks = [{"text": c["text"], "section_path": c.get("section_path", ["General"]),
                              "source_doc_id": doc_id, "chunk_index": i} for i, c in enumerate(all_chunks)]
            stored_doc = {
                "title": title,
                "tags": tags,
                "text_summary": summary_text[:500],
                "source_file": filepath,
                "total_chunks": len(stored_chunks),
            }
    else:
        # No persistence requested — skip all embedding (the most expensive
        # remote step) and return lightweight metadata only.
        stored_chunks = [{"text": c["text"], "section_path": c.get("section_path", ["General"]),
                          "source_doc_id": doc_id, "chunk_index": i} for i, c in enumerate(all_chunks)]
        stored_doc = {
            "title": title,
            "tags": tags,
            "text_summary": summary_text,
            "source_file": filepath,
            "total_chunks": len(stored_chunks),
        }

    return {
        "chunks": stored_chunks,       # A - fine-grained index
        "document": stored_doc,         # B - coarse-grained index
        "format_result": result,        # Original LLM output (for metadata)
        "md_path": md_out_path,        # Path to structured/reused .md file
        "db_path": db_path,             # sqlite-vec DB path (None if not persisted)
    }


def process_file_with_md(filepath: str, *, output_dir="./output/", **kwargs):
    """Parse file → LLM formatter → write structured markdown to output/.

    Returns the path of the generated .md file.
    
    This is the user-facing pipeline for generating human-readable documents.
    For vector DB indexing (Hybrid A+B), use process_file_hybrid() instead.
    """
    from formatters import format_text_async, write_to_md

    cfg = _get_config()

    # Parse & Clean
    parser = utils.resolve_parser(filepath)
    if parser is None:
        logger.warning("Skipped %s — no parser found", filepath)
        return None

    raw_text = parser.parse(filepath)
    cleaned = TextCleaner(**kwargs).clean(raw_text)

    # LLM Format
    future = format_text_async(cleaned, source_type=utils.source_type_for(filepath))
    try:
        result = future.result(timeout=cfg.format_timeout)
    except concurrent.futures.TimeoutError:
        logger.warning("LLM formatting timed out after %ds for %s", cfg.format_timeout, filepath)
        return None
    
    # Write markdown to output_dir
    md_path = write_to_md(result, output_dir)
    
    return md_path


def process_directory(dirpath: str, *, extensions=None, chunk_size=1024, **kwargs) -> list[dict]:
    """Walk a directory and process all supported files (traditional RAG)."""
    path = Path(dirpath)

    if extensions is None:
        from parsers.dispatcher import PARSERS
        extensions = set(PARSERS.keys())

    results: list[dict] = []
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        ext = file.suffix.lstrip(".")
        if ext.lower() in {e.lower() for e in extensions}:
            chunks = process_file(str(file), **kwargs)
            results.extend(chunks)

    logger.info("Processed %d files from %s → %d total chunks", len(results), dirpath, len(results))
    return results


def process_directory_hybrid(dirpath: str, *, store_path=None, md_output_dir=None,
                             extensions=None, chunk_size=1024, max_workers=4, **kwargs) -> dict:
    """Process every supported file in *dirpath* with the LLM-formatted Hybrid A+B pipeline.

    Files are parsed/formatted/chunked/embedded concurrently (the expensive
    remote LLM + embedding steps dominate), then each result is persisted to a
    single sqlite-vec store. ``doc_id`` is derived deterministically from the
    file's path relative to *dirpath* so re-running overwrites the same records
    instead of duplicating them.

    Args:
        dirpath: Directory to walk (recursively).
        store_path: Optional sqlite-vec DB; when given, every file's chunks +
                    document record are persisted.
        md_output_dir: Optional dir to write each file's structured .md.
        extensions: Optional set of extensions to include (defaults to all registered parsers).
        chunk_size: Max characters per chunk.
        max_workers: Concurrency for the parse/format/embed phase.

    Returns:
        dict mapping each file path → the per-file result dict from process_file_hybrid.
    """
    from parsers.dispatcher import PARSERS

    path = Path(dirpath)
    if extensions is None:
        extensions = set(PARSERS.keys())
    ext_set = {e.lower() for e in extensions}

    files = [
        str(f) for f in sorted(path.rglob("*"))
        if f.is_file() and f.suffix.lstrip(".").lower() in ext_set
    ]
    if not files:
        logger.info("No supported files found in %s", dirpath)
        return {}

    def _doc_id_for(fp: str) -> str:
        rel = Path(fp).relative_to(path)
        return str(rel).replace("/", "_").replace("\\", "_")

    def _one(fp: str):
        try:
            res = process_file_hybrid(
                fp,
                doc_id=_doc_id_for(fp),
                chunk_size=chunk_size,
                store_path=store_path,
                md_output_dir=md_output_dir,
                **kwargs,
            )
            return fp, res
        except Exception as exc:  # noqa: BLE001 — one bad file shouldn't abort the batch
            logger.warning("Failed to process %s: %s", fp, exc)
            return fp, {"chunks": [], "document": {}}

    summary: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fp, res in ex.map(_one, files):
            summary[fp] = res
            logger.info("Processed %s → %d chunks", Path(fp).name, len(res.get("chunks", [])))

    logger.info("Processed %d files from %s", len(files), dirpath)
    return summary


def rag_query(question: str, db_path: str, *, k: int = 5,
              db: "SQLiteVecStore | None" = None,
              embedder: "Embedder | None" = None) -> dict:
    """Retrieve relevant chunks from sqlite-vec and generate an LLM answer.

    Args:
        question: The user's natural-language query.
        db_path: Path to the sqlite-vec database (created by process_file_hybrid).
        k: Number of top chunks to retrieve for context assembly.
        db: Optional pre-opened :class:`SQLiteVecStore` (reused across calls in a
            session to avoid re-opening the connection per query).
        embedder: Optional pre-constructed :class:`Embedder` (likewise reused).

    Returns dict with:
        "answer": str — LLM-generated answer text.
        "context": list[dict] — retrieved chunks used as context.
        "question": str — the original question (echoed back).
    """
    from storage.sqlite_vec import SQLiteVecStore
    from embedders import Embedder

    # 1. Embed the query with the retrieval instruction prefix (bge-m3 needs it).
    #    embed_query returns a single vector for str input; normalize for the
    #    local backend which returns list[list[float]].
    owned_e = embedder is None
    owned_db = db is None
    e = embedder if embedder is not None else Embedder()
    try:
        query_result = e.embed_query(question)
        query_vector = (
            query_result[0]
            if query_result and isinstance(query_result[0], list)
            else query_result
        )

        # 2. Retrieve relevant chunks (hybrid: vector + FTS5 RRF fusion)
        store = db if db is not None else SQLiteVecStore(db_path)
        try:
            # A: fine-grained chunk retrieval (vector + full-text).
            results = store.hybrid_search(question, query_vector=query_vector, k=k)

            # B: coarse-grained document-level fallback for broad context.
            doc_results = store.search_documents(query_vector=query_vector, k=1)

            # Pre-fetch stored chunk embeddings for MMR — reuse what's already
            # in the index instead of re-embedding every retrieved chunk.
            chunk_vectors = (
                store.get_embeddings_by_ids([c["id"] for c in results]) if results else []
            )
        finally:
            if owned_db:
                store.close()

        if not results and not doc_results:
            return {
                "question": question,
                "answer": "No matching documents found in the index.",
                "context": [],
            }

        # 3. Re-rank chunks (MMR) for diversity, then assemble context.
        if len(results) > 1:
            from rerank import mmr_rerank

            # Fall back to embedding only the chunks whose vectors are missing
            # from the store (e.g. legacy rows written before the embedding col).
            if any(not v for v in chunk_vectors):
                missing_texts = [
                    c["text"] for c, v in zip(results, chunk_vectors) if not v
                ]
                for emb in e.embed(missing_texts):
                    # Fill the first empty slot in order.
                    for i, v in enumerate(chunk_vectors):
                        if not v:
                            chunk_vectors[i] = emb
                            break
            results = mmr_rerank(question, query_vector, chunk_vectors, results, k=k)
    finally:
        if owned_e:
            e.__exit__(None, None, None)

    context_parts = []
    for i, chunk in enumerate(results):
        section_path = "/".join(chunk.get("section_path", ["General"]))
        context_parts.append(f"[Chunk {i+1} (source: {chunk['source_doc_id']}, "
                             f"section: {section_path}, words: {chunk['word_count']})]\n{chunk['text']}")

    # If chunk retrieval is sparse, append the top document summary as a
    # coarse-grained fallback so the LLM still has document-level context.
    if len(results) < k and doc_results:
        doc = doc_results[0]
        context_parts.append(
            f"[Document Overview (source: {doc['source_file']}, "
            f"title: {doc['title']})]\n{doc['text_summary']}"
        )

    assembled_context = "\n\n---\n\n".join(context_parts)

    # 4. Build prompt for LLM
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context.\n"
        "Use ONLY the information in the context to answer — do not make up facts.\n"
        "If the context does not contain enough information, say so clearly."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{assembled_context}"
    )

    # 5. Call LLM to generate answer (free-text, not JSON)
    from formatters import call_llm_raw

    answer = call_llm_raw(system_prompt, user_prompt).strip()

    return {
        "question": question,
        "answer": answer,
        "context": results,
    }
