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


import concurrent.futures
import httpx
import logging
import re
from pathlib import Path

from config import get_config_lazy as _get_config

# Trigger parser registration at module load time
import parsers  # noqa: F401 — loads dispatcher (MarkItDown + Trafilatura)

logger = logging.getLogger(__name__)


# Internal helper — not exported. Renders LLM output sections into markdown.
# If you need this, import directly from pipeline.core (not via __init__).
def _render_markdown_with_sections(result: dict) -> str:
    """Build markdown text with proper ##/### headers from metadata.sections.

    The LLM formatter's body field may or may not contain markdown headers
    (it's non-deterministic). This function guarantees headers by rendering
    them from metadata.sections, which is the reliable structured source.
    Existing headings in the body are stripped to avoid duplicates.
    """
    title = result.get("title", "Untitled")
    body = result.get("body", "") or ""

    # The LLM body is non-deterministic: it may already contain its own
    # section headings (the common case, especially for chunked output where
    # metadata.sections is derived from the body). If so, keep the body's
    # structure intact and only drop a duplicate top-level title H1 if the
    # model echoed it. Hoisting all metadata.sections headers above the body
    # would shift every section's content under the wrong header.
    if re.search(r'^#{1,6}\s+', body):
        body_lines = body.split("\n")
        kept = []
        title_seen = False
        for line in body_lines:
            stripped = line.strip()
            if not title_seen and re.match(rf'^#\s+{re.escape(title)}$', stripped):
                title_seen = True
                continue  # drop duplicate title H1; we render our own below
            kept.append(line)
        body_block = "\n".join(kept).strip()
        return f"# {title}\n\n{body_block}\n"

    # Body has no headings — render section headers from metadata.sections.
    lines = [f"# {title}"]
    for section in result.get("metadata", {}).get("sections", []):
        level = section.get("level", 2)
        prefix = "#" * level
        lines.append(f"{prefix} {section['title']}")

    lines.append("")
    lines.append(body.strip())
    return "\n\n".join(lines) + "\n"


def _match_entities_to_chunks(chunks: list[dict], entities: list[dict]) -> list[dict]:
    """Match document-level entities to individual chunks by text presence.

    Scans each chunk's text for entity names (case-insensitive).
    Only entities that actually appear in a chunk get tagged on that chunk.
    This keeps entity search granular — querying 'GPT-4' returns only chunks
    that mention GPT-4, not every chunk from the same document.

    Args:
        chunks: List of chunk dicts, each with at least a 'text' key.
        entities: List of entity dicts with 'name' keys from formatter output.

    Returns:
        Same chunks list with 'entity_names' added to each chunk.
    """
    if not entities:
        return chunks

    # Pre-classify entities: CJK names have no word boundaries, so \b never
    # matches them. For those we use a plain substring test; for Latin names we
    # keep the case-insensitive word-boundary match to avoid partial matches.
    cjk_entities = [e["name"] for e in entities if _contains_cjk(e["name"])]
    latin_entities = [e["name"] for e in entities if not _contains_cjk(e["name"])]
    latin_patterns = [
        (name, re.compile(r'\b' + re.escape(name.lower()) + r'\b')) for name in latin_entities
    ]

    for chunk in chunks:
        chunk_text_lower = chunk["text"].lower()
        matched = [name for name in cjk_entities if name.lower() in chunk_text_lower]
        matched += [name for name, pat in latin_patterns if pat.search(chunk_text_lower)]
        chunk["entity_names"] = matched
    return chunks


def _contains_cjk(text: str) -> bool:
    """Return True if *text* contains any CJK Unified Ideograph."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _resolve_parser(filepath: str):
    from parsers.dispatcher import resolve_parser as rp
    return rp(filepath)


def _source_type_for(filepath: str) -> str:
    """Map a file extension to the formatter's source_type hint."""
    ext = Path(filepath).suffix.lstrip(".").lower()
    return {
        "pdf": "pdf",
        "docx": "pdf",
        "html": "web",
        "htm": "web",
        "md": "markdown",
        "mkd": "markdown",
        "txt": "web",
    }.get(ext, "web")


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
    parser = _resolve_parser(filepath)
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
                        collapse_whitespace=True, rules_config="conf/clean_rules.yaml", chunk_size=1024, store_path=None, md_output_dir=None):
    """Parse file with LLM formatter → chunker → embedder → sqlite-vec (Hybrid A+B).

    Args:
        filepath: Path to the document file.
        doc_id: Unique identifier for this document in the index.
        chunk_size: Max characters per chunk.
        store_path: Optional path to sqlite-vec database. If provided, chunk
                    vectors are persisted for later retrieval.
        md_output_dir: Optional directory to write structured markdown (via write_to_md).

    Returns dict with:
        chunks  — list of dicts with embedding data (A - fine-grained)
        document — single dict with summary + embedding (B - coarse-grained)
        db_path  — path to sqlite-vec DB if store_path was provided, else None
        md_path  — path to generated .md file if md_output_dir was provided, else None
    """
    from formatters import format_text_async, write_to_md
    
    # 1. Parse & Clean
    parser = _resolve_parser(filepath)
    if parser is None:
        logger.warning("Skipped %s — no parser found", filepath)
        return {"chunks": [], "document": {}}

    raw_text = parser.parse(filepath)
    cleaned = TextCleaner(remove_page_breaks=remove_page_breaks, collapse_whitespace=collapse_whitespace, rules_config=rules_config).clean(raw_text)

    # 2. LLM Format (async)
    cfg = _get_config()
    future = format_text_async(cleaned, source_type=_source_type_for(filepath))
    try:
        result = future.result(timeout=cfg.format_timeout)
    except concurrent.futures.TimeoutError:
        logger.warning("LLM formatting timed out after %ds for %s", cfg.format_timeout, filepath)
        return {"chunks": [], "document": {}}

    # Write structured markdown if output_dir provided (same path as process_file_with_md)
    md_path = None
    if md_output_dir:
        md_path = write_to_md(result, md_output_dir)
        logger.info("  → Markdown written to %s", md_path)

    # 3. Render markdown with headers from metadata.sections, then chunk
    formatted_md = _render_markdown_with_sections(result)
    chunker = Chunker(chunk_size=chunk_size)
    all_chunks = chunker.chunk(formatted_md)

    # Match document-level entities to individual chunks (Phase C)
    entities = result.get("metadata", {}).get("entities", [])
    all_chunks = _match_entities_to_chunks(all_chunks, entities)

    # Build summary text before the try block so it's always available in the except handler.
    title = result.get("title", "Untitled")
    tags = result.get("tags", [])
    # Embed the LLM-formatted body (not the raw cleaned text) so the coarse-grained
    # B index captures the document's structured semantics.
    body_for_summary = result.get("body", "") or cleaned
    summary_text = f"Title: {title}\nTags: {' '.join(tags)}\n{body_for_summary[:500]}"

    # 4. Embed + optionally persist to sqlite-vec
    db_path = None
    
    try:
        from embedders import Embedder
    
        e = Embedder()
        stored_chunks = e.store_chunks(all_chunks, doc_id=doc_id)
        
        # Document-level index (B)
        stored_doc = e.store_document(
            title=title,
            tags=tags,
            text_summary=summary_text,
            source_file=filepath,
            total_chunks=len(stored_chunks),
        )

        # Persist to sqlite-vec if requested
        if store_path:
            from storage.sqlite_vec import SQLiteVecStore
            db = SQLiteVecStore(store_path)
            
            # Store chunks with embeddings
            db.upsert_chunks(stored_chunks, doc_id=doc_id)
            
            # Store document-level record
            doc_embedding = stored_doc.get("embedding")
            db.upsert_document(
                title=title,
                tags=tags,
                text_summary=summary_text,
                source_file=filepath,
                total_chunks=len(stored_chunks),
                embedding=doc_embedding,
            )
            
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

    return {
        "chunks": stored_chunks,       # A - fine-grained index
        "document": stored_doc,         # B - coarse-grained index
        "format_result": result,        # Original LLM output (for metadata)
        "md_path": md_path,            # Path to structured .md file (write_to_md output)
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
    parser = _resolve_parser(filepath)
    if parser is None:
        logger.warning("Skipped %s — no parser found", filepath)
        return None

    raw_text = parser.parse(filepath)
    cleaned = TextCleaner(**kwargs).clean(raw_text)

    # LLM Format
    future = format_text_async(cleaned, source_type=_source_type_for(filepath))
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


def rag_query(question: str, db_path: str, *, k: int = 5) -> dict:
    """Retrieve relevant chunks from sqlite-vec and generate an LLM answer.

    Args:
        question: The user's natural-language query.
        db_path: Path to the sqlite-vec database (created by process_file_hybrid).
        k: Number of top chunks to retrieve for context assembly.

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
    with Embedder() as e:
        query_result = e.embed_query(question)
    query_vector = (
        query_result[0]
        if query_result and isinstance(query_result[0], list)
        else query_result
    )

    # 2. Retrieve relevant chunks (hybrid: vector + FTS5 RRF fusion)
    db = SQLiteVecStore(db_path)
    try:
        # A: fine-grained chunk retrieval (vector + full-text).
        results = db.hybrid_search(question, query_vector=query_vector, k=k)

        # B: coarse-grained document-level fallback for broad context.
        doc_results = db.search_documents(query_vector=query_vector, k=1)
    finally:
        db.close()

    if not results and not doc_results:
        return {
            "question": question,
            "answer": "No matching documents found in the index.",
            "context": [],
        }

    # 3. Assemble context from retrieved chunks
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

    # 5. Call LLM to generate answer
    from formatters import call_llm

    llm_result = call_llm(system_prompt, user_prompt)

    # Extract the LLM's response (it returns structured JSON with a body field)
    answer = ""
    if isinstance(llm_result, dict):
        answer = llm_result.get("body", "") or llm_result.get("answer", str(llm_result))
    else:
        answer = str(llm_result)

    return {
        "question": question,
        "answer": answer.strip(),
        "context": results,
    }
