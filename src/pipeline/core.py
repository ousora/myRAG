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
       [chunk] → bge-m3 embedding → FAISS/Milvus vector DB
    
    B: Document-level index — coarse-grained context fallback
       [doc_summary] → bge-m3 embedding → FAISS/Milvus vector DB

Usage (traditional RAG):
    from pipeline import process_file, process_directory
    
    chunks = process_file("path/to/report.pdf")  # traditional chunking only

Usage (LLM-formatted + Hybrid A+B):
    from pipeline import process_file_hybrid
    
    result = process_file_hybrid(
        filepath="path/to/document.pdf",
        doc_id="doc_001"
    )
    # Returns: {
    #   "chunks": list[dict]  (A - fine-grained, ready for FAISS/Milvus)
    #   "document": dict      (B - coarse-grained, ready for FAISS/Milvus)
    # }

Usage (LLM-formatted + Markdown output):
    from pipeline import process_file_with_md
    
    md_path = process_file_with_md(
        filepath="path/to/document.pdf",
        output_dir="./output/",
    )
"""


import httpx
import logging
from pathlib import Path

# Trigger parser registration at module load time
import parsers  # noqa: F401 — loads dispatcher (MarkItDown + Trafilatura)

logger = logging.getLogger(__name__)


def _render_markdown_with_sections(result: dict) -> str:
    """Build markdown text with proper ##/### headers from metadata.sections.

    The LLM formatter's body field may or may not contain markdown headers
    (it's non-deterministic). This function guarantees headers by rendering
    them from metadata.sections, which is the reliable structured source.
    """
    lines = [f"# {result.get('title', 'Untitled')}"]

    for section in result.get("metadata", {}).get("sections", []):
        level = section.get("level", 2)
        prefix = "#" * level
        lines.append(f"{prefix} {section['title']}")

    lines.append("")
    lines.append(result.get("body", ""))
    return "\n\n".join(lines) + "\n"


def _resolve_parser(filepath: str):
    from parsers.dispatcher import resolve_parser as rp
    return rp(filepath)


class TextCleaner:
    """Facade — delegates to parsers.text_cleaner.TextCleaner.
    
    Kept as a class in pipeline.py for backward compatibility with existing callers.
    The actual implementation lives in parsers/text_cleaner.py (YAML config support).
    """

    def __init__(self, *, remove_page_breaks=True, collapse_whitespace=True):
        from parsers.text_cleaner import TextCleaner as _RealCleaner  # noqa: F811
        self._cleaner = _RealCleaner(
            remove_page_breaks=remove_page_breaks,
            collapse_whitespace=collapse_whitespace,
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


def process_file(filepath: str, *, remove_page_breaks=True, collapse_whitespace=True, chunk_size=512) -> list[dict]:
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
    cleaned = TextCleaner(remove_page_breaks=remove_page_breaks, collapse_whitespace=collapse_whitespace).clean(raw_text)
    chunks = Chunker(chunk_size=chunk_size).chunk(cleaned)

    result = [
        {"text": chunk["text"], "metadata": {"source": filepath}}
        for chunk in chunks
    ]
    logger.info("  → %d chunks from %s", len(result), Path(filepath).name)
    return result


def process_file_hybrid(filepath: str, *, doc_id="doc_0", remove_page_breaks=True, 
                        collapse_whitespace=True, chunk_size=512, store_path=None):
    """Parse file with LLM formatter → chunker → embedder → sqlite-vec (Hybrid A+B).

    Args:
        filepath: Path to the document file.
        doc_id: Unique identifier for this document in the index.
        chunk_size: Max characters per chunk (LangChain splitter).
        store_path: Optional path to sqlite-vec database. If provided, chunk
                    vectors are persisted for later retrieval.

    Returns dict with:
        chunks  — list of dicts with embedding data (A - fine-grained)
        document — single dict with summary + embedding (B - coarse-grained)
        db_path  — path to sqlite-vec DB if store_path was provided, else None
    """
    from formatters import format_text_async
    
    # 1. Parse & Clean
    parser = _resolve_parser(filepath)
    if parser is None:
        logger.warning("Skipped %s — no parser found", filepath)
        return {"chunks": [], "document": {}}

    raw_text = parser.parse(filepath)
    cleaned = TextCleaner(remove_page_breaks=remove_page_breaks, collapse_whitespace=collapse_whitespace).clean(raw_text)
    
    # 2. LLM Format (async)
    future = format_text_async(cleaned, source_type="pdf")
    result = future.result(timeout=3600)

    # 3. Render markdown with headers from metadata.sections, then chunk
    formatted_md = _render_markdown_with_sections(result)
    chunker = Chunker(chunk_size=chunk_size)
    all_chunks = chunker.chunk(formatted_md)

    # Build summary text before the try block so it's always available in the except handler.
    title = result.get("title", "Untitled")
    tags = result.get("tags", [])
    summary_text = f"Title: {title}\nTags: {' '.join(tags)}\n{cleaned[:500]}"

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
        "md_path": None,               # TODO: write_to_md() if configured
        "db_path": db_path,             # sqlite-vec DB path (None if not persisted)
    }


def process_file_with_md(filepath: str, *, output_dir="./output/", **kwargs):
    """Parse file → LLM formatter → write structured markdown to output/.

    Returns the path of the generated .md file.
    
    This is the user-facing pipeline for generating human-readable documents.
    For vector DB indexing (Hybrid A+B), use process_file_hybrid() instead.
    """
    from formatters import format_text_async, write_to_md
    
    # Parse & Clean
    parser = _resolve_parser(filepath)
    if parser is None:
        logger.warning("Skipped %s — no parser found", filepath)
        return None

    raw_text = parser.parse(filepath)
    cleaned = TextCleaner(**kwargs).clean(raw_text)
    
    # LLM Format
    future = format_text_async(cleaned, source_type="pdf")
    result = future.result(timeout=3600)
    
    # Write markdown to output_dir
    md_path = write_to_md(result, output_dir)
    
    return md_path


def process_directory(dirpath: str, *, extensions=None, chunk_size=512, **kwargs) -> list[dict]:
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

    # 1. Embed the query (embed() returns list[list[float]]; extract first element for single text)
    e = Embedder()
    query_vectors = e.embed(question)
    query_vector = query_vectors[0]

    # 2. Retrieve relevant chunks
    db = SQLiteVecStore(db_path)
    try:
        results = db.search_chunks(query_vector, k=k)
    finally:
        db.close()

    if not results:
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
                             f"section: {section_path}, words: {chunk['word_count']}])\n{chunk['text']}")

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
