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


import json
import logging
import logging.handlers
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
    
    # 4. Embed + optionally persist to sqlite-vec
    db_path = None
    try:
        from embedders import Embedder
    
        e = Embedder()
        stored_chunks = e.store_chunks(all_chunks, doc_id=doc_id)
        
        # Document-level index (B)
        summary_text = f"Title: {result.get('title', '')}\nTags: {' '.join(result.get('tags', []))}\n{cleaned[:500]}"
        stored_doc = e.store_document(
            title=result.get("title", "Untitled"),
            tags=result.get("tags", []),
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
                title=result.get("title", "Untitled"),
                tags=result.get("tags", []),
                text_summary=summary_text,
                source_file=filepath,
                total_chunks=len(stored_chunks),
                embedding=doc_embedding,
            )
            
            db_path = store_path
            logger.info("  → Persisted %d chunks + 1 doc to %s", len(stored_chunks), store_path)
        
    except Exception as exc:
        logger.warning("Embedding/storage failed: %s", exc)
        stored_chunks = [{"text": c["text"], "section_path": c.get("section_path", ["General"]), 
                          "source_doc_id": doc_id, "chunk_index": i} for i, c in enumerate(all_chunks)]
        stored_doc = {
            "title": result.get("title", "Untitled"),
            "tags": result.get("tags", []),
            "text_summary": summary_text[:500] if 'summary_text' in dir() else "",
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


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG data cleanup pipeline")
    subparsers = parser.add_subparsers(dest="command")
    
    # process_file command (traditional)
    file_parser = subparsers.add_parser("process-file", help="Process a single file (traditional)")
    file_parser.add_argument("input", help="file to process")
    file_parser.add_argument("--chunk-size", type=int, default=512)
    
    # process_directory command (batch traditional)
    dir_parser = subparsers.add_parser("process-directory", help="Process all files in directory")
    dir_parser.add_argument("input", help="directory to process")
    dir_parser.add_argument("--chunk-size", type=int, default=512)
    
    # hybrid command (A+B indexing)
    hybrid_parser = subparsers.add_parser("hybrid", help="Process with LLM formatter + Hybrid A+B indexing")
    hybrid_parser.add_argument("input", help="file to process")
    hybrid_parser.add_argument("--doc-id", default="doc_0", help="Document ID for storage")
    hybrid_parser.add_argument("--store", default=None, help="Path to sqlite-vec database for persistence")
    
    # md command (generate markdown output)
    md_parser = subparsers.add_parser("md", help="Generate structured markdown output")
    md_parser.add_argument("input", help="file to process")
    md_parser.add_argument("--output-dir", default="./output/", help="Output directory for .md files")

    args = parser.parse_args()

    # Setup logging: console + file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "pipeline.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(console)

    # File handler (append, 5MB max, keep 3 backups)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(fh)

    logger.info("Logging to %s", log_file.resolve())

    if args.command == "process-file":
        chunks = process_file(args.input, chunk_size=args.chunk_size)
        print(json.dumps(chunks, indent=2, ensure_ascii=False))
        
    elif args.command == "process-directory":
        chunks = process_directory(args.input, chunk_size=args.chunk_size)
        print(json.dumps(chunks, indent=2, ensure_ascii=False))
        
    elif args.command == "hybrid":
        result = process_file_hybrid(args.input, doc_id=args.doc_id, store_path=args.store)
        print(f"Chunks: {len(result['chunks'])}")
        if result.get("db_path"):
            print(f"DB:     {result['db_path']}")
        print("Document index created")
        if result["format_result"]:
            print(f"Title: {result['format_result'].get('title', 'N/A')}")
            
    elif args.command == "md":
        path = process_file_with_md(args.input, output_dir=args.output_dir)
        if path:
            print(f"Written to: {path}")


if __name__ == "__main__":
    main()
