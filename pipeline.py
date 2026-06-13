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
    from myrag.pipeline import process_file, process_directory
    
    chunks = process_file("path/to/report.pdf")  # traditional chunking only

Usage (LLM-formatted + Hybrid A+B):
    from myrag.pipeline import process_file_hybrid
    
    result = process_file_hybrid(
        filepath="path/to/document.pdf",
        doc_id="doc_001"
    )
    # Returns: {
    #   "chunks": list[dict]  (A - fine-grained, ready for FAISS/Milvus)
    #   "document": dict      (B - coarse-grained, ready for FAISS/Milvus)
    # }

Usage (LLM-formatted + Markdown output):
    from myrag.pipeline import process_file_with_md
    
    md_path = process_file_with_md(
        filepath="path/to/document.pdf",
        output_dir="./output/",
    )
"""


import json
import logging
from pathlib import Path

# Trigger parser registration at module load time
import myrag.parsers  # noqa: F401 — loads dispatcher (MarkItDown + Trafilatura)

logger = logging.getLogger(__name__)


def _resolve_parser(filepath: str):
    from myrag.parsers.dispatcher import resolve_parser as rp
    return rp(filepath)


class TextCleaner:
    """Facade — delegates to parsers.text_cleaner.TextCleaner.
    
    Kept as a class in pipeline.py for backward compatibility with existing callers.
    The actual implementation lives in parsers/text_cleaner.py (YAML config support).
    """

    def __init__(self, *, remove_page_breaks=True, collapse_whitespace=True):
        from myrag.parsers.text_cleaner import TextCleaner as _RealCleaner  # noqa: F811
        self._cleaner = _RealCleaner(
            remove_page_breaks=remove_page_breaks,
            collapse_whitespace=collapse_whitespace,
        )

    def clean(self, text: str) -> str:
        return self._cleaner.clean(text)


class Chunker:
    """Split text into overlapping chunks with semantic context.

    Accepts section_path from LLM formatter to generate proper headers.
    Each output dict contains the chunked text along with its semantic context.
    """

    def __init__(self, *, max_chars=512, min_chunk_chars=3, overlap_chars=64):
        self.max_chars = max_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_chars = overlap_chars

    def _render_section_header(self, section_path: list[str]) -> str:
        """Render a markdown header from the section path."""
        if not section_path:
            return ""
        
        # Use H3 for section-level headers (H2 is reserved for document title)
        prefix = "##" if len(section_path) == 1 else f"{'# ' * min(len(section_path), 4).rstrip()}"
        return "\n\n".join(f"{prefix} {s}" for s in section_path)

    def chunk(self, text: str, *, section_path=None) -> list[dict]:
        """Split text into chunks with semantic context.

        Args:
            text: Raw text content to split
            section_path: List of section titles from LLM formatter
                         e.g., ["What Is Changing?", "Structured Address"]

        Returns:
            List of dicts with 'text' and 'section_path' keys.
        """
        if not isinstance(text, str) or len(text.strip()) < self.min_chunk_chars:
            return []

        header = self._render_section_header(section_path) if section_path else ""
        prefix_text = f"{header}\n\n" if header else ""
        
        # Build list of (chunk_index, chunk_text) pairs
        chunks: list[dict] = []
        start = 0
        
        while start + self.max_chars < len(text):
            end = min(start + self.max_chars, len(text))
            raw_chunk = text[start:end].strip()
            
            if len(raw_chunk) >= self.min_chunk_chars:
                chunk_text = f"{prefix_text}{raw_chunk}"
                chunks.append({
                    "text": chunk_text,
                    "section_path": section_path or ["General"],
                })
            
            start += max(self.max_chars - self.overlap_chars, 1)

        # Handle remaining text
        remaining = text[start:].strip()
        if remaining and len(remaining) >= self.min_chunk_chars:
            chunk_text = f"{prefix_text}{remaining}"
            chunks.append({
                "text": chunk_text,
                "section_path": section_path or ["General"],
            })

        return chunks


def process_file(filepath: str, *, remove_page_breaks=True, collapse_whitespace=True, max_chars=512) -> list[dict]:
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
    chunks = Chunker(max_chars=max_chars).chunk(cleaned)

    result = [
        {"text": chunk["text"], "metadata": {"source": filepath}}
        for chunk in chunks
    ]
    logger.info("  → %d chunks from %s", len(result), Path(filepath).name)
    return result


def process_file_hybrid(filepath: str, *, doc_id="doc_0", remove_page_breaks=True, 
                        collapse_whitespace=True, max_chars=512):
    """Parse file with LLM formatter → chunker → embedder (Hybrid A+B indexing).

    Returns dict with two indexes ready for FAISS/Milvus:
        chunks  — list of dicts with embedding data (A - fine-grained)
        document — single dict with summary + embedding (B - coarse-grained)
    
    Note: Embeddings require bge-m3 endpoint configured in embedders/bge_m3.py.
          Set base_url to your local service before calling this function.
    """
    from myrag.formatters import format_text_async
    
    # 1. Parse & Clean
    parser = _resolve_parser(filepath)
    if parser is None:
        logger.warning("Skipped %s — no parser found", filepath)
        return {"chunks": [], "document": {}}

    raw_text = parser.parse(filepath)
    cleaned = TextCleaner(remove_page_breaks=remove_page_breaks, collapse_whitespace=collapse_whitespace).clean(raw_text)
    
    # 2. LLM Format (async)
    future = format_text_async(cleaned, source_type="pdf")
    result = future.result(timeout=300)
    
    metadata = result.get("metadata", {})
    
    # 3. Chunk with section headers
    chunker = Chunker(max_chars=max_chars)
    chunks_with_headers = []
    
    for section in metadata.get("sections", []):
        title = section["title"] if isinstance(section, dict) else section
        
        # Find text belonging to this section (simplified — real impl would parse from result["chunks"])
        pass  # TODO: Map sections to actual text ranges
    
    # For now, chunk the full cleaned text with a generic header
    all_chunks = chunker.chunk(cleaned, section_path=["General"])
    
    # 4. Store in embedder (requires bge-m3 endpoint)
    try:
        from myrag.embedders import Embedder
    
        e = Embedder()  # Uses default base_url: http://192.168.191.112:11435/v1 (bge-m3)
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
        
    except Exception as exc:
        logger.warning("Embedding failed (bge-m3 endpoint not configured): %s", exc)
        stored_chunks = [{"text": c["text"], "section_path": c.get("section_path", ["General"]), 
                          "source_doc_id": doc_id, "chunk_index": i} for i, c in enumerate(all_chunks)]
        stored_doc = {
            "title": result.get("title", "Untitled"),
            "tags": result.get("tags", []),
            "text_summary": summary_text[:500],
            "source_file": filepath,
            "total_chunks": len(stored_chunks),
        }

    return {
        "chunks": stored_chunks,       # A - fine-grained index
        "document": stored_doc,         # B - coarse-grained index
        "format_result": result,        # Original LLM output (for metadata)
        "md_path": None,                # TODO: write_to_md() if configured
    }


def process_file_with_md(filepath: str, *, output_dir="./output/", **kwargs):
    """Parse file → LLM formatter → write structured markdown to output/.

    Returns the path of the generated .md file.
    
    This is the user-facing pipeline for generating human-readable documents.
    For vector DB indexing (Hybrid A+B), use process_file_hybrid() instead.
    """
    from myrag.formatters import format_text_async, write_to_md
    
    # Parse & Clean
    parser = _resolve_parser(filepath)
    if parser is None:
        logger.warning("Skipped %s — no parser found", filepath)
        return None

    raw_text = parser.parse(filepath)
    cleaned = TextCleaner(**kwargs).clean(raw_text)
    
    # LLM Format
    future = format_text_async(cleaned, source_type="pdf")
    result = future.result(timeout=300)
    
    # Write markdown to output_dir
    md_path = write_to_md(result, output_dir)
    
    return md_path


def process_directory(dirpath: str, *, extensions=None, **kwargs) -> list[dict]:
    """Walk a directory and process all supported files (traditional RAG)."""
    path = Path(dirpath)

    if extensions is None:
        from myrag.parsers.dispatcher import PARSERS
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
    file_parser.add_argument("--max-chars", type=int, default=512)
    
    # process_directory command (batch traditional)
    dir_parser = subparsers.add_parser("process-directory", help="Process all files in directory")
    dir_parser.add_argument("input", help="directory to process")
    dir_parser.add_argument("--max-chars", type=int, default=512)
    
    # hybrid command (A+B indexing)
    hybrid_parser = subparsers.add_parser("hybrid", help="Process with LLM formatter + Hybrid A+B indexing")
    hybrid_parser.add_argument("input", help="file to process")
    hybrid_parser.add_argument("--doc-id", default="doc_0", help="Document ID for storage")
    
    # md command (generate markdown output)
    md_parser = subparsers.add_parser("md", help="Generate structured markdown output")
    md_parser.add_argument("input", help="file to process")
    md_parser.add_argument("--output-dir", default="./output/", help="Output directory for .md files")

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "process-file":
        chunks = process_file(args.input, max_chars=args.max_chars)
        print(json.dumps(chunks, indent=2, ensure_ascii=False))
        
    elif args.command == "process-directory":
        chunks = process_directory(args.input, max_chars=args.max_chars)
        print(json.dumps(chunks, indent=2, ensure_ascii=False))
        
    elif args.command == "hybrid":
        result = process_file_hybrid(args.input, doc_id=args.doc_id)
        print(f"Chunks: {len(result['chunks'])}")
        print(f"Document index created")
        if result["format_result"]:
            print(f"Title: {result['format_result'].get('title', 'N/A')}")
            
    elif args.command == "md":
        path = process_file_with_md(args.input, output_dir=args.output_dir)
        if path:
            print(f"Written to: {path}")


if __name__ == "__main__":
    main()
