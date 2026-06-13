"""RAG data cleanup pipeline — parse → clean → chunk → embed.

Usage:
    from myrag.pipeline import process_file, process_directory
    
    # Parse a single file
    chunks = process_file("path/to/report.pdf")
    
    # Or walk an entire directory
    all_chunks = process_directory("./docs", max_chars=512)
"""

import json
import logging
from pathlib import Path

# Trigger parser registration at module load time
import myrag.parsers  # noqa: F401 — loads all parsers via __init__ loop

logger = logging.getLogger(__name__)


def _resolve_parser(filepath: str):
    from myrag.parsers.dispatcher import resolve_parser as rp
    return rp(filepath)


class TextCleaner:
    """Apply a series of cleaning steps to raw extracted text."""

    def __init__(self, *, remove_page_breaks=True, collapse_whitespace=True):
        self.remove_page_breaks = remove_page_breaks
        self.collapse_whitespace = collapse_whitespace

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        import re
        if self.remove_page_breaks:
            page_pattern = r"(?:^|\n)\s*[-=\*_]{3,}\s*(PAGE\s*\d+\s*)?(?:\n|$)"
            text = re.sub(page_pattern, "\n", text)

        if self.collapse_whitespace:
            text = re.sub(r"[ \t\v\f]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()


class Chunker:
    """Split text into overlapping chunks for embedding."""

    def __init__(self, *, max_chars=512, min_chunk_chars=3, overlap_chars=64):
        self.max_chars = max_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str) -> list[str]:
        if not isinstance(text, str) or len(text.strip()) < self.min_chunk_chars:
            return []

        chunks: list[str] = []
        start = 0
        while start + self.max_chars < len(text):
            end = min(start + self.max_chars, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_chunk_chars:
                chunks.append(chunk_text)
            start += max(self.max_chars - self.overlap_chars, 1)

        remaining = text[start:].strip()
        if remaining and len(remaining) >= self.min_chunk_chars:
            chunks.append(remaining)

        return chunks


def process_file(filepath: str, *, remove_page_breaks=True, collapse_whitespace=True, max_chars=512) -> list[dict]:
    """Parse a single file and return structured chunks.

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
        {"text": chunk.strip(), "metadata": {"source": filepath}}
        for chunk in chunks
    ]
    logger.info("  → %d chunks from %s", len(result), Path(filepath).name)
    return result


def process_directory(dirpath: str, *, extensions=None, **kwargs) -> list[dict]:
    """Walk a directory and process all supported files."""
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
    parser.add_argument("input", help="file or directory to process")
    parser.add_argument("--max-chars", type=int, default=512)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    chunks = process_directory(args.input, max_chars=args.max_chars)
    print(json.dumps(chunks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
