"""Text chunking — split cleaned text into embeddable chunks."""


class Chunker:
    """Split text into overlapping chunks for embedding.

    Accepts section_path from LLM formatter to preserve semantic context.
    Each output dict contains the chunked text along with its metadata (section_path).
    
    NOTE: Chunks do NOT render header markdown — section_path is stored in 
          metadata only, used by embedder for vector indexing. This keeps
          human-readable .md output clean.
    """

    def __init__(self, *, max_chars=512, min_chunk_chars=3, overlap_chars=64):
        self.max_chars = max_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str, section_path=None) -> list[dict]:
        """Split text into chunks with semantic context.

        Args:
            text: Raw text content to split (no header prefix needed).
            section_path: List of section titles from LLM formatter
                         e.g., ["What Is Changing?", "Structured Address"]

        Returns:
            List of dicts with 'text' and 'section_path' keys.
        """
        if not isinstance(text, str) or len(text.strip()) < self.min_chunk_chars:
            return []

        # Build list of (chunk_index, chunk_text) pairs — no header prefix
        chunks: list[dict] = []
        start = 0
        
        while start + self.max_chars < len(text):
            end = min(start + self.max_chars, len(text))
            raw_chunk = text[start:end].strip()
            
            if len(raw_chunk) >= self.min_chunk_chars:
                chunks.append({
                    "text": raw_chunk,
                    "section_path": section_path or ["General"],
                })
            
            start += max(self.max_chars - self.overlap_chars, 1)

        # Handle remaining text
        remaining = text[start:].strip()
        if remaining and len(remaining) >= self.min_chunk_chars:
            chunks.append({
                "text": remaining,
                "section_path": section_path or ["General"],
            })

        return chunks


def chunk_text(text: str, **kwargs) -> list[dict]:
    """Convenience wrapper."""
    return Chunker(**kwargs).chunk(text)