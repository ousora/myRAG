"""Text chunking — split cleaned text into embeddable chunks."""


class Chunker:
    """Split text into overlapping chunks for embedding.

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


def chunk_text(text: str, **kwargs) -> list[dict]:
    """Convenience wrapper."""
    return Chunker(**kwargs).chunk(text)
