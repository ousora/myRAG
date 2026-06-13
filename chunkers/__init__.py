"""Text chunking — split cleaned text into embeddable chunks."""


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


def chunk_text(text: str, **kwargs) -> list[str]:
    """Convenience wrapper."""
    return Chunker(**kwargs).chunk(text)
