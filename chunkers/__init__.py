"""Text chunking — split cleaned text into embeddable chunks."""


class Chunker:
    """Split text into overlapping chunks for embedding.

    Parses section headers (##, ###, etc.) from the input to assign each chunk
    its correct semantic context via `section_path`. If no sections are found,
    all chunks receive ["General"].

    Chunks do NOT render header markdown — section_path is stored in metadata only,
    used by embedder for vector indexing. This keeps human-readable .md output clean.
    """

    def __init__(self, *, max_chars=512, min_chunk_chars=3, overlap_chars=64):
        self.max_chars = max_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_chars = overlap_chars

    @staticmethod
    def _parse_sections(text: str) -> list[tuple[str, int]]:
        """Extract section headers and their byte offsets from markdown text."""
        import re

        sections: list[tuple[str, int]] = []
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            offset = match.start()
            sections.append((title, offset))

        return sections

    def chunk(self, text: str) -> list[dict]:
        """Split text into chunks with semantic context.

        Args:
            text: Raw markdown text to split (may contain ## headers).

        Returns:
            List of dicts with 'text' and 'section_path' keys.
        """
        if not isinstance(text, str) or len(text.strip()) < self.min_chunk_chars:
            return []

        sections = self._parse_sections(text)
        
        # Build active section tracker: for each position in text, find the most recent header
        section_tracker: list[tuple[int, str]] = [(0, "General")]  # (offset, title)
        if not sections:
            pass  # All chunks will be ["General"]
        else:
            section_tracker.extend(sections)

        chunks: list[dict] = []
        start = 0
        
        while True:
            end = min(start + self.max_chars, len(text))
            raw_chunk = text[start:end].strip()

            if not raw_chunk or len(raw_chunk) < self.min_chunk_chars:
                break
            
            # Find the most recent section header before this chunk's start position
            current_section = ["General"]
            for i in range(len(section_tracker) - 1, 0, -1):
                title, offset = section_tracker[i]
                if offset <= start:
                    current_section = [title]
                    break

            chunks.append({
                "text": raw_chunk,
                "section_path": current_section,
            })
            
            # Check if we've reached the end
            if end >= len(text):
                break
                
            start += max(self.max_chars - self.overlap_chars, 1)

        return chunks


def chunk_text(text: str, **kwargs) -> list[dict]:
    """Convenience wrapper."""
    return Chunker(**kwargs).chunk(text)
