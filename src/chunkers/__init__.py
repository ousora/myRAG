"""Text chunking — pure Python (no LangChain).

Uses markdown-it-py for header-aware markdown splitting with hierarchical metadata.
Oversized chunks (> chunk_size) get a secondary recursive character split with overlap.

Output format (backward compatible with existing pipeline):
    {"text": "## Section\n\ncontent...",
     "section_path": ["H2 Title"] or ["H2 Title", "H3 Sub"],
     "metadata": {"H1": "...", "H2": "...", "H3": "..."}}
"""

import re
from typing import Optional

from markdown_it import MarkdownIt


class Chunker:
    """Split markdown text into embeddable chunks using markdown-it-py.

    Primary split: header-aware splitting on #/##/### boundaries,
    preserving hierarchical metadata (H1, H2, H3) per chunk.

    Secondary split: recursive character split — applied only to chunks
    exceeding chunk_size, preserving header metadata on all sub-chunks.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        headers_to_split_on: Optional[list[tuple[str, str]]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ]

        self.headers_to_split_on = headers_to_split_on

        # Build level ↔ key mappings — H1 → 1, "H1" → 1, etc.
        self._level_to_key: dict[int, str] = {}
        self._key_to_level: dict[str, int] = {}
        for marker, key in headers_to_split_on:
            level = len(marker)
            self._level_to_key[level] = key
            self._key_to_level[key] = level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, *, section_path=None) -> list[dict]:
        """Split text into chunks with semantic context.

        Args:
            text: Markdown text to split. Falls back to plain-text splitting
                  when no markdown headers are detected.
            section_path: Ignored — section_path is auto-derived from headers.

        Returns:
            List of dicts with 'text', 'section_path', and 'metadata' keys.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        headings = self._parse_headings(text)
        lines = text.split("\n")

        if not headings:
            # Plain text — no headers detected
            return self._recursive_split(text, is_plain_text=True)

        # Split text at heading boundaries, building section metadata
        sections = self._split_by_headings(lines, headings)

        # Step 2: Apply secondary split for oversized chunks
        all_chunks: list[dict] = []
        for sec in sections:
            content = sec["text"]
            if not content:
                continue

            if len(content) > self.chunk_size:
                sub_texts = self._recursive_split_text(content)
                for sub in sub_texts:
                    sp = self._metadata_to_section_path(sec["metadata"])
                    all_chunks.append({
                        "text": sub.strip(),
                        "section_path": sp,
                        "metadata": dict(sec["metadata"]),
                    })
            else:
                sp = self._metadata_to_section_path(sec["metadata"])
                all_chunks.append({
                    "text": content,
                    "section_path": sp,
                    "metadata": dict(sec["metadata"]),
                })

        return all_chunks

    # ------------------------------------------------------------------
    # Heading parsing
    # ------------------------------------------------------------------

    def _parse_headings(self, text: str) -> list[dict]:
        """Parse markdown headings (ATX + setext) using markdown-it-py.

        Returns list of dicts with keys: level, key, title, line (0-indexed).
        markdown-it-py handles setext headers (underlined with ===/---) natively.
        """
        md = MarkdownIt()
        tokens = md.parse(text)

        headings = []
        for i, token in enumerate(tokens):
            if token.type == "heading_open":
                tag = token.tag  # 'h1', 'h2', 'h3'
                level = int(tag[1])
                hkey = self._level_to_key.get(level)
                if hkey is None:
                    continue
                title = ""
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    title = tokens[i + 1].content.strip()
                line_start = token.map[0] if token.map else 0
                headings.append({
                    "level": level,
                    "key": hkey,
                    "title": title,
                    "line": line_start,
                })

        return headings

    # ------------------------------------------------------------------
    # Section building
    # ------------------------------------------------------------------

    def _split_by_headings(self, lines: list[str], headings: list[dict]) -> list[dict]:
        """Split text at heading boundaries, tracking active heading hierarchy.

        CRITICAL: Consecutive headings with no body content between them
        are merged into one section. A boundary is created only when there
        is actual non-heading content before a new heading.

        Each section gets metadata reflecting the current heading context:
        e.g. {"H1": "Doc Title", "H2": "Section A", "H3": "Sub"}
        """
        heading_lines = {h["line"] for h in headings}
        total_lines = len(lines)

        active: dict[str, str] = {}
        sections: list[dict] = []
        section_start = 0
        last_body_line = -1  # Last line idx with non-whitespace, non-heading content

        for line_idx in range(total_lines):
            line = lines[line_idx]
            heading_at_line = None
            for h in headings:
                if h["line"] == line_idx:
                    heading_at_line = h
                    break

            if heading_at_line is not None:
                h = heading_at_line
                # Boundary only if there was body content since the last section start
                if last_body_line >= section_start:
                    text = "\n".join(lines[section_start:line_idx]).strip()
                    if text:
                        sections.append({
                            "text": text,
                            "metadata": dict(active),
                        })
                    section_start = line_idx

                # Update active hierarchy
                active[h["key"]] = h["title"]
                current_level = h["level"]
                for k in list(active.keys()):
                    if self._key_to_level.get(k, 999) > current_level:
                        del active[k]
            elif line.strip():
                last_body_line = line_idx

        # Last section (from last section_start to end)
        if section_start < total_lines:
            text = "\n".join(lines[section_start:]).strip()
            if text:
                sections.append({
                    "text": text,
                    "metadata": dict(active),
                })

        return sections

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_to_section_path(metadata: dict) -> list[str]:
        """Convert metadata dict to a flat section_path list.

        LangChain-compatible metadata: {"H1": "Title", "H2": "Section", "H3": "Sub"}
        → section_path: ["Section"] or ["Section", "Sub"]

        If there's no H1 (document title), the top-level header is used directly.
        Otherwise H1 is stripped — section_path starts at H2.
        """
        parts = []
        for key in ("H1", "H2", "H3"):
            val = metadata.get(key, "").strip()
            if val:
                parts.append(val)

        if not parts:
            return ["General"]

        # If H1 exists, strip it (it's the doc title, not a section)
        has_h1 = bool(metadata.get("H1", "").strip())
        if has_h1 and len(parts) > 1:
            return parts[1:]
        return parts

    @staticmethod
    def _render_section_header(section_path: list[str]) -> str:
        """Render markdown headers from section_path.

        H2 for single-level, H3+ for nested (H1 reserved for doc title).
        """
        if not section_path or section_path == ["General"]:
            return ""
        prefix = "#" * (len(section_path) + 1)
        return "\n\n".join(f"{prefix} {s}" for s in section_path)

    # ------------------------------------------------------------------
    # Recursive split (fallback for oversized chunks)
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, *, is_plain_text: bool = False) -> list[dict]:
        """Split plain text or oversized sections by semantic boundaries.

        Tries paragraph → sentence → character-level boundaries.
        When is_plain_text=True, section_path is ["General"].
        """
        if is_plain_text and len(text) <= self.chunk_size:
            return [{
                "text": text.strip(),
                "section_path": ["General"],
                "metadata": {},
            }]

        sub_texts = self._recursive_split_text(text)
        section_path = ["General"] if is_plain_text else []

        return [
            {
                "text": t.strip(),
                "section_path": list(section_path),
                "metadata": {},
            }
            for t in sub_texts
        ]

    def _recursive_split_text(self, text: str) -> list[str]:
        """Split text into sub-chunks respecting semantic boundaries.

        Separator priority: paragraphs → sentences → character.
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        # Try paragraph split first
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        chunks = self._merge_segments(paragraphs)

        # If any chunk still oversized, split by sentence boundary
        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                final_chunks.extend(self._split_by_sentence(chunk))
            else:
                final_chunks.append(chunk)

        # Apply overlap
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._apply_overlap(final_chunks)

        return final_chunks

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """Merge segments into chunks of at most chunk_size."""
        chunks: list[str] = []
        current = ""

        for seg in segments:
            # If the segment itself is oversized, flush current and split
            if len(seg) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(seg)
                continue

            candidate = (current + "\n\n" + seg).strip() if current else seg
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = seg

        if current:
            chunks.append(current)

        return chunks

    def _split_by_sentence(self, text: str) -> list[str]:
        """Split text at sentence boundaries (Chinese + English punctuation)."""
        # Sentence boundary pattern: Chinese punctuation or English sentence end
        sentences = re.split(
            r"(?<=[。！？.!?])\s*",
            text,
        )
        sentences = [s.strip() for s in sentences if s.strip()]

        # Re-merge sentences into chunks
        chunks: list[str] = []
        current = ""

        for sent in sentences:
            if len(sent) > self.chunk_size:
                # Single sentence too long — flush and fall back to char split
                if current:
                    chunks.append(current)
                    current = ""
                # Character-level fallback for this sentence
                for i in range(0, len(sent), self.chunk_size):
                    chunks.append(sent[i:i + self.chunk_size])
                continue

            candidate = (current + sent).strip() if current else sent
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sent

        if current:
            chunks.append(current)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply chunk_overlap by prepending tail of previous chunk."""
        if self.chunk_overlap <= 0:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap:]
            result.append(prev_tail + chunks[i])
        return result


def chunk_text(text: str, **kwargs) -> list[dict]:
    """Convenience wrapper."""
    return Chunker(**kwargs).chunk(text)
