"""Text chunking — pure Python via markdown-it-py.

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
    # Pre-compiled MarkdownIt parser — created once in __init__ and reused.
    _MD_PARSER: MarkdownIt | None = None
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
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive (got {chunk_size})")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative (got {chunk_overlap})")

        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 4)

        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ]

        self.headers_to_split_on = headers_to_split_on
        self._md = MarkdownIt()

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

    def chunk(self, text: str) -> list[dict]:
        """Split text into chunks with semantic context.

        Args:
            text: Markdown text to split. Falls back to plain-text splitting
                  when no markdown headers are detected.

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
        tokens = self._md.parse(text)

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

        A new section starts at each heading boundary. If there is body content
        between two headings, the first heading's section captures that content.
        The next heading begins a fresh section with its own metadata context.

        Each section gets metadata reflecting the current heading context:
        e.g. {"H1": "Doc Title", "H2": "Section A", "H3": "Sub"}
        """
        total_lines = len(lines)

        # Pre-build a lookup from line index → heading dict for O(1) access.
        heading_by_line: dict[int, dict] = {h["line"]: h for h in headings}

        active: dict[str, str] = {}
        sections: list[dict] = []
        section_start = 0
        last_body_line = -1  # Last line idx with non-whitespace, non-heading content

        for line_idx in range(total_lines):
            line = lines[line_idx]
            heading_at_line = heading_by_line.get(line_idx)

            if heading_at_line is not None:
                h = heading_at_line
                # Boundary: there was body content since the last section start
                if last_body_line >= section_start:
                    text = "\n".join(lines[section_start:line_idx]).strip()
                    if text:
                        sections.append({
                            "text": text,
                            "metadata": dict(active),
                        })

                # Always advance section_start to this heading line so the next
                # heading starts a new section (even if no body between them).
                section_start = line_idx

                # Update active hierarchy for the new section context
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

    def _metadata_to_section_path(self, metadata: dict) -> list[str]:
        """Convert metadata dict to a flat section_path list.

        Metadata keys are dynamic (e.g. {"H1": "Title", "H2": "Section"})
        built from ``headers_to_split_on`` at construction time.
        → section_path: ["Section"] or ["Section", "Sub"]

        If H1 exists, it is stripped — section_path starts at the next level.
        """
        if not metadata:
            return ["General"]

        parts = []
        for key in self._level_to_key.values():
            val = metadata.get(key, "").strip()
            if val:
                parts.append(val)

        if not parts:
            return ["General"]

        # If H1 (or first-level key) exists and there are deeper levels, strip it.
        has_first = bool(metadata.get(next(iter(self._level_to_key.values()), ""), "").strip())
        if has_first and len(parts) > 1:
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

        # If any chunk still oversized, split by sentence boundary then char-level.
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                sub = self._split_by_sentence(chunk)
                # If sentence split still produces oversized pieces, do char-level.
                remaining_oversized = [c for c in sub if len(c) > self.chunk_size]
                if remaining_oversized:
                    _logger.warning(
                        "Chunk %d (%d chars) exceeded chunk_size after sentence split; using character-level fallback",
                        len(final_chunks), len(remaining_oversized[0]),
                    )
                    sub = self._split_by_char(sub)
                final_chunks.extend(sub)
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

    # Common English abbreviations that should NOT trigger a sentence split.
    _SENTENCE_ABBREVIATIONS: frozenset[str] = frozenset({
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "sgt", "cpl", "pvt",
        "gen", "adm", "col", "maj", "capt", "lt", "st", "ave",
        "blvd", "dept", "est", "inc", "ltd", "corp", "co", "vol", "vs",
        "eg", "ie", "etc", "approx", "asp", "avg", "cf", "cm", "eq",
        "fig", "govt", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec", "min", "max", "msg", "num", "opp",
        "orig", "p", "pp", "pred", "pres", "repr", "rev", "sec", "sen",
        "rep", "sq", "sra", "ssa", "us", "usa", "uk", "un", "nato",
        "who", "fbi", "cia", "na", "no", "nos", "pt", "pts",
    })

    def _is_abbreviation_boundary(self, text: str, dot_pos: int) -> bool:
        """Return True if the '.' at *dot_pos* follows a known abbreviation."""
        before = text[:dot_pos].rstrip()
        match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*\.?)$", before + ".")
        if not match:
            return False
        abbr_raw = match.group(1).lower().rstrip(".")
        candidates = {abbr_raw, re.sub(r"\.", "", abbr_raw)}
        return bool(candidates & self._SENTENCE_ABBREVIATIONS)

    def _split_by_sentence(self, text: str) -> list[str]:
        """Split text at sentence boundaries (Chinese + English punctuation).

        Known abbreviations (Mr., Dr., U.S.A., etc.) are preserved — the split
        only fires on periods that follow a non-abbreviation token.
        """
        # Step 1: Split on Chinese sentence-ending punctuation unconditionally.
        parts = re.split(r"(?<=[。！？])\s*", text)

        # Step 2: For each segment, find English '.' positions that are NOT
        # abbreviation boundaries and split there too.
        sentences: list[str] = []
        for seg in parts:
            if not seg.strip():
                continue
            if any(c in seg for c in "。！？"):
                sentences.append(seg.strip())
                continue

            # Find positions of '.' that are sentence boundaries (not abbreviations).
            boundary_positions: list[int] = []
            for i, ch in enumerate(seg):
                if ch == "." and not self._is_abbreviation_boundary(seg, i):
                    boundary_positions.append(i)

            if not boundary_positions:
                sentences.append(seg.strip())
                continue

            # Split at each sentence-boundary '.' (keep the dot with the previous piece).
            pieces = []
            prev = 0
            for bp in boundary_positions:
                pieces.append(seg[prev:bp + 1].strip())
                prev = bp + 1
            if prev < len(seg):
                pieces.append(seg[prev:].strip())

            sentences.extend(p.strip() for p in pieces if p.strip())

        return sentences

    def _split_by_char(self, segments: list[str]) -> list[str]:
        """Hard fallback: split oversized text at character boundaries.

        Used when sentence-level splitting still produces chunks larger than
        chunk_size (e.g., URLs without spaces, base64 blobs). Splits greedily
        into fixed-size pieces with overlap to preserve continuity.
        """
        result: list[str] = []
        if not segments:
            return result

        # Merge all oversized segments into one string for contiguous splitting.
        merged = "\n\n".join(s for s in segments if len(s) > self.chunk_size)
        if not merged.strip():
            return [s for s in segments if len(s) <= self.chunk_size]

        start = 0
        while start < len(merged):
            end = min(start + self.chunk_size, len(merged))
            # Try to break at whitespace near the chunk boundary.
            if end < len(merged):
                break_at = merged.rfind(" ", max(start, end - 128), end)
                if break_at > start:
                    end = break_at + 1
            result.append(merged[start:end].strip())
            # Overlap: reuse the last portion of this chunk as prefix for next.
            overlap_text = ""
            if self.chunk_overlap > 0 and len(result[-1]) > self.chunk_overlap:
                tail = result[-1][-self.chunk_overlap:]
                ws_pos = tail.rfind(" ")
                if ws_pos >= 0:
                    overlap_text = " " + tail[ws_pos + 1:]
            start = end - (len(overlap_text))

        return [c for c in result if c.strip()]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply chunk_overlap by prepending tail of previous chunk.

        Extends the overlap window back to the nearest whitespace boundary so
        that words are never split in the middle — continuity across chunks is
        preserved at natural word edges instead of arbitrary character cuts.
        Does NOT exceed chunk_size on the combined length.
        """
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1]
            # Take up to chunk_overlap chars from the end of the previous chunk.
            raw_tail = prev_text[-self.chunk_overlap:] if len(prev_text) > self.chunk_overlap else prev_text

            # Walk backwards through raw_tail finding the last whitespace char.
            overlap_start = 0
            for j in range(len(raw_tail) - 1, -1, -1):
                if raw_tail[j].isspace():
                    overlap_start = j + 1
                    break

            prev_chunk_overlap = raw_tail[overlap_start:]
            combined = prev_chunk_overlap + chunks[i]
            # If combined exceeds chunk_size, trim from the front.
            if len(combined) > self.chunk_size:
                combined = combined[-self.chunk_size:]
            result.append(combined.strip())
        return result

    def __repr__(self) -> str:
        return (f"Chunker(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})")


def chunk_text(text: str, **kwargs) -> list[dict]:
    """Convenience wrapper."""
    return Chunker(**kwargs).chunk(text)
