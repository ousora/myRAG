"""Text chunking — LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter.

Uses LangChain for header-aware markdown splitting with hierarchical metadata.
Oversized chunks (> chunk_size) get a secondary recursive character split with overlap.
"""

from typing import Optional

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Chunker:
    """Split markdown text into embeddable chunks using LangChain splitters.

    Primary split: MarkdownHeaderTextSplitter — splits on #/##/### boundaries,
    preserving hierarchical metadata (H1, H2, H3) per chunk.

    Secondary split: RecursiveCharacterTextSplitter — applied only to chunks
    exceeding chunk_size, preserving header metadata on all sub-chunks.

    Output format (backward compatible with existing pipeline):
        {"text": "## Section\\n\\ncontent...",
         "section_path": ["H2 Title"] or ["H2 Title", "H3 Sub"],
         "metadata": {"H1": "...", "H2": "...", "H3": "..."}}
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

        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        self._rc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def _metadata_to_section_path(metadata: dict) -> list[str]:
        """Convert LangChain metadata dict to a flat section_path list.

        LangChain metadata: {"H1": "Title", "H2": "Section", "H3": "Sub"}
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

        # Step 1: Header-aware split via MarkdownHeaderTextSplitter
        md_docs = self._md_splitter.split_text(text)

        # Detect plain text: MarkdownHeaderTextSplitter produces a single
        # Document with empty metadata when there are no #/##/### headers
        is_plain_text = (
            len(md_docs) == 1
            and not md_docs[0].metadata
            and len(md_docs[0].page_content) > self.chunk_size
        )

        if is_plain_text:
            # Fallback: plain text → recursive character split only
            sub_texts = self._rc_splitter.split_text(md_docs[0].page_content)
            return [
                {
                    "text": sub.strip(),
                    "section_path": ["General"],
                    "metadata": {},
                }
                for sub in sub_texts
            ]

        # Step 2: Secondary split for oversized chunks
        all_chunks: list[dict] = []
        for doc in md_docs:
            content = doc.page_content.strip()
            if not content:
                continue

            if len(content) > self.chunk_size:
                # Oversized — apply recursive character split
                sub_texts = self._rc_splitter.split_text(content)
                for sub in sub_texts:
                    sp = self._metadata_to_section_path(doc.metadata)
                    all_chunks.append({
                        "text": sub.strip(),
                        "section_path": sp,
                        "metadata": dict(doc.metadata),
                    })
            else:
                sp = self._metadata_to_section_path(doc.metadata)
                all_chunks.append({
                    "text": content,
                    "section_path": sp,
                    "metadata": dict(doc.metadata),
                })

        return all_chunks


def chunk_text(text: str, **kwargs) -> list[dict]:
    """Convenience wrapper."""
    return Chunker(**kwargs).chunk(text)
