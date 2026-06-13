"""Text cleaning utilities for RAG pipelines."""


class TextCleaner:
    """Apply a series of cleaning steps to raw extracted text.

    Runs before the LLM formatter — deterministic regex-based noise removal only.
    Semantic understanding is delegated to format_text().
    """

    def __init__(self, *, remove_page_breaks=True, collapse_whitespace=True):
        self.remove_page_breaks = remove_page_breaks
        self.collapse_whitespace = collapse_whitespace

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        import re

        # Remove control characters (PDFs often contain \x07 BELL chars etc.)
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

        if self.remove_page_breaks:
            # Match page breaks: --- PAGE N ---, ===, or *** (with optional spaces from \x07→space conversion)
            text = re.sub(r"(?:^|\n)\s*[-=_\*\s]+\s*(PAGE\s+\d+\s*)?\s*[-=_\*\s]+", " ", text)

        if self.collapse_whitespace:
            text = re.sub(r"[ \t\v\f]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

        # Join lines preserving newlines between them (strip each line's trailing/leading whitespace only)
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()


def clean_text(text: str, *, remove_page_breaks=True, collapse_whitespace=True) -> str:
    """Convenience function to apply cleaning steps."""
    cleaner = TextCleaner(
        remove_page_breaks=remove_page_breaks,
        collapse_whitespace=collapse_whitespace,
    )
    return cleaner.clean(text)
