"""Text cleaning for RAG — remove noise, normalize content.

Usage (standalone):
    from myrag.cleaners import clean_text
    
Usage (via pipeline module):
    from myrag.pipeline import process_file
"""

import re


class TextCleaner:
    """Apply a series of cleaning steps to raw extracted text.

    Typical pipeline:
        1. Remove page breaks and separators inserted by PDF parsers
        2. Collapse whitespace (tabs → spaces, multiple newlines)
        3. Remove headers/footers patterns
        4. Normalize Unicode / half-width chars
    """

    def __init__(self, *, remove_page_breaks=True, collapse_whitespace=True):
        self.remove_page_breaks = remove_page_breaks
        self.collapse_whitespace = collapse_whitespace

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Page breaks / separators
        if self.remove_page_breaks:
            page_pattern = r"(?:^|\n)\s*[-=\*_]{3,}\s*(PAGE\s*\d+\s*)?(?:\n|$)"
            text = re.sub(page_pattern, "\n", text)

        # 2. Collapse whitespace
        if self.collapse_whitespace:
            # Normalize various whitespace to single space
            text = re.sub(r"[ \t\v\f]+", " ", text)
            # Collapse multiple newlines (keep paragraph breaks up to 2 blank lines)
            text = re.sub(r"\n{3,}", "\n\n", text)

        # 3. Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines).strip()

        return text


def clean_text(text: str) -> str:
    """Convenience wrapper."""
    return TextCleaner().clean(text)
