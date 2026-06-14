"""Text cleaning utilities for RAG pipelines.

Facade — delegates to parsers.text_cleaner.TextCleaner (canonical implementation with YAML config support).
The cleaners module exists for backward compatibility; new code should import from parsers.text_cleaner.
"""


from parsers.text_cleaner import TextCleaner  # noqa: F401, re-exported

# Convenience function — delegates to the canonical class above
def clean_text(text: str, *, remove_page_breaks=True, collapse_whitespace=True) -> str:
    """Convenience function to apply cleaning steps."""
    return TextCleaner(
        remove_page_breaks=remove_page_breaks,
        collapse_whitespace=collapse_whitespace,
    ).clean(text)


__all__ = ["TextCleaner", "clean_text"]
