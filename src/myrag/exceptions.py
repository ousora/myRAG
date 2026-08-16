"""Unified exception hierarchy for myRAG pipeline.

Provides typed exceptions for common failure modes across the RAG pipeline:
- Parser errors (file format detection, parsing)
- Embedding errors (model loading, API failures)
- Chunking errors (text segmentation)
- Formatting errors (LLM response validation, JSON extraction)
- Storage errors (SQLite/vector store operations)

All exceptions inherit from ``MyRagException`` for easy catching.
"""


class MyRagException(Exception):
    """Base class for all myRAG pipeline exceptions."""

    def __init__(self, message: str = "", context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ParserNotFoundError(MyRagException):
    """Raised when no parser can handle the given file extension."""

    def __init__(self, filepath: str, available_parsers: list[str] | None = None):
        msg = f"No parser found for '{filepath}'"
        if available_parsers:
            msg += f"\nAvailable parsers: {', '.join(available_parsers)}"
        super().__init__(msg)


class EmbeddingError(MyRagException):
    """Raised when embedding generation fails (model loading, API errors)."""

    def __init__(self, message: str = "", model_name: str | None = None, context: dict | None = None):
        self.model_name = model_name
        super().__init__(message, context=context)


class ChunkingError(MyRagException):
    """Raised when text chunking/segmentation fails."""

    def __init__(self, message: str = "", text_length: int | None = None):
        if text_length is not None:
            message += f" (text length: {text_length} chars)"
        super().__init__(message)


class FormattingError(MyRagException):
    """Raised when LLM formatting/output validation fails."""

    def __init__(self, message: str = "", llm_response: dict | None = None):
        self.llm_response = llm_response or {}
        super().__init__(message)


class StorageError(MyRagException):
    """Raised when SQLite/vector store operations fail."""

    def __init__(self, operation: str = "", message: str = ""):
        msg = f"Storage {operation} failed:" if operation else "Storage error:"
        if message:
            msg += f" {message}"
        super().__init__(msg)
