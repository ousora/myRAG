"""Pure Python markdown chunker via markdown-it-py.

Re-exports from the canonical implementation.
"""

from .chunker import Chunker, chunk_text

__all__ = ["Chunker", "chunk_text"]
