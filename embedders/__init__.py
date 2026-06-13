"""Embedding client — call your local bge-m3 service."""

from .bge_m3 import Embedder, embed_texts


__all__ = ["Embedder", "embed_texts"]