"""Embedding client — call your local bge-m3 service."""

from .bge_m3 import Embedder, create_embedder, embed_texts


__all__ = ["Embedder", "create_embedder", "embed_texts"]