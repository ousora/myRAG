"""Embedding client — call your local bge-m3 service."""

from .bge_m3 import Embedder, create_embedder, embed_texts
from .local_bge import LocalEmbedder


__all__ = ["Embedder", "LocalEmbedder", "create_embedder", "embed_texts"]