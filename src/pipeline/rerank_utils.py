"""MMR re-ranking and missing-vector filling helpers for rag_query.

Extracted from ``pipeline.hybrid`` to keep that module under 500 lines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embedders.bge_m3 import Embedder

logger = logging.getLogger(__name__)


def fill_missing_chunk_vectors(
    chunk_vectors: list[list[float]],
    results: list[dict[str, Any]],
    embedder: Embedder,
) -> list[list[float]]:
    """Fill embedding vectors missing from the store.

    Legacy rows written before the ``embedding`` column existed may lack
    vectors.  This function embeds only the missing chunks and inserts the
    results back into *chunk_vectors* at the correct indices.

    Args:
        chunk_vectors: Existing vectors (possibly empty lists for missing).
        results: Retrieved chunk dicts, aligned with *chunk_vectors*.
        embedder: Active embedder instance for on-demand embedding.

    Returns:
        The updated *chunk_vectors* list (mutated in place).

    """
    from embedders.bge_m3 import chunk_embed_text

    missing_texts = [
        chunk_embed_text(c)
        for c, v in zip(results, chunk_vectors, strict=True) if not v
    ]
    if not missing_texts:
        return chunk_vectors

    embedded = embedder.embed(missing_texts)
    if not isinstance(embedded, list) or not embedded:
        return chunk_vectors
    for emb in embedded:
        if not isinstance(emb, list) or not emb or isinstance(emb[0], list):
            continue
        # Fill the first empty slot in order.
        for i, v in enumerate(chunk_vectors):
            if not v:
                chunk_vectors[i] = emb
                break

    return chunk_vectors


def rerank_with_mmr(
    question: str,
    query_vector: list[float],
    chunk_vectors: list[list[float]],
    results: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Re-rank *results* by Maximal Marginal Relevance.

    Args:
        question: Original query string.
        query_vector: Embedded query vector.
        chunk_vectors: Embeddings of each result chunk.
        results: Retrieved chunk dicts.
        k: Number of chunks to keep.

    Returns:
        Re-ordered list of *results* (a subset of length <= k).

    """
    from rerank import mmr_rerank

    return mmr_rerank(question, query_vector, chunk_vectors, results, k=k)
