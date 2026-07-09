"""Re-ranking of retrieved chunks.

Implements Maximal Marginal Relevance (MMR) re-ranking: it trades off
relevance against diversity so the top-k context blocks are not near-duplicates
of each other. Relevance is estimated from lexical overlap with the query;
diversity uses chunk embeddings (cosine) so semantically redundant chunks are
down-ranked.

This is a dependency-free, model-agnostic re-ranker — a lightweight alternative
to a cross-encoder reranker. It only re-orders chunks already retrieved by the
vector + FTS hybrid search; it does not change recall.
"""

import logging
import re

logger = logging.getLogger(__name__)


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[a-zA-Z]{2,}")


def _tokenize(text: str) -> list[str]:
    """Tokenize into lowercase Latin words + individual CJK characters."""
    tokens = [t.lower() for t in _LATIN_RE.findall(text)]
    tokens += list(_CJK_RE.findall(text))
    return tokens


def _lexical_score(query: str, text: str) -> float:
    """Normalized lexical overlap (query-token recall) in [0, 1]."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    t_freq = {}
    for t in _tokenize(text):
        t_freq[t] = t_freq.get(t, 0) + 1
    overlap = sum(min(q_tokens.count(t), t_freq.get(t, 0)) for t in set(q_tokens))
    return overlap / len(q_tokens)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity with on-the-fly L2 normalization."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def mmr_rerank(
    query_text: str,
    query_vector: list[float],
    chunk_vectors: list[list[float]],
    candidates: list[dict],
    *,
    k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Re-rank *candidates* by Maximal Marginal Relevance.

    Args:
        query_text: Original query string (for lexical relevance).
        query_vector: Embedded query (unused for relevance here, kept for API
                      symmetry / future query-side scoring).
        chunk_vectors: Embeddings of each candidate chunk, aligned by index.
        candidates: Retrieved chunk dicts (e.g. from hybrid_search).
        k: Number of chunks to keep.
        lambda_param: 1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        Re-ordered list of *candidates* (a subset of length <= k).
    """
    if not candidates:
        return []
    if len(candidates) <= k:
        # Nothing to drop — order by relevance only.
        scored = sorted(
            candidates,
            key=lambda c: _lexical_score(query_text, c.get("text", "")),
            reverse=True,
        )
        return scored

    relevance = [_lexical_score(query_text, c.get("text", "")) for c in candidates]
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while len(selected) < k and remaining:
        best_idx, best_score = remaining[0], None
        for idx in remaining:
            max_sim = (
                max((_cosine(chunk_vectors[idx], chunk_vectors[s]) for s in selected), default=0.0)
                if selected
                else 0.0
            )
            mmr = lambda_param * relevance[idx] - (1.0 - lambda_param) * max_sim
            if best_score is None or mmr > best_score:
                best_score, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    logger.info("MMR re-ranked %d candidates → top %d", len(candidates), len(selected))
    return [candidates[i] for i in selected]
