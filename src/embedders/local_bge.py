"""Local bge-m3 embedding via sentence-transformers (no server needed).

Usage:
    from embedders.local_bge import LocalEmbedder

    e = LocalEmbedder(model_name="BAAI/bge-m3")
    emb = e.embed("你好世界")        # → list[list[float]], 1024-d
"""

from __future__ import annotations

import logging

from config import get_config
from myrag.cjk import count_cjk

from .bge_m3 import (
    _embed_cache_get,
    _embed_cache_put,
    _hash_embed,
    _validate_embedding_dimension,
    chunk_embed_text,
)

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """bge-m3 embedding using a local sentence-transformers model.

    Downloads model from HuggingFace Hub on first use (~1.1 GB for bge-m3).
    CPU-only by default; auto-detects CUDA if available.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
        max_tokens_per_batch: int | None = None,
        local_files_only: bool | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = (
                "sentence-transformers is not installed. Either run "
                "'uv sync --extra local-embeddings', switch embedding.mode to "
                "'remote', or set it to 'none' for keyword-only offline retrieval."
            )
            raise RuntimeError(msg) from exc

        # Avoid HuggingFace Hub HEAD requests when the model is already cached
        # locally (or when running offline). Defaults to the HF_HUB_OFFLINE env
        # var so CI / air-gapped environments skip network verification entirely.
        if local_files_only is None:
            import os
            local_files_only = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
        self._model = SentenceTransformer(
            model_name,
            device=device or "cpu",
            local_files_only=local_files_only,
        )
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch or 512 * 32  # ~16K tokens

    def __enter__(self) -> LocalEmbedder:
        """Return self so the model can be used in ``with`` blocks."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """No-op — sentence-transformers holds no handles to release."""

    def close(self) -> None:
        """No-op: sentence-transformers holds no open handles to release.

        Mirrors ``Embedder.close`` so callers can treat both backends
        uniformly (e.g. ``finally: e.close()`` in pipeline code).
        """

    # ── Attributes that tests and callers expect on both backends ────────
    # ``model`` mirrors Embedder.model so create_embedder() can return either
    # backend without callers needing to branch.
    model: str = "BAAI/bge-m3"

    def store_chunk(
        self,
        chunk_text: str,
        *,
        section_path: list[str] | None = None,
        doc_id: str = "doc_0",
        chunk_idx: int = 0,
    ) -> dict:
        """Embed a single chunk and return metadata for storage.

        Mirrors ``embedders.bge_m3.Embedder.store_chunk`` so the same
        store code works for both remote and local backends.
        """
        embedding = self.embed(chunk_text)

        return {
            "text": chunk_text.strip(),
            "section_path": section_path or ["General"],
            "source_doc_id": doc_id,
            "chunk_index": chunk_idx,
            "word_count": len(chunk_text.split()),
            "embedding": embedding,
        }

    # ── Public API (mirrors embedders.bge_m3.Embedder) ────────────────────

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Get embeddings for one or multiple texts.

        Returns:
            - str input: list[float] (single embedding)
            - list[str] input: list[list[float]] (batch embeddings)

        """
        if isinstance(text, str):
            cached = _embed_cache_get(text)
            if cached is not None:
                return cached
            try:
                emb: list[float] = self._model.encode(text).tolist()
            except Exception as exc:
                if getattr(get_config(), "embedding_hash_fallback", False):
                    logger.warning("Local embed failed (%s); using hash fallback", exc, exc_info=True)
                    # Bypass the cache: fallback vectors are non-semantic and
                    # must not outlive the model failure.
                    fallback = _hash_embed(text)
                    _validate_embedding_dimension(fallback)
                    return fallback
                raise
            _validate_embedding_dimension(emb)
            _embed_cache_put(text, emb)
            return emb

        # Per-text cache lookups first: only uncached texts reach the model,
        # so re-ingesting unchanged chunks costs nothing.
        slot_by_idx: dict[int, list[float]] = {}
        pending_idx: list[int] = []
        for i, t in enumerate(text):
            cached = _embed_cache_get(t)
            if cached is not None:
                slot_by_idx[i] = cached
            else:
                pending_idx.append(i)

        effective_bs = self._adaptive_batch_size([text[i] for i in pending_idx])

        try:
            for start in range(0, len(pending_idx), effective_bs):
                idxs = pending_idx[start:start + effective_bs]
                batch = [text[i] for i in idxs]
                embeddings = self._encode_with_oom_fallback(batch)
                for j, vec in zip(idxs, embeddings, strict=True):
                    _validate_embedding_dimension(vec)
                    _embed_cache_put(text[j], vec)
                    slot_by_idx[j] = vec
        except Exception as exc:
            if getattr(get_config(), "embedding_hash_fallback", False):
                logger.warning("Local batch embed failed (%s); using hash fallback", exc, exc_info=True)
                return [_hash_embed(t) for t in text]
            raise

        return [slot_by_idx[i] for i in range(len(text))]

    def _encode_with_oom_fallback(self, batch: list[str]) -> list[list[float]]:
        """Encode *batch*; on GPU/CPU OOM, bisect down to single items."""
        try:
            result: list[list[float]] = self._model.encode(batch).tolist()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(batch) == 1:
                raise
            logger.warning("OOM on %d-item batch; splitting", len(batch))
            mid = max(1, len(batch) // 2)
            return (
                self._encode_with_oom_fallback(batch[:mid])
                + self._encode_with_oom_fallback(batch[mid:])
            )
        else:
            return result

    def _maybe_prepend_instruction(self, text: str | list[str]) -> str | list[str]:
        """Prepend the retrieval query instruction for query texts only.

        Mirrors ``Embedder._maybe_prepend_instruction`` on the remote backend:
        documents are embedded without the prefix, queries with it.
        """
        from config import get_config

        instruction = getattr(get_config(), "embedding_query_instruction", "") or ""
        if not instruction:
            return text
        if isinstance(text, str):
            return f"{instruction}{text}"
        return [f"{instruction}{t}" for t in text]

    def embed_query(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Embed a user *query with the retrieval instruction prefix.

        Returns the same shape as ``embed``: ``list[float]`` for str input,
        ``list[list[float]]`` for list[str] input.
        """
        return self.embed(self._maybe_prepend_instruction(text))

    def store_chunks(self, chunks: list[dict], *, doc_id: str = "doc_0") -> list[dict]:
        """Embed multiple chunks and return metadata for storage.

        Each input chunk must have at least ``text`` and ``section_path``.
        Returns the same chunks augmented with ``embedding``, ``source_doc_id``,
        ``chunk_index``, and ``word_count``.
        """
        texts = [chunk_embed_text(c) for c in chunks]
        embeddings = self.embed(texts) if texts else []

        results = []
        for i, chunk in enumerate(chunks):
            result = dict(chunk)
            result["source_doc_id"] = doc_id
            result["chunk_index"] = i
            result["word_count"] = len(chunk.get("text", "").split())
            if embeddings:
                result["embedding"] = embeddings[i]
            results.append(result)

        return results

    def store_document(
        self,
        title: str,
        tags: list[str],
        text_summary: str,
        source_file: str,
        total_chunks: int,
    ) -> dict:
        """Embed a document-level summary and return metadata for storage."""
        embedding = self.embed(text_summary)

        return {
            "title": title,
            "tags": tags,
            "text_summary": text_summary[:1000],
            "source_file": source_file,
            "total_chunks": total_chunks,
            "embedding": embedding,
        }

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Token estimate for bge-m3's SentencePiece tokenizer.

        Uses a multi-language-aware heuristic: CJK characters count as ~1 token,
        ASCII alphanumerics average ~4 chars/token (with whitespace). This is
        more accurate than the naive ``len // 2`` which underestimates Chinese
        text by ~50%.
        """
        cjk = count_cjk(text)
        ascii_alnum = sum(1 for ch in text if ch.isascii() and ch.isalnum())
        other_ascii = len(text) - cjk - ascii_alnum

        # CJK: ~1 char/token (SentencePiece byte-level).
        # ASCII letters/digits: ~4 chars/token (digits included — numeric-heavy
        # text previously landed in the 0.5-token "other" bucket, skewing the
        # adaptive batch size).
        # Punctuation/whitespace: counted as 0.5 token each.
        return cjk + (ascii_alnum // 4) + (other_ascii // 2)

    def _adaptive_batch_size(self, texts: list[str]) -> int:
        """Dynamically reduce batch size if total tokens exceed limit."""
        estimated_total_tokens = sum(self._estimate_tokens(t) for t in texts)
        if estimated_total_tokens > self.max_tokens_per_batch:
            scale = self.max_tokens_per_batch / max(estimated_total_tokens, 1)
            return max(4, int(self.batch_size * scale))
        return self.batch_size
