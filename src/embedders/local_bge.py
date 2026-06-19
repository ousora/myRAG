"""Local bge-m3 embedding via sentence-transformers (no server needed).

Usage:
    from embedders.local_bge import LocalEmbedder

    e = LocalEmbedder(model_name="BAAI/bge-m3")
    emb = e.embed("你好世界")        # → list[list[float]], 1024-d
"""

from __future__ import annotations

import logging

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
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_name,
            device=device or "cpu",
        )
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch or 512 * 32  # ~16K tokens

    # ── Public API (mirrors embedders.bge_m3.Embedder) ────────────────────

    def embed(self, text: str | list[str]) -> list[list[float]]:
        """Get embeddings for one or multiple texts.

        Returns:
            - str input: list[float] (single embedding) — matches remote Embedder behavior
            - list[str] input: list[list[float]] (batch embeddings)
        """
        if isinstance(text, str):
            return self._model.encode(text).tolist()

        # Batch encoding with memory protection
        all_embeddings: list[list[float]] = []
        effective_bs = self._adaptive_batch_size(text)

        for i in range(0, len(text), effective_bs):
            batch = text[i:i + effective_bs]
            try:
                embeddings = self._model.encode(batch)  # (batch_n, 1024) numpy
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "OOM on batch %d–%d, falling back to single-item encoding",
                        i, i + effective_bs,
                    )
                    for item in batch:
                        all_embeddings.append(self._model.encode(item).tolist())
                    continue
                raise
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def store_chunks(self, chunks: list[dict], *, doc_id: str = "doc_0") -> list[dict]:
        """Embed multiple chunks and return metadata for storage.

        Each input chunk must have at least ``text`` and ``section_path``.
        Returns the same chunks augmented with ``embedding``, ``source_doc_id``,
        ``chunk_index``, and ``word_count``.
        """
        texts = [c["text"] for c in chunks]
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
        """Rough token estimate — bge-m3 tokenizer is similar to GPT-like."""
        return len(text) // 2

    def _adaptive_batch_size(self, texts: list[str]) -> int:
        """Dynamically reduce batch size if total tokens exceed limit."""
        estimated_total_tokens = sum(self._estimate_tokens(t) for t in texts)
        if estimated_total_tokens > self.max_tokens_per_batch:
            scale = self.max_tokens_per_batch / max(estimated_total_tokens, 1)
            return max(4, int(self.batch_size * scale))
        return self.batch_size
