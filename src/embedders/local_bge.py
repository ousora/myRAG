"""Local bge-m3 embedding via sentence-transformers (no server needed).

Usage:
    from embedders.local_bge import LocalEmbedder

    e = LocalEmbedder(model_name="BAAI/bge-m3")
    emb = e.embed("你好世界")        # → list[list[float]], 1024-d
"""

from __future__ import annotations

import logging

from config import get_config
from .bge_m3 import _embed_cache_get, _embed_cache_put, _hash_embed, _validate_embedding_dimension

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

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        # sentence-transformers holds no open network/file handles to close.
        return False

    # ── Attributes that tests and callers expect on both backends ────────
    # ``model`` mirrors Embedder.model so Embedder.__new__ can return a
    # LocalEmbedder in local mode without callers needing to branch.
    model: str = "BAAI/bge-m3"

    def store_chunk(
        self,
        chunk_text: str,
        *,
        section_path=None,
        doc_id="doc_0",
        chunk_idx=0,
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
                emb = self._model.encode(text).tolist()
            except Exception as exc:
                if getattr(get_config(), "embedding_hash_fallback", False):
                    logger.warning("Local embed failed (%s); using hash fallback", exc)
                    emb = _hash_embed(text)
                else:
                    raise
            _validate_embedding_dimension(emb)
            _embed_cache_put(text, emb)
            return emb

        # Batch encoding with memory protection
        all_embeddings: list[list[float]] = []
        effective_bs = self._adaptive_batch_size(text)

        try:
            for i in range(0, len(text), effective_bs):
                batch = text[i:i + effective_bs]
                try:
                    embeddings = self._model.encode(batch)  # (batch_n, EXPECTED_EMBEDDING_DIMENSION) numpy
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(
                            "OOM on batch %d–%d (%d items), progressively reducing to single-item encoding",
                            i, i + effective_bs, len(batch),
                        )
                        # Retry with smaller batches; fall back to single-item on persistent OOM.
                        sub_batch_size = max(1, len(batch) // 2)
                        if sub_batch_size == 0:
                            sub_batch_size = 1
                        for j in range(i, i + len(batch), sub_batch_size):
                            sub_batch = text[j:j + sub_batch_size]
                            try:
                                sub_emb = self._model.encode(sub_batch).tolist()
                                all_embeddings.extend(sub_emb)
                            except RuntimeError as e2:
                                if "out of memory" in str(e2).lower():
                                    # Absolute fallback: encode one item at a time.
                                    logger.warning("OOM persists; encoding remaining items individually")
                                    for single_item in text[j:j + sub_batch_size]:
                                        emb = self._model.encode(single_item).tolist()
                                        _validate_embedding_dimension(emb)
                                        all_embeddings.append(emb)
                                else:
                                    raise
                        continue
                    raise
                for e in embeddings.tolist():
                    _validate_embedding_dimension(e)
                all_embeddings.extend(embeddings.tolist())
        except Exception as exc:
            if getattr(get_config(), "embedding_hash_fallback", False):
                logger.warning("Local batch embed failed (%s); using hash fallback", exc)
                return [_hash_embed(t) for t in text]
            raise

        return all_embeddings

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
        """Token estimate for bge-m3's SentencePiece tokenizer.

        Uses a multi-language-aware heuristic: CJK characters count as ~1 token,
        ASCII words average ~4 chars/token (with whitespace). This is more accurate
        than the naive ``len // 2`` which underestimates Chinese text by ~50%.
        """
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        ascii_chars = sum(1 for ch in text if ("\u0041" <= ch <= "\u005a") or ("a" <= ch <= "z"))
        other_ascii = len(text) - cjk - ascii_chars

        # CJK: ~1 char/token (SentencePiece byte-level).
        # ASCII letters/digits: ~4 chars/token.
        # Punctuation/whitespace: counted as 0.5 token each.
        return cjk + (ascii_chars // 4) + (other_ascii // 2)

    def _adaptive_batch_size(self, texts: list[str]) -> int:
        """Dynamically reduce batch size if total tokens exceed limit."""
        estimated_total_tokens = sum(self._estimate_tokens(t) for t in texts)
        if estimated_total_tokens > self.max_tokens_per_batch:
            scale = self.max_tokens_per_batch / max(estimated_total_tokens, 1)
            return max(4, int(self.batch_size * scale))
        return self.batch_size
