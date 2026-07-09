"""Embedding client — call your local bge-m3 service (vLLM / Ollama compatible).

Usage:
    from embedders import Embedder
    
    e = Embedder()
    
    # Single text embedding
    emb = e.embed("你好世界")  # → list[list[float]]
    
    # Chunk-level storage (recommended for RAG)
    chunks = [{"text": "chunk content", "section_path": ["Section"]}]
    docs = e.store_chunks(chunks, doc_id="my_doc_123")

Mode switching:
    Config's ``embedding.mode`` selects the backend at construction time:
      - ``"remote"`` (default): HTTP API at ``embedding.base_url``
      - ``"local"``: sentence-transformers with ``embedding.local_model``

Schema:
    chunk_store (fine-grained):
        - text: str
        - section_path: list[str]
        - source_doc_id: str
        - vector: float[]  # bge-m3 → 1024-d
    
    doc_store (coarse-grained):
        - title: str
        - tags: list[str]
        - text_summary: str
        - source_file: str
        - total_chunks: int
        - vector: float[]  # bge-m3 → 1024-d
"""

import httpx
import logging

from config import get_config

from myrag.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Expected embedding dimension for bge-m3 models. Other models may differ,
# but this serves as a sanity check to catch misconfigured endpoints early.
EXPECTED_EMBEDDING_DIMENSION = 1024


def _validate_embedding_dimension(embedding: list[float]) -> None:
    """Assert that the returned embedding has the expected dimensionality."""
    if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
        raise EmbeddingError(
            message=(
                f"Embedding dimension mismatch: expected {EXPECTED_EMBEDDING_DIMENSION}, "
                f"got {len(embedding)}. Check that your endpoint serves the correct model."
            ),
            context={"expected_dim": EXPECTED_EMBEDDING_DIMENSION, "actual_dim": len(embedding)},
        )


class Embedder:
    """Unified embedder — delegates to remote (HTTP) or local (sentence-transformers).

    Mode is selected by config's ``embedding.mode`` field:
      - ``"remote"`` (default): calls HTTP API at ``embedding.base_url``
      - ``"local"``: uses sentence-transformers with ``embedding.local_model``

    Calling ``Embedder()`` with no arguments reads all settings from config.
    Explicit arguments override config for remote mode only.
    """

    def __new__(cls, **kwargs):  # type: ignore[override]
        cfg = get_config()
        mode = getattr(cfg, "embedding_mode", "remote")

        if mode == "local":
            from .local_bge import LocalEmbedder

            instance = object.__new__(LocalEmbedder)
            local_model = getattr(cfg, "embedding_local_model", None) or "BAAI/bge-m3"
            instance.__init__(model_name=local_model)
            return instance  # type: ignore[return-value]

        return super().__new__(cls)

    # ── Remote embedder (default) ────────────────────────────────────────
    # __init__ and all methods below are only used when mode == "remote".
    # In local mode, ``__new__`` returns a LocalEmbedder instance instead,
    # so Python never calls these methods.

    def __init__(self, *, base_url: str = "", model: str = ""):
        cfg = get_config()
        base_url = base_url or cfg.embedding_base_url
        model = model or cfg.embedding_model

        self.client = httpx.Client(base_url=base_url, timeout=cfg.embedding_timeout)
        self.model = model

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            self.client = None

    def _post_with_retry(self, url: str, payload: dict, *, max_retries: int = 3) -> httpx.Response:
        """POST to the embedding API with exponential backoff on transient errors.

        Retries on HTTP 429 (rate limit), 502/503/504 (server errors). Exponential
        backoff starts at 1s, capped at 8s. Non-transient errors raise immediately.
        """
        import time as _time

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.post(url, json=payload)
                if not resp.is_server_error and resp.status_code != 429:
                    return resp
                # Transient error — retry with backoff.
                wait = min(2 ** attempt, 8)
                logger.warning(
                    "Embedding API returned %d (attempt %d/%d), retrying in %ds",
                    resp.status_code, attempt + 1, max_retries + 1, wait,
                )
                last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                _time.sleep(wait)
            except httpx.TimeoutException as exc:
                logger.warning(
                    "Embedding API timed out (attempt %d/%d), retrying in %ds",
                    attempt + 1, max_retries + 1, min(2 ** attempt, 8),
                )
                last_exc = exc
                _time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(f"Embedding API failed after {max_retries} retries: {last_exc}") from last_exc

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Get embeddings for one or multiple texts.

        Returns a single ``list[float]`` when *text* is a ``str``, and a
        ``list[list[float]]`` when *text* is a ``list[str]``. Callers that pass
        a string (e.g. a query or document summary) must use the vector
        directly rather than indexing ``[0]``.
        """
        if isinstance(text, str):
            payload = {"model": self.model, "input": [text]}
        else:
            payload = {"model": self.model, "input": text}

        resp = self._post_with_retry("/v1/embeddings", payload)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(text, str):
            emb = data["data"][0]["embedding"]
            _validate_embedding_dimension(emb)
            return emb
        else:
            embeddings = [d["embedding"] for d in data["data"]]
            for e in embeddings:
                _validate_embedding_dimension(e)
            return embeddings

    def store_chunk(
        self,
        chunk_text: str,
        *,
        section_path=None,
        doc_id="doc_0",
        chunk_idx=0,
    ) -> dict:
        """Embed a single chunk and return metadata for storage."""
        embedding = self.embed(chunk_text)

        return {
            "text": chunk_text.strip(),
            "section_path": section_path or ["General"],
            "source_doc_id": doc_id,
            "chunk_index": chunk_idx,
            "word_count": len(chunk_text.split()),
            "embedding": embedding,
        }

    def store_chunks(self, chunks: list[dict], *, doc_id: str = "doc_0") -> list[dict]:
        """Embed multiple chunks and return metadata for storage."""
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


def create_embedder(
    mode: str | None = None, *, base_url: str = "", model: str = "", validate: bool = False
) -> "Embedder":
    """Create an embedder instance for the given mode.

    Args:
        mode: ``"remote"`` or ``"local"``. Defaults to config's ``embedding.mode``.
        base_url: Override for remote mode endpoint (ignored in local mode).
        model: Override for the embedding model name.
        validate: If True, performs a test embedding and validates dimension on creation.
                  Raises :class:`EmbeddingError` if dimension mismatch is detected.

    Returns:
        An Embedder instance whose concrete class depends on *mode*.
    """
    if mode is None:
        cfg = get_config()
        mode = getattr(cfg, "embedding_mode", "remote")

    if mode == "local":
        from .local_bge import LocalEmbedder
        local_model = model or (getattr(get_config(), "embedding_local_model", None) or "BAAI/bge-m3")
        return LocalEmbedder(model_name=local_model)  # type: ignore[return-value]

    e = Embedder(base_url=base_url, model=model)
    if validate:
        _validate_embedding_dimension(e.embed("validation"))
    return e


def embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
    """Convenience wrapper."""
    mode = kwargs.pop("mode", None)
    e = create_embedder(mode=mode, **kwargs)
    return e.embed(texts)
