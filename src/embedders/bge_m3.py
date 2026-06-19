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

import logging

from config import get_config

logger = logging.getLogger(__name__)


class Embedder:
    """Unified embedder — delegates to remote (HTTP) or local (sentence-transformers).

    Mode is selected by config's ``embedding.mode`` field:
      - ``"remote"`` (default): calls HTTP API at ``embedding.base_url``
      - ``"local"``: uses sentence-transformers with ``embedding.local_model``

    Calling ``Embedder()`` with no arguments reads all settings from config.
    Explicit arguments override config for remote mode only.
    """

    def __new__(cls, **kwargs):
        cfg = get_config()
        mode = getattr(cfg, "embedding_mode", "remote")

        if mode == "local":
            from .local_bge import LocalEmbedder

            instance = object.__new__(LocalEmbedder)
            # Read local_model from config and pass to LocalEmbedder
            local_model = getattr(cfg, "embedding_local_model", None) or "BAAI/bge-m3"
            instance.__init__(model_name=local_model)
            return instance

        return super().__new__(cls)

    # ── Remote embedder (default) ────────────────────────────────────────
    # __init__ and all methods below are only used when mode == "remote".
    # In local mode, ``__new__`` returns a LocalEmbedder instance instead,
    # so Python never calls these methods.

    def __init__(self, *, base_url: str = "", model: str = ""):
        cfg = get_config()
        base_url = base_url or cfg.embedding_base_url
        model = model or cfg.embedding_model

        import httpx

        self.client = httpx.Client(base_url=base_url, timeout=cfg.embedding_timeout)
        self.model = model

    def embed(self, text: str | list[str]) -> list[list[float]]:
        """Get embeddings for one or multiple texts."""
        if isinstance(text, str):
            payload = {"model": self.model, "input": [text]}
        else:
            payload = {"model": self.model, "input": text}

        resp = self.client.post("/v1/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(text, str):
            return data["data"][0]["embedding"]
        else:
            return [d["embedding"] for d in data["data"]]

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


def embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
    """Convenience wrapper."""
    e = Embedder(**kwargs)
    return e.embed(texts)
