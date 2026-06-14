"""Embedding client — call your local bge-m3 service (vLLM / Ollama compatible).

Usage:
    from embedders import Embedder
    
    e = Embedder(base_url="http://192.168.191.112:8081")
    
    # Single text embedding
    emb = e.embed("你好世界")  # → list[float]
    
    # Chunk-level storage (recommended for RAG)
    chunks = [{"text": "chunk content", "section_path": ["Section"]}]
    docs = e.store_chunks(chunks, doc_id="my_doc_123")

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


class Embedder:
    """Wrap a local bge-m3 API server.

    If no base_url is provided, reads from config (config/config.yaml).
    """

    def __init__(self, *, base_url: str = "", model: str = ""):
        import httpx

        # Load config defaults, then override with explicit params
        from config import get_config
        cfg = get_config()
        base_url = base_url or cfg.embedding_base_url
        model = model or cfg.embedding_model

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

    def store_chunk(self, chunk_text: str, *, section_path=None, doc_id="doc_0", chunk_idx=0) -> dict:
        """Embed a single chunk and return metadata for storage.
        
        Args:
            chunk_text: Text content of the chunk (includes section header prefix)
            section_path: Semantic path from LLM formatter
            doc_id: Source document identifier
            chunk_idx: Position within source document

        Returns:
            dict with text, vector, and metadata ready for FAISS/Milvus insertion.
        """
        embedding = self.embed(chunk_text)
        
        return {
            "text": chunk_text.strip(),
            "section_path": section_path or ["General"],
            "source_doc_id": doc_id,
            "chunk_index": chunk_idx,
            "word_count": len(chunk_text.split()),
            "embedding": embedding,  # bge-m3 → 1024-d vector
        }

    def store_chunks(self, chunks: list[dict], *, doc_id="doc_0") -> list[dict]:
        """Embed multiple chunks and return metadata for storage."""
        texts = [c["text"] for c in chunks]
        embeddings = self.embed(texts) if texts else []
        
        results = []
        for i, chunk in enumerate(chunks):
            result = dict(chunk)  # copy input keys (section_path already set by chunker)
            result["source_doc_id"] = doc_id
            result["chunk_index"] = i
            result["word_count"] = len(chunk.get("text", "").split())
            if embeddings:
                result["embedding"] = embeddings[i]
            results.append(result)
        
        return results

    def store_document(self, title: str, tags: list[str], text_summary: str, 
                       source_file: str, total_chunks: int) -> dict:
        """Embed a document-level summary and return metadata for storage."""
        embedding = self.embed(text_summary)
        
        return {
            "title": title,
            "tags": tags,
            "text_summary": text_summary[:1000],  # Keep it short for doc index
            "source_file": source_file,
            "total_chunks": total_chunks,
            "embedding": embedding,  # bge-m3 → 1024-d vector
        }


def embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
    """Convenience wrapper."""
    e = Embedder(**kwargs)
    return e.embed(texts)
