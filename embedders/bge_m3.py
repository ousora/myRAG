"""Embedding client — call your local bge-m3 service (vLLM / Ollama compatible).

Usage:
    from myrag.embedders import Embedder
    
    e = Embedder(base_url="http://192.168.191.112:8081")
    emb = e.embed("你好世界")  # → list[float]
"""


class Embedder:
    """Wrap a local bge-m3 API server."""

    def __init__(self, *, base_url="http://127.0.0.1:8000", model="bge-m3"):
        import httpx
        self.client = httpx.Client(base_url=base_url, timeout=60)
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
        
        # OpenAI-compatible API returns embeddings in a list of objects
        if isinstance(text, str):
            return data["data"][0]["embedding"]
        else:
            return [d["embedding"] for d in data["data"]]


def embed_texts(texts: list[str], **kwargs) -> list[list[float]]:
    """Convenience wrapper."""
    e = Embedder(**kwargs)
    return e.embed(texts)
