"""Self-check: hash fallback produces stable 1024-d vectors.

Run directly:  uv run python src/embedders/tests/test_hash_fallback.py
"""

from embedders.bge_m3 import EXPECTED_EMBEDDING_DIMENSION, _hash_embed


def main() -> None:
    a = _hash_embed("hello world")
    b = _hash_embed("hello world")
    c = _hash_embed("goodbye world")

    assert len(a) == EXPECTED_EMBEDDING_DIMENSION, f"dim {len(a)}"
    assert a == b, "same text must produce identical vectors"
    assert a != c, "different text must produce different vectors"
    assert all(-1.0 <= v < 1.0 for v in a), "values must be in [-1, 1)"
    print(f"OK: {EXPECTED_EMBEDDING_DIMENSION}-d, deterministic, distinct")


if __name__ == "__main__":
    main()
