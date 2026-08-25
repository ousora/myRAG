"""Self-check: hash embedding produces stable, semantically useful 1024-d vectors.

Run directly:  uv run python src/embedders/tests/test_hash_fallback.py
"""

from embedders.bge_m3 import EXPECTED_EMBEDDING_DIMENSION, _hash_embed


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def main() -> None:
    a = _hash_embed("hello world")
    b = _hash_embed("hello world")
    c = _hash_embed("goodbye world")

    assert len(a) == EXPECTED_EMBEDDING_DIMENSION, f"dim {len(a)}"
    assert a == b, "same text must produce identical vectors"
    assert a != c, "different text must produce different vectors"
    assert all(-1.0 <= v <= 1.0 for v in a), "L2-normalized values in [-1, 1]"

    # Lexical semantics: overlapping tokens → higher similarity than disjoint.
    zh_a = _hash_embed("支付系统的风险管控措施")
    zh_b = _hash_embed("支付系统的管理规定")
    zh_c = _hash_embed("天气预报说明天有雨")
    assert _cosine(zh_a, zh_b) > _cosine(zh_a, zh_c), "overlap must outrank disjoint"

    print(f"OK: {EXPECTED_EMBEDDING_DIMENSION}-d, deterministic, distinct, lexical")


if __name__ == "__main__":
    main()
