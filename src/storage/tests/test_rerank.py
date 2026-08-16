"""Tests for rerank.py — MMR re-ranking of retrieved chunks."""

from rerank import _cosine, _lexical_score, _tokenize, mmr_rerank


class TestTokenize:
    """Tests for _tokenize — splits text into query tokens."""

    def test_latin_words(self):
        tokens = _tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_cjk_characters(self):
        tokens = _tokenize("你好世界")
        assert "你" in tokens
        assert "好" in tokens
        assert "世" in tokens
        assert "界" in tokens

    def test_mixed_content(self):
        tokens = _tokenize("hello 你好 world")
        assert "hello" in tokens
        assert "world" in tokens
        assert "你" in tokens

    def test_case_insensitive(self):
        tokens = _tokenize("Hello HELLO hello")
        assert tokens.count("hello") == 3

    def test_single_char_latin_not_tokenized(self):
        """Single letter words are excluded (requires 2+ chars)."""
        tokens = _tokenize("a b c")
        assert all(len(t) >= 2 for t in tokens)


class TestLexicalScore:
    """Tests for _lexical_score — normalized lexical overlap."""

    def test_exact_match(self):
        score = _lexical_score("hello world", "hello world")
        assert score == 1.0

    def test_partial_match(self):
        score = _lexical_score("hello world", "hello there")
        assert 0 < score < 1.0

    def test_no_match(self):
        score = _lexical_score("hello world", "foo bar baz")
        assert score == 0.0

    def test_empty_query(self):
        score = _lexical_score("", "hello world")
        assert score == 0.0

    def test_empty_text(self):
        score = _lexical_score("hello world", "")
        assert score == 0.0

    def test_cjk_match(self):
        score = _lexical_score("你好世界", "你好世界")
        assert score == 1.0

    def test_cjk_partial_match(self):
        score = _lexical_score("你好世界", "你好地球")
        assert 0 < score < 1.0


class TestCosine:
    """Tests for _cosine — cosine similarity."""

    def test_identical_vectors(self):
        v = [0.3, 0.4, 0.5]
        assert abs(_cosine(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine(a, b)) < 0.001

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine(a, b) - (-1.0)) < 0.001

    def test_empty_vectors(self):
        assert _cosine([], [0.1]) == 0.0
        assert _cosine([0.1], []) == 0.0

    def test_zero_vector(self):
        assert _cosine([0.0, 0.0], [0.3, 0.4]) == 0.0


class TestMmrRerank:
    """Tests for mmr_rerank — Maximal Marginal Relevance re-ranking."""

    def _make_embedding(self, seed=42, dim=1024):
        import random
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(dim)]

    def _make_chunk(self, text, embedding=None):
        if embedding is None:
            embedding = self._make_embedding()
        return {
            "text": text,
            "section_path": ["Section"],
            "source_doc_id": "doc_0",
            "chunk_index": 0,
            "word_count": len(text.split()),
            "embedding": embedding,
        }

    def test_empty_candidates(self):
        result = mmr_rerank("query", self._make_embedding(), [], [])
        assert result == []

    def test_fewer_than_k_returns_sorted_by_relevance(self):
        """When candidates <= k, return all sorted by lexical relevance."""
        candidates = [
            self._make_chunk("hello world"),
            self._make_chunk("foo bar"),
        ]
        query_vec = self._make_embedding(seed=3)
        chunk_vectors = [c["embedding"] for c in candidates]

        result = mmr_rerank("hello world", query_vec, chunk_vectors, candidates, k=5)

        assert len(result) == 2
        # "hello world" should rank first (exact match)
        assert "hello world" in result[0]["text"]

    def test_mmr_diversity_selects_dissimilar_chunks(self):
        """MMR should prefer diverse chunks over near-duplicates."""
        query_vec = self._make_embedding(seed=100)

        # Two very similar chunks (nearly identical embeddings)
        base_emb = self._make_embedding(seed=200)
        dup_emb1 = base_emb[:]
        dup_emb2 = [x + 0.001 for x in base_emb]

        similar = [
            self._make_chunk("about machine learning AI", embedding=dup_emb1),
            self._make_chunk("about machine learning AI", embedding=dup_emb2),
        ]
        # One very different chunk (orthogonal-ish embedding)
        diff_emb = self._make_embedding(seed=300)

        different_chunk = self._make_chunk("about quantum physics", embedding=diff_emb)
        candidates = [similar[0], similar[1], different_chunk]
        chunk_vectors = [c["embedding"] for c in candidates]

        # Use lambda=0.0 to make it pure diversity — should pick the most diverse chunk
        result = mmr_rerank(
            "machine learning", query_vec, chunk_vectors, candidates,
            k=2, lambda_param=0.0,
        )

        # Should return 2 chunks. With pure diversity, picks first by relevance (similar[0])
        # then the most diverse remaining chunk (different_chunk, since it has lowest cosine
        # similarity to similar[0] compared to similar[1])
        assert len(result) == 2
        texts = [c["text"] for c in result]
        combined = "\n".join(texts)
        # The different chunk should be selected as the second item due to diversity
        assert "quantum" in combined or "physics" in combined

    def test_lambda_one_pure_relevance(self):
        """lambda=1.0 should be pure relevance (no diversity penalty)."""
        candidates = [
            self._make_chunk("hello world"),
            self._make_chunk("foo bar"),
        ]
        query_vec = self._make_embedding(seed=3)
        chunk_vectors = [c["embedding"] for c in candidates]

        result = mmr_rerank(
            "hello world", query_vec, chunk_vectors, candidates,
            k=2, lambda_param=1.0,
        )

        assert len(result) == 2
        # Pure relevance: "hello world" should be first
        assert "hello world" in result[0]["text"]

    def test_lambda_zero_pure_diversity(self):
        """lambda=0.0 should be pure diversity (no relevance)."""
        candidates = [
            self._make_chunk("hello world"),
            self._make_chunk("hello world"),
        ]
        query_vec = self._make_embedding(seed=3)
        chunk_vectors = [c["embedding"] for c in candidates]

        result = mmr_rerank(
            "hello world", query_vec, chunk_vectors, candidates,
            k=1, lambda_param=0.0,
        )

        assert len(result) == 1

    def test_k_limited_output(self):
        """Output should be at most k items."""
        candidates = [self._make_chunk(f"text {i}") for i in range(10)]
        query_vec = self._make_embedding(seed=42)
        chunk_vectors = [c["embedding"] for c in candidates]

        result = mmr_rerank("test", query_vec, chunk_vectors, candidates, k=3)

        assert len(result) == 3

    def test_returns_original_chunk_dicts(self):
        """Returned chunks should be the original dicts (not copies)."""
        candidates = [self._make_chunk("text")]
        query_vec = self._make_embedding(seed=2)
        chunk_vectors = [c["embedding"] for c in candidates]

        result = mmr_rerank("test", query_vec, chunk_vectors, candidates, k=1)

        assert result[0] is candidates[0]

    def test_preserves_chunk_metadata(self):
        """Each returned chunk should retain all original fields."""
        chunk = {
            "text": "test content",
            "section_path": ["Intro", "Section"],
            "source_doc_id": "my_doc",
            "chunk_index": 42,
            "word_count": 10,
            "embedding": self._make_embedding(seed=99),
        }
        candidates = [chunk]
        query_vec = self._make_embedding(seed=1)
        chunk_vectors = [c["embedding"] for c in candidates]

        result = mmr_rerank("test", query_vec, chunk_vectors, candidates, k=1)

        assert result[0]["source_doc_id"] == "my_doc"
        assert result[0]["chunk_index"] == 42
        assert result[0]["section_path"] == ["Intro", "Section"]

    def test_single_candidate(self):
        """Single candidate should be returned as-is."""
        candidates = [self._make_chunk("only one")]
        query_vec = self._make_embedding(seed=2)
        chunk_vectors = [c["embedding"] for c in candidates]

        result = mmr_rerank("test", query_vec, chunk_vectors, candidates, k=5)

        assert len(result) == 1
        assert result[0]["text"] == "only one"

    def test_large_candidate_set(self):
        """Should handle large candidate sets without error."""
        candidates = [self._make_chunk(f"document about topic {i}") for i in range(50)]
        query_vec = self._make_embedding(seed=200)
        chunk_vectors = [c["embedding"] for c in candidates]

        result = mmr_rerank("topic", query_vec, chunk_vectors, candidates, k=5)

        assert len(result) == 5

    def test_missing_embedding_fallback(self):
        """Chunks with missing embeddings should still work (cosine returns 0)."""
        candidates = [
            {"text": "hello world", "section_path": ["S"], "source_doc_id": "d",
             "chunk_index": 0, "word_count": 2, "embedding": None},
            {"text": "foo bar", "section_path": ["S"], "source_doc_id": "d",
             "chunk_index": 1, "word_count": 2, "embedding": None},
        ]
        query_vec = self._make_embedding(seed=42)
        chunk_vectors = [c.get("embedding") for c in candidates]

        result = mmr_rerank("hello world", query_vec, chunk_vectors, candidates, k=2)

        assert len(result) == 2
        # "hello world" should rank first by lexical relevance
        assert "hello world" in result[0]["text"]
