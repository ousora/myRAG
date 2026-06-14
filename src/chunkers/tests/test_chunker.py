"""Tests for LangChain-based Chunker."""

from chunkers import Chunker, chunk_text

SIMPLE_MD = """# Test Doc

## Section A
This is the content of section A. It has some text here.

## Section B
Section B has more content. This section is quite detailed and covers multiple aspects.
It continues with additional sentences that make the section longer.
"""

LONG_SECTION_MD = """# Doc

## Long Section
""" + ("This is a very long paragraph that keeps going on and on. " * 30) + """

## Short Section
Brief content here.
"""


class TestChunker:
    def test_empty_input_returns_empty(self):
        c = Chunker()
        assert c.chunk("") == []
        assert c.chunk("   ") == []

    def test_simple_split_on_headers(self):
        c = Chunker(chunk_size=1000)
        chunks = c.chunk(SIMPLE_MD)
        assert len(chunks) >= 2, f"Expected ≥2 chunks, got {len(chunks)}"

    def test_section_path_from_metadata(self):
        c = Chunker(chunk_size=1000)
        chunks = c.chunk(SIMPLE_MD)
        section_paths = [ch["section_path"] for ch in chunks]
        assert ["Section A"] in section_paths, f"Missing Section A in {section_paths}"
        assert ["Section B"] in section_paths, f"Missing Section B in {section_paths}"

    def test_hierarchical_metadata(self):
        nested_md = """# Top
## H2 Title
### H3 Sub
Nested content here.
"""
        c = Chunker(chunk_size=500)
        chunks = c.chunk(nested_md)
        meta = chunks[0]["metadata"]
        assert meta.get("H1") == "Top"
        assert meta.get("H2") == "H2 Title"
        assert meta.get("H3") == "H3 Sub"

    def test_oversized_section_split(self):
        c = Chunker(chunk_size=200, chunk_overlap=30)
        chunks = c.chunk(LONG_SECTION_MD)
        # The long section should produce multiple sub-chunks
        long_chunks = [ch for ch in chunks if "Long Section" in str(ch["section_path"])]
        assert len(long_chunks) >= 2, f"Expected ≥2 sub-chunks for long section, got {len(long_chunks)}"

    def test_oversized_chunks_keep_metadata(self):
        c = Chunker(chunk_size=200, chunk_overlap=30)
        chunks = c.chunk(LONG_SECTION_MD)
        for ch in chunks:
            assert "metadata" in ch, "Missing metadata in chunk"
            assert "section_path" in ch, "Missing section_path in chunk"
            assert "text" in ch, "Missing text in chunk"

    def test_convenience_wrapper(self):
        chunks = chunk_text(SIMPLE_MD, chunk_size=1000)
        assert len(chunks) >= 2

    def test_pipeline_facade_compatible(self):
        from pipeline import Chunker as PipeChunker
        c = PipeChunker(chunk_size=1000)
        chunks = c.chunk(SIMPLE_MD)
        assert len(chunks) >= 2
        assert "section_path" in chunks[0]
