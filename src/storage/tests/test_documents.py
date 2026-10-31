"""Tests for document-level CRUD operations."""

class TestDocumentOps:
    """Test document-level CRUD operations."""

    def test_upsert_document(self, store):
        doc = store.upsert_document(
            title="Test Doc",
            tags=["tag1", "tag2"],
            text_summary="A summary of the document.",
            source_file="/path/to/file.pdf",
            total_chunks=5,
        )
        assert doc["title"] == "Test Doc"
        assert doc["total_chunks"] == 5

    def test_search_documents_by_tags(self, store):
        store.upsert_document(
            title="Doc A", tags=["finance"],
            text_summary="Financial document.",
            source_file="/a.pdf", total_chunks=1,
        )
        store.upsert_document(
            title="Doc B", tags=["tech"],
            text_summary="Technology document.",
            source_file="/b.pdf", total_chunks=1,
        )

        results = store.search_documents(tags=["finance"])
        assert len(results) == 1
        assert results[0]["title"] == "Doc A"

    def test_search_documents_no_filter(self, store):
        """Search without filters returns all documents."""
        store.upsert_document(
            title="Doc X", tags=[],
            text_summary="Summary X.", source_file="/x.pdf", total_chunks=1,
        )
        results = store.search_documents()
        assert len(results) >= 1
