"""Tests for the parser dispatcher module."""

import logging


class TestRegisterParser:
    """Test that parsers register correctly under their extensions and aliases."""

    def test_register_parser(self):
        from parsers.dispatcher import PARSERS, register_parser

        class DummyParser:
            def parse(self, filepath: str) -> str:
                return ""

        register_parser("xyz", DummyParser)
        assert "xyz" in PARSERS
        assert PARSERS["xyz"] is DummyParser

    def test_register_alias(self):
        from parsers.dispatcher import PARSERS, register_parser

        class MarkdownLike:
            def parse(self, filepath: str) -> str:
                return ""

        # Clear and re-register to avoid side effects from module load
        old = dict(PARSERS)
        try:
            register_parser("markdown", MarkdownLike)
            assert "md" in PARSERS
            assert "mkd" in PARSERS
            assert PARSERS["md"] is MarkdownLike
            assert PARSERS["mkd"] is MarkdownLike
        finally:
            PARSERS.clear()
            PARSERS.update(old)

    def test_register_lowercase(self):
        from parsers.dispatcher import PARSERS, register_parser

        class UpperParser:
            def parse(self, filepath: str) -> str:
                return ""

        old = dict(PARSERS)
        try:
            register_parser("MARKDOWN", UpperParser)
            assert "markdown" in PARSERS
            assert PARSERS["markdown"] is UpperParser
        finally:
            PARSERS.clear()
            PARSERS.update(old)


class TestResolveParser:
    """Test resolve_parser routing logic."""

    def test_resolve_pdf(self):
        from parsers.dispatcher import resolve_parser, MarkItDownParser

        parser = resolve_parser("report.pdf")
        assert isinstance(parser, MarkItDownParser)

    def test_resolve_docx(self):
        from parsers.dispatcher import resolve_parser, MarkItDownParser

        parser = resolve_parser("document.docx")
        assert isinstance(parser, MarkItDownParser)

    def test_resolve_markdown(self):
        from parsers.dispatcher import resolve_parser, MarkItDownParser

        parser = resolve_parser("readme.md")
        assert isinstance(parser, MarkItDownParser)

    def test_resolve_txt(self):
        from parsers.dispatcher import resolve_parser, MarkItDownParser

        parser = resolve_parser("notes.txt")
        assert isinstance(parser, MarkItDownParser)

    def test_resolve_html(self):
        from parsers.dispatcher import resolve_parser, TrafilaturaParser

        parser = resolve_parser("page.html")
        assert isinstance(parser, TrafilaturaParser)

    def test_resolve_htm_alias(self):
        from parsers.dispatcher import resolve_parser, TrafilaturaParser

        parser = resolve_parser("page.htm")
        assert isinstance(parser, TrafilaturaParser)

    def test_unsupported_extension_returns_none(self):
        """Unsupported extensions should return None without side effects."""
        from parsers.dispatcher import PARSERS, resolve_parser

        # Temporarily remove all registered parsers to isolate this test
        original = dict(PARSERS)
        try:
            PARSERS.clear()
            result = resolve_parser("file.xyz")
            assert result is None
        finally:
            PARSERS.clear()
            PARSERS.update(original)

    def test_fallback_to_markitdown_for_pptx(self):
        from parsers.dispatcher import resolve_parser, MarkItDownParser

        parser = resolve_parser("presentation.pptx")
        assert isinstance(parser, MarkItDownParser)

    def test_fallback_to_markitdown_for_xlsx(self):
        from parsers.dispatcher import resolve_parser, MarkItDownParser

        parser = resolve_parser("spreadsheet.xlsx")
        assert isinstance(parser, MarkItDownParser)


class TestTextParserProtocol:
    """Test that TextParser protocol is correctly defined."""

    def test_protocol_runtime_checkable(self):
        from parsers.dispatcher import TextParser

        class ValidParser:
            def parse(self, filepath: str) -> str:
                return ""

        assert isinstance(ValidParser(), TextParser)
