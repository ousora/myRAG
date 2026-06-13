"""BeautifulSoup + readability-based HTML parser."""

import logging
from pathlib import Path


try:
    from bs4 import BeautifulSoup  # noqa: F401
except ImportError:
    pass  # skipped by __init__ loop if dependency not installed

logger = logging.getLogger(__name__)


class HTMLParser:
    """Extract article text from HTML pages.

    Uses readability-lxml to strip boilerplate (ads, navbars), then BeautifulSoup for cleanup.
    """

    def parse(self, filepath: str) -> str:
        path = Path(filepath).resolve()
        html_text = path.read_text(encoding="utf-8", errors="replace")

        soup = BeautifulSoup(html_text, "lxml")
        title = soup.find("title").text.strip() if soup.title else ""
        body = self._extract_body(soup)

        text_parts: list[str] = []
        if title:
            text_parts.append(title.upper())
        if body:
            text_parts.append(body)

        full_text = "\n\n".join(text_parts).strip()
        logger.info("  Extracted title='%s', %.0f chars from %s", title[:40], len(full_text), path.name)
        return full_text

    def _extract_body(self, soup: BeautifulSoup) -> str:
        """Try readability-lxml first; fall back to BeautifulSoup heuristics."""
        try:
            from readability import Document as ReadabilityDoc
            reader = ReadabilityDoc(html_text=soup.encode("utf-8"), url=str(soup.find("title")))
            cleaned_html = reader.summary()
            clean_soup = BeautifulSoup(cleaned_html, "lxml")
            return self._strip_tags(clean_soup.get_text(separator="\n", strip=True))
        except Exception as e:
            logger.debug("readability-lxml failed (%s), falling back to BS4 heuristics.", type(e).__name__)
            # Fallback: get article/main content div
            for selector in ("#article", "#content", ".post-body", "main"):
                el = soup.select_one(selector)
                if el is not None:
                    return self._strip_tags(el.get_text(separator="\n", strip=True))
            # Last resort: all body text
            return self._strip_tags(soup.body.get_text(separator="\n", strip=True)) if soup.body else ""

    @staticmethod
    def _strip_tags(text: str) -> str:
        """Collapse whitespace and remove non-printable chars."""
        import re
        lines = [l.strip() for l in text.splitlines()]
        lines = [l for l in lines if l]
        return "\n".join(lines)


# Register after class definition
from .dispatcher import register_parser, TextParser  # noqa: E402
register_parser("html", HTMLParser)