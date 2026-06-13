"""TXT/plain-text parser."""

import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class TXTParser:
    """Read plain text files with encoding fallback."""

    def parse(self, filepath: str) -> str:
        path = Path(filepath).resolve()
        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
        for enc in encodings:
            try:
                text = path.read_text(encoding=enc, errors="replace")
                logger.info("  Read %s (%s) — %.0f chars", path.name, enc, len(text))
                return text
            except (UnicodeDecodeError, LookupError):
                continue
        raise ValueError(f"Could not decode {path}: tried {encodings}")


# Register after class definition
from .dispatcher import register_parser, TextParser  # noqa: E402
register_parser("txt", TXTParser)