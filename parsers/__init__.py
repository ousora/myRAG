"""Parser modules — auto-load registered parsers."""

import importlib

for _mod in ("pdf", "docx", "html", "md_parser", "txt"):
    try:
        importlib.import_module(f".{_mod}", __name__)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("Parser .%s skipped (%s): %s", _mod, type(e).__name__, e)