"""Parser modules — register built-in parsers at load time."""

import parsers.dispatcher  # noqa: F401 — registers all built-in parsers (MarkItDown + Trafilatura)
# Old individual parsers removed; dispatcher handles registration.