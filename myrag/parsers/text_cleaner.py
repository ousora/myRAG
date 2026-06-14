"""TextCleaner — deterministic preprocessing for extracted text.

Runs before the LLM formatter: whitespace normalization, page-break removal,
and user-configurable regex rules from YAML (optional).
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class TextCleaner:
    """Apply deterministic cleaning steps to raw extracted text.

    Semantic understanding is delegated to format_text_async() in the formatter.
    This class only does regex-based noise removal and whitespace normalization.

    Built-in rules (always applied):
        1. Remove control characters
        2. Collapse page breaks (-- PAGE N --, ===)
        3. Normalize whitespace (tabs → spaces, blank lines → \\n\\n)
        4. Strip leading/trailing whitespace per line

    Optional YAML config rules loaded at construction time:
        - User-defined patterns from clean_rules.yaml are applied after built-ins
          and before the final newline normalization pass.
    """

    # ------------------------------------------------------------------ #
    # Default regex rules (always applied)                                #
    # ------------------------------------------------------------------ #
    _CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
    _PAGE_BREAK_RE = re.compile(
        r"(?:^|\n)\s*[-=*_]\s*(PAGE\s*\d+\s*)?\s*(-{2,})?",
    )

    # ------------------------------------------------------------------ #
    # Constructor                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        remove_page_breaks: bool = True,
        collapse_whitespace: bool = True,
        rules_config: str | None = "clean_rules.yaml",
    ) -> None:
        """Initialize TextCleaner with optional YAML rule file.

        Args:
            remove_page_breaks: Whether to strip --- PAGE N --- patterns.
            collapse_whitespace: Normalize whitespace and blank lines.
            rules_config: Path to YAML config (optional). If None, skip user rules.
        """
        self.remove_page_breaks = remove_page_breaks
        self.collapse_whitespace = collapse_whitespace

        # Load optional user-defined regex rules from YAML
        self._user_rules: list[tuple[str, int]] = []  # [(pattern_str, re_flags)]
        if rules_config is not None:
            path = Path(rules_config)
            if path.exists():
                try:
                    import yaml

                    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                    for rule in config.get("rules", []):
                        pattern_str = str(rule["pattern"])
                        flags = getattr(re, rule.get("flags", "0"), 0) or re.IGNORECASE
                        self._user_rules.append((pattern_str, flags))
                except Exception as exc:
                    logger.warning(
                        "Failed to load rules from %s (%s): %s",
                        path,
                        type(exc).__name__,
                        exc,
                    )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def clean(self, text: str) -> str:
        """Clean raw extracted text. Returns normalized string."""
        if not isinstance(text, str):
            return ""

        result = text.strip()
        if not result:
            return ""

        # 1. Remove control characters (PDFs often contain \x07 BELL etc.)
        result = self._CONTROL_RE.sub(" ", result)

        # 2. Page-break removal
        if self.remove_page_breaks:
            result = self._PAGE_BREAK_RE.sub("\n", result)

        # 3. Apply user-defined rules (if any loaded from YAML)
        for pattern_str, flags in self._user_rules:
            try:
                result = re.sub(pattern_str, "", result, flags=flags)
            except re.error as exc:
                logger.warning("Skipping invalid regex rule '%s': %s", pattern_str, exc)

        # 4. Whitespace normalization (must come last to avoid double-processing)
        if self.collapse_whitespace:
            result = self._collapse_whitespace(result)

        return result.strip()

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapse whitespace and blank lines."""
        # Tabs, vertical tab, form feed → space; multiple spaces → single space
        text = re.sub(r"[ \t\v\f]+", " ", text)
        # Multiple consecutive newlines → double newline (paragraph separator)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace per line (preserves structure)
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()
