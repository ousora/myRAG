"""TextCleaner — deterministic preprocessing for extracted text.

Runs before the LLM formatter: whitespace normalization, page-break removal,
and user-configurable regex rules from YAML (optional).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

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

    # 1. Control characters — delete entirely to prevent word merging
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    # 2. Page breaks / Horizontal rules
    # Generalized: Starts with 3+ separators, allows words/digits in middle,
    # ends with 2+ separators.
    # Examples: "--- PAGE 1 ---", "*** 12 ***", "=== Section ==="
    _PAGE_BREAK_RE = re.compile(
        r"^(?:[-=_*]\s*){3,}[\w\s]*?(?:[-=_*]\s*){2,}$",
        flags=re.IGNORECASE,
    )

    # 3. Whitespace collapsing (pre-compiled for performance)
    _TAB_TO_SPACE_RE = re.compile(r"[\t\v\f]+")
    # Only trim TRAILING spaces — preserve leading spaces for Markdown code blocks/lists
    _TRIM_TRAILING_SPACE_RE = re.compile(r"(?m)[ ]+$")
    _MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")

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
        self.custom_rules: list[dict[str, Any]] = []

        if rules_config is not None:
            config_path = Path(rules_config)
            if config_path.is_file():
                self._load_custom_rules(config_path)

    def _load_custom_rules(self, path: Path) -> None:
        """Safely loads and pre-compiles custom regex rules from a YAML file."""
        try:
            import yaml  # noqa: F811 — optional dependency
        except ImportError:
            logger.warning(
                "PyYAML is not installed. Skipping custom rules. "
                "Install via: pip install pyyaml"
            )
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            for rule in data.get("rules", []):
                pattern_str = rule.get("pattern")
                if not pattern_str:
                    continue

                try:
                    compiled_pattern = re.compile(
                        pattern_str, self._parse_flags(rule.get("flags"))
                    )
                    self.custom_rules.append(
                        {"pattern": compiled_pattern, "replace": rule.get("replace", "")}
                    )
                except re.error as e:
                    logger.warning("Failed to compile custom rule [%s]: %s", pattern_str, e)

        except Exception as e:
            logger.warning("Failed to load YAML rules file %s: %s", path, e)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def clean(self, text: str) -> str:
        """Clean raw extracted text. Returns normalized string."""
        if not isinstance(text, str) or not text:
            return ""

        result = self._remove_control_chars(text)

        if self.remove_page_breaks:
            result = self._remove_page_breaks(result)

        # 3. Repair broken table rows
        result = self._fix_broken_tables(result)

        # 4. Apply pre-compiled custom regex rules
        for rule in self.custom_rules:
            result = rule["pattern"].sub(rule["replace"], result)

        # 5. Collapse excessive whitespace (must come last to avoid double-processing)
        if self.collapse_whitespace:
            result = self._collapse_whitespace(result)

        return result.strip()

    # ------------------------------------------------------------------ #
    # Private methods                                                     #
    # ------------------------------------------------------------------ #

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters from the input."""
        return self._CONTROL_CHAR_RE.sub("", text)

    def _remove_page_breaks(self, text: str) -> str:
        """Filters out lines that match the generalized page break pattern."""
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            # Safeguard: only apply regex if the line is long enough to be a page break.
            # This prevents deleting normal short lines like "---" (Markdown HR) or "- item".
            if len(stripped) > 8 and self._PAGE_BREAK_RE.match(stripped):
                continue
            lines.append(line)
        return "\n".join(lines)

    def _fix_broken_tables(self, text: str) -> str:
        """Repair table rows broken during PDF extraction.

        If the current row has fewer columns than the previous, it appends the content
        into the last cell of the previous row to preserve Markdown column structure.
        """
        lines = text.splitlines()
        if not lines:
            return text

        merged = []
        prev_line = ""
        prev_col_count = 0

        for line in lines:
            stripped = line.strip()

            # Generalized table row detection (must start and end with |)
            is_table_row = stripped.startswith("|") and stripped.endswith("|")
            col_count = stripped.count("|") - 1 if is_table_row else 0

            if is_table_row and prev_line and 0 < col_count < prev_col_count:
                last_pipe_idx = prev_line.rfind("|", 0, -1)
                if last_pipe_idx != -1:
                    append_text = stripped.strip("|").strip()
                    prev_line = f"{prev_line[:last_pipe_idx]} {append_text} |"
                    merged[-1] = prev_line
                    continue

            merged.append(line)
            prev_line = line
            prev_col_count = col_count

        return "\n".join(merged)

    @staticmethod
    def _parse_flags(raw_flags: Any) -> int:
        """Parse regex flags from YAML config (supports int, str, or list of strings)."""
        if not raw_flags:
            return 0
        if isinstance(raw_flags, int):
            return raw_flags

        flags = 0
        flag_list = raw_flags if isinstance(raw_flags, list) else [raw_flags]
        for f in flag_list:
            flags |= getattr(re, str(f).upper(), 0)
        return flags

    @classmethod
    def _collapse_whitespace(cls, text: str) -> str:
        """Collapse whitespace using pre-compiled regexes.

        Preserves leading spaces (indentation) for Markdown/Code blocks.
        """
        text = cls._TAB_TO_SPACE_RE.sub(" ", text)
        text = cls._TRIM_TRAILING_SPACE_RE.sub("", text)  # Only trim trailing!
        text = cls._MULTIPLE_NEWLINES_RE.sub("\n\n", text)
        return text
