"""Regression tests for process_file_with_md cleaning-rule resolution.

``process_file_with_md`` must resolve custom cleaning rules from the absolute
``CLEAN_RULES_PATH`` (so ``conf/clean_rules.yaml`` is applied regardless of the
current working directory) — matching ``_parse_and_format`` used by the hybrid
pipeline. Previously it fell back to TextCleaner's default relative path, which
silently dropped the custom rules when run from a different CWD.
"""

from unittest.mock import Mock, patch

from config import CLEAN_RULES_PATH


class TestProcessFileWithMDCleanRules:
    """process_file_with_md must apply conf/clean_rules.yaml from any CWD."""

    def test_uses_absolute_clean_rules_path(self):
        """TextCleaner must be constructed with str(CLEAN_RULES_PATH)."""
        from pipeline.hybrid import process_file_with_md

        with patch("pipeline.hybrid.utils.resolve_parser") as resolve_mock, patch(
            "pipeline.hybrid.TextCleaner"
        ) as text_cleaner_cls, patch("formatters.write_to_md") as write_mock:
            resolve_mock.return_value = Mock(parse=Mock(return_value="raw text"))
            write_mock.return_value = "output/test.md"

            process_file_with_md("doc.txt", output_dir="output/", use_llm=False)

        # TextCleaner was constructed with the absolute CLEAN_RULES_PATH, not the
        # default relative "clean_rules.yaml".
        _, kwargs = text_cleaner_cls.call_args
        assert kwargs["rules_config"] == str(CLEAN_RULES_PATH)

    def test_caller_can_override_rules_config(self):
        """A caller-supplied rules_config must still take precedence."""
        from pipeline.hybrid import process_file_with_md

        override = "/custom/clean_rules.yaml"
        with patch("pipeline.hybrid.utils.resolve_parser") as resolve_mock, patch(
            "pipeline.hybrid.TextCleaner"
        ) as text_cleaner_cls, patch("formatters.write_to_md") as write_mock:
            resolve_mock.return_value = Mock(parse=Mock(return_value="raw text"))
            write_mock.return_value = "output/test.md"

            process_file_with_md(
                "doc.txt", output_dir="output/", use_llm=False, rules_config=override
            )

        _, kwargs = text_cleaner_cls.call_args
        assert kwargs["rules_config"] == override
