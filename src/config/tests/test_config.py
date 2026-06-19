"""Tests for configuration loading and validation."""

import os
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from config import Config, get_config


# ---------------------------------------------------------------------------
# Config defaults — each new field must have a sensible default
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """Verify that all Config fields have reasonable fallback values."""

    def test_formatter_defaults(self):
        cfg = Config({})
        assert cfg.chunk_threshold_chars == 28000
        assert cfg.chunk_max_tokens == 16384
        assert cfg.chunk_timeout == 300

    def test_pipeline_defaults(self):
        cfg = Config({})
        assert cfg.format_timeout == 3600

    def test_logging_defaults(self):
        cfg = Config({})
        assert cfg.log_max_bytes == 5 * 1024 * 1024


class TestConfigOverrides:
    """Verify that YAML values override defaults."""

    def test_formatter_override(self):
        raw = {
            "formatter": {
                "chunk_threshold_chars": 32000,
                "chunk_max_tokens": 8192,
                "chunk_timeout": 600,
            }
        }
        cfg = Config(raw)
        assert cfg.chunk_threshold_chars == 32000
        assert cfg.chunk_max_tokens == 8192
        assert cfg.chunk_timeout == 600

    def test_pipeline_override(self):
        raw = {"pipeline": {"format_timeout": 7200}}
        cfg = Config(raw)
        assert cfg.format_timeout == 7200

    def test_logging_override(self):
        raw = {"logging": {"max_bytes": 10 * 1024 * 1024}}
        cfg = Config(raw)
        assert cfg.log_max_bytes == 10 * 1024 * 1024


class TestConfigValidation:
    """Verify that _validate() catches misconfigurations."""

    def test_validate_passes_with_defaults(self):
        cfg = Config({})
        errors = cfg._validate()
        assert errors == []

    def test_validate_rejects_zero_chunk_timeout(self):
        cfg = Config({"formatter": {"chunk_timeout": 0}})
        errors = cfg._validate()
        assert any("chunk_timeout" in e for e in errors)

    def test_validate_rejects_negative_format_timeout(self):
        cfg = Config({"pipeline": {"format_timeout": -1}})
        errors = cfg._validate()
        assert any("format_timeout" in e for e in errors)


class TestGetConfig:
    """Verify that get_config() loads from the correct file."""

    def test_get_config_returns_instance(self):
        get_config.cache_clear()  # clear lru_cache so mock takes effect
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text") as mock_read:
            mock_read.return_value = (
                "llm:\n  endpoint: http://example.com\n"
                "formatter:\n  chunk_threshold_chars: 30000\n"
            )
            cfg = get_config()
        assert isinstance(cfg, Config)
        assert cfg.chunk_threshold_chars == 30000

    def test_get_config_caches(self):
        get_config.cache_clear()  # clear lru_cache so mock takes effect
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text") as mock_read:
            mock_read.return_value = (
                "llm:\n  endpoint: http://example.com\n"
                "formatter:\n  chunk_threshold_chars: 30000\n"
            )
            cfg1 = get_config()
            cfg2 = get_config()
        assert cfg1 is cfg2
