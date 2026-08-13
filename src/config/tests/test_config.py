"""Tests for configuration loading and validation."""

import os

from config import Config, get_config


# ---------------------------------------------------------------------------
# Config defaults — each new field must have a sensible default
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """Verify that all Config fields have reasonable fallback values."""

    def test_formatter_defaults(self):
        cfg = Config({})
        assert cfg.chunk_threshold_chars == 20000
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

    def _temp_config(self, content: str):
        """Create a temporary config file and set MYRAG_CONFIG env var."""
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        old_env = os.environ.get("MYRAG_CONFIG")
        os.environ["MYRAG_CONFIG"] = path
        get_config.cache_clear()
        return path, old_env

    def _restore_env(self, path: str, old_env: str | None):
        os.unlink(path)
        if old_env is None:
            os.environ.pop("MYRAG_CONFIG", None)
        else:
            os.environ["MYRAG_CONFIG"] = old_env
        get_config.cache_clear()

    def test_get_config_returns_instance(self):
        cfg_content = (
            "llm:\n  endpoint: http://example.com\n"
            "formatter:\n  chunk_threshold_chars: 30000\n"
        )
        path, old_env = self._temp_config(cfg_content)
        try:
            cfg = get_config()
        finally:
            self._restore_env(path, old_env)
        assert isinstance(cfg, Config)
        assert cfg.chunk_threshold_chars == 30000

    def test_get_config_caches(self):
        cfg_content = (
            "llm:\n  endpoint: http://example.com\n"
            "formatter:\n  chunk_threshold_chars: 30000\n"
        )
        path, old_env = self._temp_config(cfg_content)
        try:
            cfg1 = get_config()
            cfg2 = get_config()
            assert cfg1 is cfg2
        finally:
            self._restore_env(path, old_env)
