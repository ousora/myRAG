"""Configuration loader — reads YAML config with fallback chain.

Resolution order (first wins):
    1. $MYRAG_CONFIG environment variable
    2. <project_root>/conf/config.yaml           (user instance, gitignored)
    3. <project_root>/conf/config.example.yaml   (safe defaults, committed)

Usage:
    from config import get_config, Config

    cfg = get_config()
    cfg.llm_endpoint          # "http://192.168.191.112:8081/v1/chat/completions"
    cfg.embedding_base_url    # "http://192.168.191.112:11435"
"""

import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resolve project root — parent of the myrag package directory (i.e., the repo root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Config directory where YAML files live
_CONF_DIR = _PROJECT_ROOT / "conf"

# Absolute path to the clean rules YAML so callers don't need to worry about CWD.
CLEAN_RULES_PATH: Path = _CONF_DIR / "clean_rules.yaml"


def _resolve_config_path() -> Path | None:
    """Find the first existing config file in the resolution chain."""
    # 1. Env var override
    env_path = os.environ.get("MYRAG_CONFIG")
    if env_path:
        env_file = Path(env_path).expanduser()
        if env_file.exists():
            return env_file

    # 2. Local instance config (gitignored — contains real IPs)
    local = _CONF_DIR / "config.yaml"
    if local.exists():
        return local

    # 3. Example template (safe defaults, committed to git)
    example = _CONF_DIR / "config.example.yaml"
    if example.exists():
        return example

    return None


class Config:
    """Typed access to all myRAG configuration values.

    All fields have sensible defaults so the package works out-of-the-box
    with localhost endpoints when no config file is present.
    """

    def __init__(self, raw: dict[str, Any]):
        # ── LLM ──
        llm = raw.get("llm", {})
        self.llm_endpoint: str         = llm.get("endpoint", "http://localhost:8081/v1/chat/completions")
        self.llm_model: str            = llm.get("model", "local-model")
        self.llm_temperature: float    = llm.get("temperature", 0.3)
        self.llm_max_tokens: int       = llm.get("max_tokens", 16384)
        self.llm_timeout: int          = llm.get("timeout", 300)

        # ── Embedding ──
        emb = raw.get("embedding", {})
        self.embedding_base_url: str   = emb.get("base_url", "http://localhost:11435")
        self.embedding_model: str      = emb.get("model", "bge-m3")
        self.embedding_timeout: int    = emb.get("timeout", 60)
        self.embedding_mode: str       = emb.get("mode", "remote")
        self.embedding_local_model: str | None = emb.get("local_model")
        # When the remote/local endpoint is unavailable, fall back to a
        # deterministic hash-based pseudo-embedding. Useful for offline
        # development or when no embedding service is reachable.
        self.embedding_hash_fallback: bool = emb.get("hash_fallback", False)
        # Instruction prepended to *queries* (not documents) for retrieval-trained
        # models like bge-m3. Empty string disables the prefix.
        self.embedding_query_instruction: str = emb.get(
            "query_instruction", "Represent this sentence for searching relevant passages: "
        )

        # ── Formatter ──
        fmt = raw.get("formatter", {})
        self.chunk_threshold_chars: int = fmt.get("chunk_threshold_chars", 20000)
        self.chunk_max_tokens: int      = fmt.get("chunk_max_tokens", 16384)
        self.chunk_timeout: int         = fmt.get("chunk_timeout", 300)

        # ── Pipeline ──
        pipe = raw.get("pipeline", {})
        self.format_timeout: int = pipe.get("format_timeout", 3600)

        # ── Logging ──
        log_cfg = raw.get("logging", {})
        self.log_max_bytes: int = log_cfg.get("max_bytes", 5 * 1024 * 1024)
        self.debug_log_llm_responses: bool = log_cfg.get("debug_log_llm_responses", False)

    def _validate(self) -> list[str]:
        """Return a list of validation error messages, or [] if valid."""
        errors: list[str] = []

        # LLM settings
        if not self.llm_endpoint:
            errors.append("llm.endpoint is required")
        if not self.llm_model:
            errors.append("llm.model is required")

        # Embedding mode validation
        if self.embedding_mode not in ("remote", "local"):
            errors.append(f"embedding.mode must be 'remote' or 'local' (got {self.embedding_mode!r})")
        elif self.embedding_mode == "remote":
            if not self.embedding_base_url:
                errors.append("embedding.base_url required when mode=remote")
        elif self.embedding_mode == "local":
            if not self.embedding_local_model:
                errors.append("embedding.local_model required when mode=local")

        # Formatter settings must be positive
        for field in ("chunk_threshold_chars", "chunk_max_tokens", "chunk_timeout"):
            val = getattr(self, field)
            if not isinstance(val, int) or val <= 0:
                errors.append(f"{field} must be a positive integer (got {val})")

        # Pipeline settings must be positive
        for field in ("format_timeout",):
            val = getattr(self, field)
            if not isinstance(val, int) or val <= 0:
                errors.append(f"{field} must be a positive integer (got {val})")

        return errors

    def __repr__(self) -> str:
        return (
            f"Config(llm={self.llm_endpoint} [{self.llm_model}], "
            f"embed={self.embedding_base_url} [{self.embedding_model}])"
        )


_config_cache: Config | None = None


def get_config(reset: bool = False) -> Config:
    """Load and cache configuration. Safe to call repeatedly from any module.

    Args:
        reset: If True, clear the cached instance and reload from disk.

               Use this after modifying *conf/config.yaml* without restarting
               the process. Defaults to False (return cached instance).

    """
    global _config_cache
    if reset:
        _config_cache = None
    if _config_cache is not None:
        return _config_cache

    path = _resolve_config_path()
    if path is None:
        cfg = Config({})
    else:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = Config(raw)

    errors = cfg._validate()
    if errors:
        raise ValueError(
            "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        )

    _config_cache = cfg
    return cfg


def get_config_lazy() -> Config:
    """Lazy-loaded config — returns the cached instance on first call.

    Thin wrapper around ``get_config()`` for callers that need to defer
    configuration loading until explicitly invoked (e.g., CLI entry points).
    Prefer calling ``get_config()`` from new code when possible.
    """
    return get_config()
