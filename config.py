"""Configuration loader — reads YAML config with fallback chain.

Resolution order (first wins):
    1. $MYRAG_CONFIG environment variable
    2. <project_root>/conf/config.yaml           (user instance, gitignored)
    3. <project_root>/conf/config.example.yaml   (safe defaults, committed)

Usage:
    from myrag.config import get_config, Config

    cfg = get_config()
    cfg.llm_endpoint          # "http://192.168.191.112:8081/v1/chat/completions"
    cfg.embedding_base_url    # "http://192.168.191.112:11435"
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Resolve project root — parent of the myrag package directory (i.e., the repo root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent

# Config directory where YAML files live
_CONF_DIR = _PROJECT_ROOT / "conf"


def _resolve_config_path() -> Optional[Path]:
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

    def __init__(self, raw: Dict[str, Any]):
        # ── LLM ──
        llm = raw.get("llm", {})
        self.llm_endpoint: str         = llm.get("endpoint", "http://localhost:8081/v1/chat/completions")
        self.llm_model: str            = llm.get("model", "local-model")
        self.llm_temperature: float    = llm.get("temperature", 0.3)
        self.llm_max_tokens: int       = llm.get("max_tokens", 8192)
        self.llm_timeout: int          = llm.get("timeout", 180)

        # ── Embedding ──
        emb = raw.get("embedding", {})
        self.embedding_base_url: str   = emb.get("base_url", "http://localhost:11435")
        self.embedding_model: str      = emb.get("model", "bge-m3")
        self.embedding_timeout: int    = emb.get("timeout", 60)

    def __repr__(self) -> str:
        return (
            f"Config(llm={self.llm_endpoint} [{self.llm_model}], "
            f"embed={self.embedding_base_url} [{self.embedding_model}])"
        )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and cache configuration. Safe to call repeatedly from any module."""
    path = _resolve_config_path()
    if path is None:
        return Config({})

    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config(raw)
