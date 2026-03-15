"""
Configuration and path management for Memory Layer.

Provides a single canonical location for all memory data:
    ~/.memory-layer/
        memory.db           # SQLite database (source of truth)
        config.ini          # User configuration
        models/             # Cached embedding models

All components (CLI, MCP server, REST API) converge on the same paths
so the user has one brain shared across every integration.
"""

import configparser
import os
import sys
from pathlib import Path
from typing import Optional


_APP_DIR_NAME = ".memory-layer"

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

def get_home_dir() -> Path:
    """Return the memory-layer home directory (~/.memory-layer/)."""
    return Path.home() / _APP_DIR_NAME


def get_db_path() -> str:
    env = os.environ.get("MEMORY_DB_PATH")
    if env:
        return env
    return str(get_home_dir() / "memory.db")


def get_config_path() -> Path:
    return get_home_dir() / "config.ini"


def get_models_dir() -> str:
    return str(get_home_dir() / "models")


def ensure_home_dir() -> Path:
    """Create ~/.memory-layer/ and sub-dirs if they don't exist."""
    home = get_home_dir()
    home.mkdir(parents=True, exist_ok=True)
    (home / "models").mkdir(exist_ok=True)
    return home


# ─────────────────────────────────────────────
# CONFIG FILE
# ─────────────────────────────────────────────

_DEFAULT_CONFIG = {
    "general": {
        "default_namespace": "default",
    },
    "storage": {
        "backend": "sqlite",
        "postgres_url": "",
        "aws_region": "",
        "dynamodb_table": "memory-layer",
        "s3_bucket": "",
        "s3_prefix": "faiss-indices/",
    },
    "embeddings": {
        "mode": "local",
        "model": "all-mpnet-base-v2",
        "google_api_key": "",
        "openai_api_key": "",
    },
    "server": {
        "host": "127.0.0.1",
        "port": "8484",
    },
    "llm": {
        "extract": "false",
        "model": "gpt-4o-mini",
    },
}


def load_config() -> configparser.ConfigParser:
    """
    Load config from ~/.memory-layer/config.ini.

    Falls back to defaults for any missing keys. Environment variables
    override config file values.
    """
    config = configparser.ConfigParser()

    for section, values in _DEFAULT_CONFIG.items():
        config[section] = values

    config_path = get_config_path()
    if config_path.exists():
        config.read(str(config_path))

    _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: configparser.ConfigParser):
    """Environment variables take precedence over config file."""
    env_map = {
        ("storage", "backend"):   "MEMORY_STORAGE_BACKEND",
        ("storage", "postgres_url"): "MEMORY_POSTGRES_URL",
        ("storage", "aws_region"): "AWS_REGION",
        ("storage", "dynamodb_table"): "MEMORY_DYNAMODB_TABLE",
        ("storage", "s3_bucket"): "MEMORY_S3_BUCKET",
        ("storage", "s3_prefix"): "MEMORY_S3_PREFIX",
        ("embeddings", "mode"):  "MEMORY_EMBEDDING_MODE",
        ("embeddings", "model"): "MEMORY_EMBEDDING_MODEL",
        ("embeddings", "google_api_key"): "GOOGLE_API_KEY",
        ("embeddings", "openai_api_key"): "OPENAI_API_KEY",
        ("server", "host"):      "MEMORY_HOST",
        ("server", "port"):      "MEMORY_PORT",
        ("llm", "extract"):      "MEMORY_LLM_EXTRACT",
        ("general", "default_namespace"): "MEMORY_DEFAULT_NAMESPACE",
    }
    for (section, key), env_var in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            config[section][key] = val

    if os.environ.get("MEMORY_LLM_EXTRACT") == "1":
        config["llm"]["extract"] = "true"


def save_default_config():
    """Write the default config file to ~/.memory-layer/config.ini."""
    ensure_home_dir()
    config_path = get_config_path()
    if config_path.exists():
        return

    config = configparser.ConfigParser()
    for section, values in _DEFAULT_CONFIG.items():
        config[section] = values

    with open(config_path, "w") as f:
        f.write("# Memory Layer Configuration\n")
        f.write("# Edit these values to customize your memory system.\n")
        f.write("# Environment variables override these settings.\n\n")
        config.write(f)


# ─────────────────────────────────────────────
# INIT COMMAND
# ─────────────────────────────────────────────

def init_memory_layer(embedding_mode: str = "local", verbose: bool = True) -> Path:
    """
    Initialize the ~/.memory-layer/ directory and config.

    Called by `memory-layer init` or automatically on first use.
    Returns the home directory path.
    """
    home = ensure_home_dir()
    save_default_config()

    if embedding_mode != "local":
        config = load_config()
        config["embeddings"]["mode"] = embedding_mode
        with open(get_config_path(), "w") as f:
            config.write(f)

    if verbose:
        print(f"  Memory Layer initialized at: {home}")
        print(f"  Database:  {get_db_path()}")
        print(f"  Config:    {get_config_path()}")
        print(f"  Models:    {get_models_dir()}")

    return home


def get_status() -> dict:
    """Get current status information for the memory-layer status command."""
    home = get_home_dir()
    db_path = get_db_path()
    db_exists = os.path.exists(db_path)

    info = {
        "home_dir": str(home),
        "home_exists": home.exists(),
        "db_path": db_path,
        "db_exists": db_exists,
        "db_size_mb": round(os.path.getsize(db_path) / (1024 * 1024), 2) if db_exists else 0,
        "config_path": str(get_config_path()),
        "config_exists": get_config_path().exists(),
        "models_dir": get_models_dir(),
    }

    if db_exists:
        try:
            from .storage import MemoryStorage
            storage = MemoryStorage(db_path=db_path)
            counts = storage.get_counts()
            info["total_memories"] = counts.get("total", 0)
            info["total_links"] = counts.get("links", 0)
        except Exception:
            info["total_memories"] = "unknown"
            info["total_links"] = "unknown"

    return info
