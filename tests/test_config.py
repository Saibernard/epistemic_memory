"""Tests for config and path management."""

import os
from pathlib import Path
from memory_layer.config import (
    get_home_dir,
    get_db_path,
    load_config,
    save_default_config,
    ensure_home_dir,
    init_memory_layer,
    get_status,
)


class TestPaths:
    def test_home_dir_is_under_home(self):
        home = get_home_dir()
        assert str(home).startswith(str(Path.home()))
        assert ".memory-layer" in str(home)

    def test_db_path_env_override(self, monkeypatch):
        monkeypatch.setenv("MEMORY_DB_PATH", "/tmp/custom.db")
        assert get_db_path() == "/tmp/custom.db"

    def test_db_path_default(self, monkeypatch):
        monkeypatch.delenv("MEMORY_DB_PATH", raising=False)
        path = get_db_path()
        assert path.endswith("memory.db")


class TestConfig:
    def test_load_defaults(self):
        config = load_config()
        assert config.get("embeddings", "mode") == "local"
        assert config.get("server", "port") == "8484"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MEMORY_EMBEDDING_MODE", "openai")
        config = load_config()
        assert config.get("embeddings", "mode") == "openai"

    def test_init_creates_dir(self, tmp_path, monkeypatch):
        fake_home = tmp_path / ".memory-layer"
        monkeypatch.setattr("memory_layer.config.get_home_dir", lambda: fake_home)
        monkeypatch.setattr("memory_layer.config.get_config_path", lambda: fake_home / "config.ini")
        monkeypatch.setattr("memory_layer.config.get_models_dir", lambda: str(fake_home / "models"))

        init_memory_layer(verbose=False)
        assert fake_home.exists()
        assert (fake_home / "models").exists()
        assert (fake_home / "config.ini").exists()


class TestStatus:
    def test_status_returns_dict(self):
        info = get_status()
        assert "home_dir" in info
        assert "db_path" in info
        assert "db_exists" in info
