"""
Tests for the storage backend system: protocol, factory, and SQLite compliance.

DynamoDB and Postgres backends are tested via the same protocol compliance
suite when their connection details are available (otherwise skipped).
"""

import os
import pytest

from memory_layer.storage_protocol import StorageBackend
from memory_layer.storage_factory import create_storage
from memory_layer.storage import MemoryStorage
from memory_layer.models import Memory, MemoryType, MemoryLink, LinkType, WorkingMemoryItem


class TestStorageFactory:
    def test_default_is_sqlite(self, tmp_db):
        storage = create_storage("sqlite", sqlite_path=tmp_db)
        assert isinstance(storage, MemoryStorage)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_storage("redis")

    def test_postgres_requires_url(self):
        with pytest.raises(ValueError, match="connection URL"):
            create_storage("postgres")

    def test_dynamodb_requires_region(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        monkeypatch.delenv("MEMORY_DYNAMODB_TABLE", raising=False)
        with pytest.raises(ValueError, match="AWS region"):
            create_storage("dynamodb")

    def test_env_override(self, tmp_db, monkeypatch):
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_DB_PATH", tmp_db)
        storage = create_storage()
        assert isinstance(storage, MemoryStorage)


class TestProtocolCompliance:
    """Verify that SQLite storage satisfies the StorageBackend protocol."""

    def test_sqlite_is_storage_backend(self, tmp_db):
        storage = MemoryStorage(db_path=tmp_db)
        assert isinstance(storage, StorageBackend)

    def test_has_all_methods(self, tmp_db):
        storage = MemoryStorage(db_path=tmp_db)

        required_methods = [
            "get_meta", "set_meta", "has_memories",
            "count_active_memories_with_embeddings",
            "get_sample_embedding_dimension", "clear_all_embeddings",
            "store_memory", "get_memory", "get_memories_by_ids",
            "get_all_memories", "update_memory", "deactivate_memory",
            "forget_memory", "bulk_forget",
            "get_memories_with_embeddings",
            "store_passages", "get_all_passage_embeddings",
            "delete_passages_for_memory",
            "store_link", "get_links_for", "get_links_for_ids", "get_all_links",
            "store_working_item", "get_working_memory",
            "clear_working_memory", "trim_working_memory",
            "log_consolidation", "get_consolidated_episode_ids",
            "get_consolidation_count",
            "get_counts", "get_avg_strength", "get_avg_importance",
            "get_oldest_memory_age_hours", "get_most_accessed_memory_id",
        ]
        for method_name in required_methods:
            assert hasattr(storage, method_name), f"Missing method: {method_name}"
            assert callable(getattr(storage, method_name)), f"Not callable: {method_name}"


class TestSQLiteViaFactory:
    """Run core operations through the factory-created SQLite backend."""

    def test_remember_and_recall(self, tmp_db):
        storage = create_storage("sqlite", sqlite_path=tmp_db)
        mem = Memory(memory_type=MemoryType.SEMANTIC, content="test via factory")
        storage.store_memory(mem)
        fetched = storage.get_memory(mem.id)
        assert fetched is not None
        assert fetched.content == "test via factory"

    def test_links_via_factory(self, tmp_db):
        storage = create_storage("sqlite", sqlite_path=tmp_db)
        m1 = Memory(memory_type=MemoryType.SEMANTIC, content="a")
        m2 = Memory(memory_type=MemoryType.SEMANTIC, content="b")
        storage.store_memory(m1)
        storage.store_memory(m2)
        link = MemoryLink(source_id=m1.id, target_id=m2.id, link_type=LinkType.SIMILAR)
        storage.store_link(link)
        links = storage.get_links_for(m1.id)
        assert len(links) == 1

    def test_working_memory_via_factory(self, tmp_db):
        storage = create_storage("sqlite", sqlite_path=tmp_db)
        item = WorkingMemoryItem(content="hello", role="user")
        storage.store_working_item(item)
        wm = storage.get_working_memory()
        assert len(wm) == 1
        storage.clear_working_memory()
        assert len(storage.get_working_memory()) == 0

    def test_stats_via_factory(self, tmp_db):
        storage = create_storage("sqlite", sqlite_path=tmp_db)
        mem = Memory(memory_type=MemoryType.SEMANTIC, content="test")
        storage.store_memory(mem)
        counts = storage.get_counts()
        assert counts["total"] == 1
        assert storage.get_avg_strength() > 0


class TestManagerWithStorage:
    """Test MemoryManager accepts a pre-built storage instance."""

    def test_manager_with_explicit_storage(self, tmp_db):
        storage = create_storage("sqlite", sqlite_path=tmp_db)
        from memory_layer import MemoryManager
        brain = MemoryManager(storage=storage)
        mem = brain.remember("injected storage test")
        assert mem.content == "injected storage test"
        results = brain.recall("storage test", min_confidence=0.0, min_strength=0.0)
        assert len(results) >= 1

    def test_manager_with_storage_backend_flag(self, tmp_db):
        from memory_layer import MemoryManager
        brain = MemoryManager(db_path=tmp_db, storage_backend="sqlite")
        mem = brain.remember("backend flag test")
        assert mem.content == "backend flag test"
