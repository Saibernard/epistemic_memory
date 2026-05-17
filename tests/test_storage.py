"""Tests for the SQLite storage layer."""

import json
import threading
from memory_layer.storage import MemoryStorage
from memory_layer.models import Memory, MemoryType, MemoryLink, LinkType


class TestStorageCRUD:
    def test_store_and_get(self, storage):
        m = Memory(memory_type=MemoryType.SEMANTIC, content="test content")
        storage.store_memory(m)
        fetched = storage.get_memory(m.id)
        assert fetched is not None
        assert fetched.content == "test content"
        assert fetched.memory_type == MemoryType.SEMANTIC

    def test_batch_get(self, storage):
        ids = []
        for i in range(5):
            m = Memory(memory_type=MemoryType.SEMANTIC, content=f"fact {i}")
            storage.store_memory(m)
            ids.append(m.id)
        loaded = storage.get_memories_by_ids(ids)
        assert len(loaded) == 5

    def test_deactivate(self, storage):
        m = Memory(memory_type=MemoryType.SEMANTIC, content="to deactivate")
        storage.store_memory(m)
        storage.deactivate_memory(m.id)
        fetched = storage.get_memory(m.id)
        assert fetched.is_active is False

    def test_hard_delete(self, storage):
        m = Memory(memory_type=MemoryType.SEMANTIC, content="to delete")
        storage.store_memory(m)
        storage.forget_memory(m.id, hard=True)
        assert storage.get_memory(m.id) is None


class TestNamespace:
    def test_namespace_stored(self, storage):
        m = Memory(memory_type=MemoryType.SEMANTIC, content="ns test", namespace="project-x")
        storage.store_memory(m)
        fetched = storage.get_memory(m.id)
        assert fetched.namespace == "project-x"

    def test_namespace_filter(self, storage):
        for ns in ["a", "b", "b"]:
            m = Memory(memory_type=MemoryType.SEMANTIC, content=f"in {ns}", namespace=ns)
            storage.store_memory(m)
        all_a = storage.get_all_memories(namespace="a")
        all_b = storage.get_all_memories(namespace="b")
        assert len(all_a) == 1
        assert len(all_b) == 2

    def test_default_namespace(self, storage):
        m = Memory(memory_type=MemoryType.SEMANTIC, content="default ns")
        storage.store_memory(m)
        fetched = storage.get_memory(m.id)
        assert fetched.namespace == "default"


class TestMigration:
    def test_schema_version_set(self, storage):
        version = storage.get_meta("schema_version")
        assert version is not None

    def test_namespace_column_exists(self, storage):
        m = Memory(memory_type=MemoryType.SEMANTIC, content="migration test", namespace="test")
        storage.store_memory(m)
        fetched = storage.get_memory(m.id)
        assert fetched.namespace == "test"


class TestLinks:
    def test_store_and_get_link(self, storage):
        m1 = Memory(memory_type=MemoryType.SEMANTIC, content="a")
        m2 = Memory(memory_type=MemoryType.SEMANTIC, content="b")
        storage.store_memory(m1)
        storage.store_memory(m2)

        link = MemoryLink(source_id=m1.id, target_id=m2.id, link_type=LinkType.SIMILAR, weight=0.8)
        storage.store_link(link)

        links = storage.get_links_for(m1.id)
        assert len(links) == 1
        assert links[0].weight == 0.8

    def test_batch_get_links(self, storage):
        m1 = Memory(memory_type=MemoryType.SEMANTIC, content="a")
        m2 = Memory(memory_type=MemoryType.SEMANTIC, content="b")
        storage.store_memory(m1)
        storage.store_memory(m2)
        link = MemoryLink(source_id=m1.id, target_id=m2.id, link_type=LinkType.SIMILAR)
        storage.store_link(link)

        links = storage.get_links_for_ids([m1.id, m2.id])
        assert len(links) >= 1


class TestBulkForget:
    def test_bulk_forget_by_namespace(self, storage):
        for i in range(3):
            m = Memory(memory_type=MemoryType.SEMANTIC, content=f"temp {i}", namespace="temp")
            storage.store_memory(m)
        m_keep = Memory(memory_type=MemoryType.SEMANTIC, content="keep", namespace="keep")
        storage.store_memory(m_keep)

        count = storage.bulk_forget(namespace="temp")
        assert count == 3

        remaining = storage.get_all_memories(namespace="keep")
        assert len(remaining) == 1

    def test_bulk_forget_requires_filter(self, storage):
        import pytest
        with pytest.raises(ValueError):
            storage.bulk_forget()


class TestConsolidationTracking:
    def test_consolidated_ids_tracked(self, storage):
        storage.log_consolidation("c1", ["e1", "e2", "e3"], "s1")
        ids = storage.get_consolidated_episode_ids()
        assert "e1" in ids
        assert "e2" in ids
        assert "e3" in ids


class TestConcurrency:
    def test_concurrent_writes(self, storage):
        """Multiple threads writing shouldn't cause database locked errors."""
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    m = Memory(memory_type=MemoryType.SEMANTIC, content=f"thread-{n}-{i}")
                    storage.store_memory(m)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent write errors: {errors}"
        counts = storage.get_counts()
        assert counts["total"] == 40


class TestMetadata:
    def test_get_set_meta(self, storage):
        storage.set_meta("test_key", "test_value")
        assert storage.get_meta("test_key") == "test_value"

    def test_counts(self, storage):
        m = Memory(memory_type=MemoryType.EPISODIC, content="ep")
        storage.store_memory(m)
        counts = storage.get_counts()
        assert counts["total"] == 1
        assert counts["episodic"] == 1
