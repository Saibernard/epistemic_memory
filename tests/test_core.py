"""Tests for core MemoryManager functionality."""

import time
from memory_layer import MemoryManager, MemoryType


class TestRemember:
    def test_basic_remember(self, brain):
        m = brain.remember("User likes Python", importance=0.8)
        assert m.content == "User likes Python"
        assert m.importance >= 0.8
        assert m.is_active is True
        assert m.namespace == "default"

    def test_remember_with_namespace(self, brain):
        m = brain.remember("Project uses React", namespace="frontend")
        assert m.namespace == "frontend"

    def test_remember_with_tags(self, brain):
        m = brain.remember("Use dark mode", tags=["preference", "ui"])
        assert "preference" in m.tags
        assert "ui" in m.tags

    def test_remember_importance_detection(self, brain):
        m = brain.remember("This is critical: always use HTTPS", importance=0.5)
        assert m.importance > 0.5

    def test_dedup_returns_existing(self, brain):
        m1 = brain.remember("User prefers vim")
        m2 = brain.remember("User prefers vim")
        assert m1.id == m2.id

    def test_remember_different_content_creates_new(self, brain):
        m1 = brain.remember("User likes cats")
        m2 = brain.remember("User likes dogs")
        assert m1.id != m2.id


class TestRecall:
    def test_basic_recall(self, brain):
        brain.remember("Python is the preferred language")
        results = brain.recall("programming language", min_confidence=0.0, min_strength=0.0)
        assert len(results) >= 1

    def test_recall_empty(self, brain):
        results = brain.recall("something random", min_confidence=0.0, min_strength=0.0)
        assert results == []

    def test_recall_namespace_isolation(self, brain):
        brain.remember("Secret project detail", namespace="project-a")
        brain.remember("Other project detail", namespace="project-b")
        results_a = brain.recall("project", namespace="project-a", min_confidence=0.0, min_strength=0.0)
        results_b = brain.recall("project", namespace="project-b", min_confidence=0.0, min_strength=0.0)
        ids_a = {r.memory.id for r in results_a}
        ids_b = {r.memory.id for r in results_b}
        assert ids_a.isdisjoint(ids_b)

    def test_recall_tag_filtering(self, brain):
        brain.remember("Frontend uses React", tags=["frontend"])
        brain.remember("Backend uses Python", tags=["backend"])
        results = brain.recall("technology", tags=["frontend"], min_confidence=0.0, min_strength=0.0)
        for r in results:
            assert "frontend" in r.memory.tags


class TestForget:
    def test_soft_forget(self, brain):
        m = brain.remember("Temporary note")
        assert brain.forget_memory(m.id, hard=False) is True
        fetched = brain.storage.get_memory(m.id)
        assert fetched is not None
        assert fetched.is_active is False

    def test_hard_forget(self, brain):
        m = brain.remember("Delete me permanently")
        assert brain.forget_memory(m.id, hard=True) is True
        fetched = brain.storage.get_memory(m.id)
        assert fetched is None

    def test_forget_nonexistent(self, brain):
        assert brain.forget_memory("nonexistent-id") is False


class TestCorrect:
    def test_correct_memory(self, brain):
        m_old = brain.remember("Database is MongoDB")
        m_new = brain.correct_memory(m_old.id, "Database is PostgreSQL", reason="migrated")
        assert m_new is not None
        assert m_new.content == "Database is PostgreSQL"
        assert "corrected" in m_new.tags
        old = brain.storage.get_memory(m_old.id)
        assert old.is_active is True
        assert old.metadata.get("superseded_by") == m_new.id
        assert old.metadata.get("valid_until") is not None
        assert old.strength <= 0.5


class TestEpisode:
    def test_record_episode(self, brain):
        m = brain.record_episode(
            user_message="How do I sort a list?",
            assistant_response="Use sorted() or list.sort()",
            feedback="positive",
        )
        assert m.memory_type == MemoryType.EPISODIC
        assert m.importance > 0.5

    def test_episode_with_namespace(self, brain):
        m = brain.record_episode(
            user_message="test",
            assistant_response="test",
            namespace="project-x",
        )
        assert m.namespace == "project-x"


class TestStats:
    def test_stats_basic(self, brain):
        brain.remember("The user prefers Python for backend development")
        brain.remember("Meeting scheduled for Tuesday at 3pm with the design team")
        stats = brain.get_stats()
        assert stats.total_memories == 2

    def test_stats_by_namespace(self, brain):
        brain.remember("a", namespace="ns1")
        brain.remember("b", namespace="ns2")
        stats = brain.get_stats(namespace="ns1")
        assert stats.total_memories == 1


class TestWorkingMemory:
    def test_working_memory(self, brain):
        brain.add_to_working_memory("hello", role="user")
        brain.add_to_working_memory("hi there", role="assistant")
        context = brain.get_working_context()
        assert len(context) == 2
        brain.clear_working_memory()
        assert len(brain.get_working_context()) == 0


class TestDecay:
    def test_run_decay(self, brain):
        brain.remember("will decay")
        result = brain.run_decay()
        assert "processed" in result

    def test_consolidate(self, brain):
        result = brain.consolidate()
        # Multi-level consolidation: L0→L1, L1→L2, L2→L3
        assert "level_0_to_1" in result
        assert "episodes_analyzed" in result["level_0_to_1"]
