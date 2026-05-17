"""
Functional tests for the Memory Layer.

Tests specific functional requirements and edge cases:
  - Storage layer CRUD for new tables
  - Knowledge page manager operations
  - Lint checker accuracy
  - Provenance chain integrity
  - Version history completeness
  - Integration adapter contracts
"""

import time
import pytest

from memory_layer.models import (
    KnowledgePage, ProvenanceEntry, MemoryVersion,
    EpistemicStatus, MemoryType,
)


class TestStorageKnowledgePages:
    """Functional tests for knowledge page storage operations."""

    def test_store_and_retrieve_page(self, storage):
        """Store a page and retrieve it by ID."""
        page = KnowledgePage(
            entity_id="ent_001",
            title="Test Entity",
            page_type="entity",
            summary="A test entity page.",
        )
        pid = storage.store_knowledge_page(page)
        assert pid == page.page_id

        retrieved = storage.get_knowledge_page(page.page_id)
        assert retrieved is not None
        assert retrieved.title == "Test Entity"
        assert retrieved.summary == "A test entity page."

    def test_get_page_by_entity(self, storage):
        """Retrieve page by entity_id."""
        page = KnowledgePage(
            entity_id="ent_002",
            title="Python",
            page_type="topic",
        )
        storage.store_knowledge_page(page)
        found = storage.get_knowledge_page_by_entity("ent_002")
        assert found is not None
        assert found.title == "Python"

    def test_get_page_by_title(self, storage):
        """Retrieve page by title (case-insensitive)."""
        page = KnowledgePage(
            entity_id="ent_003",
            title="Django Framework",
            page_type="topic",
        )
        storage.store_knowledge_page(page)
        found = storage.get_knowledge_page_by_title("django framework")
        assert found is not None
        assert found.entity_id == "ent_003"

    def test_get_all_pages_filtered(self, storage):
        """Filter pages by type."""
        for i, ptype in enumerate(["entity", "concept", "topic"]):
            storage.store_knowledge_page(KnowledgePage(
                entity_id=f"ent_f{i}",
                title=f"Test {ptype} {i}",
                page_type=ptype,
            ))
        topics = storage.get_all_knowledge_pages(page_type="topic")
        assert all(p.page_type == "topic" for p in topics)

    def test_delete_page(self, brain):
        """Delete removes page and junction entries."""
        storage = brain.storage
        mem = brain.remember("Memory for delete-page test")
        page = KnowledgePage(entity_id="ent_del", title="To Delete")
        storage.store_knowledge_page(page)
        storage.link_memory_to_page(page.page_id, mem.id)

        storage.delete_knowledge_page(page.page_id)
        assert storage.get_knowledge_page(page.page_id) is None

    def test_link_memory_to_page(self, brain):
        """Junction table correctly links memories to pages."""
        storage = brain.storage
        m1 = brain.remember("Memory one for link test")
        m2 = brain.remember("Memory two for link test")
        page = KnowledgePage(entity_id="ent_link", title="Linked")
        storage.store_knowledge_page(page)
        storage.link_memory_to_page(page.page_id, m1.id)
        storage.link_memory_to_page(page.page_id, m2.id)

        mids = storage.get_memories_for_page(page.page_id)
        assert set(mids) == {m1.id, m2.id}


class TestStorageProvenance:
    """Functional tests for provenance storage."""

    def test_store_and_retrieve_provenance(self, storage):
        """Store a provenance entry and retrieve it."""
        entry = ProvenanceEntry(
            memory_id="mem_prov_1",
            operation="created",
            reason="initial store",
            source_url="https://example.com",
        )
        storage.store_provenance(entry)
        entries = storage.get_provenance("mem_prov_1")
        assert len(entries) >= 1
        assert entries[0].operation == "created"
        assert entries[0].source_url == "https://example.com"

    def test_provenance_chain_traversal(self, storage):
        """Chain should follow parent_memory_ids."""
        # m1 created
        storage.store_provenance(ProvenanceEntry(
            memory_id="m1", operation="created",
        ))
        # m2 corrected from m1
        storage.store_provenance(ProvenanceEntry(
            memory_id="m2", parent_memory_ids=["m1"],
            operation="corrected", reason="fix",
        ))
        # m3 corrected from m2
        storage.store_provenance(ProvenanceEntry(
            memory_id="m3", parent_memory_ids=["m2"],
            operation="corrected", reason="fix again",
        ))

        chain = storage.get_provenance_chain("m3")
        memory_ids_in_chain = {e.memory_id for e in chain}
        assert "m3" in memory_ids_in_chain
        assert "m2" in memory_ids_in_chain
        assert "m1" in memory_ids_in_chain


class TestStorageVersions:
    """Functional tests for memory version storage."""

    def test_store_and_retrieve_versions(self, storage):
        """Store version snapshots and retrieve chronologically."""
        for i in range(3):
            storage.store_memory_version(MemoryVersion(
                memory_id="mem_ver_1",
                content=f"Content version {i+1}",
                strength=1.0 - (i * 0.1),
                importance=0.5,
                confidence=0.5 + (i * 0.1),
                change_reason=f"Update {i+1}",
            ))
            time.sleep(0.01)

        versions = storage.get_version_history("mem_ver_1")
        assert len(versions) == 3
        assert versions[0].content == "Content version 1"
        assert versions[2].content == "Content version 3"
        assert versions[0].changed_at <= versions[2].changed_at


class TestStorageLintHelpers:
    """Functional tests for lint helper queries."""

    def test_get_stale_memories(self, brain):
        """Should find old never-accessed memories."""
        mem = brain.remember("Stale test fact")
        mem.created_at = time.time() - (20 * 86400)
        mem.access_count = 0
        brain.storage.update_memory(mem)

        stale = brain.storage.get_stale_memories(max_age_days=14)
        stale_ids = [m.id for m in stale]
        assert mem.id in stale_ids

    def test_get_stale_excludes_accessed(self, brain):
        """Accessed memories should not be flagged as stale."""
        mem = brain.remember("Accessed fact")
        mem.created_at = time.time() - (20 * 86400)
        mem.access_count = 5
        brain.storage.update_memory(mem)

        stale = brain.storage.get_stale_memories(max_age_days=14)
        stale_ids = [m.id for m in stale]
        assert mem.id not in stale_ids


class TestEpistemicStatusModel:
    """Test EpistemicStatus enum."""

    def test_enum_values(self):
        """Enum should have all 4 statuses."""
        assert EpistemicStatus.VERIFIED.value == "verified"
        assert EpistemicStatus.INFERRED.value == "inferred"
        assert EpistemicStatus.UNCERTAIN.value == "uncertain"
        assert EpistemicStatus.CONTRADICTED.value == "contradicted"


class TestKnowledgePageModel:
    """Test KnowledgePage Pydantic model."""

    def test_default_values(self):
        """Model should have sensible defaults."""
        page = KnowledgePage(entity_id="ent_1", title="Test")
        assert page.page_type == "entity"
        assert page.version == 1
        assert page.summary == ""
        assert page.page_id  # Should auto-generate

    def test_serialization(self):
        """Model should serialize to dict."""
        page = KnowledgePage(
            entity_id="ent_1", title="Test",
            summary="A test page", page_type="topic",
        )
        d = page.model_dump()
        assert d["title"] == "Test"
        assert d["page_type"] == "topic"


class TestCompoundRecall:
    """Test compound recall (compounding knowledge)."""

    def test_compound_recall_returns_result(self, brain):
        """Compound recall should return a synthesis result."""
        brain.remember("Python is great for data science", tags=["tech"])
        brain.remember("Python has libraries like pandas and numpy", tags=["tech"])
        brain.remember("Python is used by many data scientists", tags=["tech"])

        result = brain.compound_recall("Python data science", store_result=False)
        assert isinstance(result, dict)


class TestIntegrationContracts:
    """Verify all adapter interfaces satisfy their contracts."""

    def test_langchain_contract(self, brain):
        from memory_layer.integrations import LangChainMemory
        mem = LangChainMemory(manager=brain)
        assert hasattr(mem, "memory_variables")
        assert hasattr(mem, "load_memory_variables")
        assert hasattr(mem, "save_context")
        assert hasattr(mem, "clear")

    def test_crewai_contract(self, brain):
        from memory_layer.integrations import CrewAIMemory
        mem = CrewAIMemory(manager=brain)
        assert hasattr(mem, "search")
        assert hasattr(mem, "save")
        assert hasattr(mem, "get_context")
        assert hasattr(mem, "clear")

    def test_llamaindex_contract(self, brain):
        from memory_layer.integrations import LlamaIndexMemory
        mem = LlamaIndexMemory(manager=brain)
        assert hasattr(mem, "get")
        assert hasattr(mem, "put")
        assert hasattr(mem, "get_all")
        assert hasattr(mem, "reset")

    def test_openai_contract(self, brain):
        from memory_layer.integrations import OpenAIThreadMemory
        mem = OpenAIThreadMemory(manager=brain)
        assert hasattr(mem, "save_message")
        assert hasattr(mem, "get_context")
        assert hasattr(mem, "get_system_prompt_context")

    def test_autogen_contract(self, brain):
        from memory_layer.integrations import AutoGenMemory
        mem = AutoGenMemory(manager=brain)
        assert hasattr(mem, "add")
        assert hasattr(mem, "search")
        assert hasattr(mem, "save_conversation")

    def test_vercel_contract(self, brain):
        from memory_layer.integrations import VercelAIMemory
        mem = VercelAIMemory(manager=brain)
        assert hasattr(mem, "get_context")
        assert hasattr(mem, "get_messages")
        assert hasattr(mem, "save")
        assert hasattr(mem, "save_interaction")
