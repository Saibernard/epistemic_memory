"""Tests for export/import and chat engine."""

import json
import os
import pytest

from memory_layer import MemoryManager
from memory_layer.export import export_brain, import_brain
from memory_layer.chat import ChatEngine, ChatResponse


class TestExport:
    def test_export_creates_file(self, brain, tmp_path):
        brain.remember("Python is the best programming language for data science")
        brain.remember("Dark mode is the preferred UI theme for the editor")

        out = str(tmp_path / "brain.json")
        summary = export_brain(brain.storage, out)

        assert os.path.exists(out)
        assert summary["memories_exported"] >= 1
        assert os.path.getsize(out) > 0

        with open(out) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["memories"]) >= 1

    def test_export_no_embeddings_by_default(self, brain, tmp_path):
        brain.remember("Test memory")
        out = str(tmp_path / "brain.json")
        export_brain(brain.storage, out)

        with open(out) as f:
            data = json.load(f)
        for mem in data["memories"]:
            assert "embedding" not in mem

    def test_export_with_embeddings(self, brain, tmp_path):
        brain.remember("Test memory")
        out = str(tmp_path / "brain.json")
        export_brain(brain.storage, out, include_embeddings=True)

        with open(out) as f:
            data = json.load(f)
        assert "embedding" in data["memories"][0]

    def test_export_namespace_filter(self, brain, tmp_path):
        brain.remember("Project Alpha uses React and TypeScript for frontend", namespace="alpha")
        brain.remember("Project Beta uses Vue and JavaScript for frontend", namespace="beta")

        out = str(tmp_path / "brain.json")
        summary = export_brain(brain.storage, out, namespace="alpha")
        assert summary["memories_exported"] >= 1
        with open(out) as f:
            data = json.load(f)
        for mem in data["memories"]:
            assert mem["namespace"] == "alpha"


class TestImport:
    def test_import_into_fresh_db(self, brain, tmp_path):
        brain.remember("Fact one")
        brain.remember("Fact two about different stuff")

        out = str(tmp_path / "export.json")
        export_brain(brain.storage, out)

        # Create a fresh brain and import
        fresh_db = str(tmp_path / "fresh.db")
        fresh_brain = MemoryManager(db_path=fresh_db)

        summary = import_brain(
            fresh_brain.storage, out,
            reembed=True,
            embeddings_engine=fresh_brain.embeddings,
        )
        assert summary["memories_imported"] == 2
        assert summary["memories_skipped"] == 0

    def test_import_skips_duplicates(self, brain, tmp_path):
        brain.remember("Unique fact")

        out = str(tmp_path / "export.json")
        export_brain(brain.storage, out)

        summary = import_brain(brain.storage, out)
        assert summary["memories_skipped"] >= 1

    def test_import_with_namespace_override(self, brain, tmp_path):
        brain.remember("Original namespace fact")

        out = str(tmp_path / "export.json")
        export_brain(brain.storage, out)

        fresh_db = str(tmp_path / "fresh.db")
        fresh_brain = MemoryManager(db_path=fresh_db)

        import_brain(
            fresh_brain.storage, out,
            target_namespace="imported",
            reembed=True,
            embeddings_engine=fresh_brain.embeddings,
        )
        memories = fresh_brain.storage.get_all_memories(namespace="imported")
        assert len(memories) >= 1
        assert all(m.namespace == "imported" for m in memories)


class TestChatLocal:
    def test_ask_with_results(self, brain):
        brain.remember("The project uses React for frontend")
        brain.remember("Backend is built with Django")

        engine = ChatEngine(brain=brain, mode="local")
        response = engine.ask("What tech stack is used?")

        assert isinstance(response, ChatResponse)
        assert response.mode == "local"
        assert len(response.answer) > 0

    def test_ask_no_results(self, brain):
        engine = ChatEngine(brain=brain, mode="local")
        response = engine.ask("What is quantum physics?")

        assert response.has_answer is False
        assert "don't have" in response.answer.lower()

    def test_ask_returns_sources(self, brain):
        brain.remember("Database is PostgreSQL version 15")

        engine = ChatEngine(brain=brain, mode="local")
        response = engine.ask("What database is used?")

        if response.has_answer:
            assert len(response.sources) > 0
            assert "id" in response.sources[0]
            assert "content" in response.sources[0]
            assert "relevance" in response.sources[0]

    def test_namespace_isolation(self, brain):
        brain.remember("Secret project data", namespace="secret")
        brain.remember("Public information", namespace="public")

        engine = ChatEngine(brain=brain, mode="local", namespace="public")
        response = engine.ask("project data")

        if response.has_answer:
            for src in response.sources:
                assert "Secret" not in src["content"]


class TestChatLLM:
    def test_llm_mode_requires_api_key(self, brain, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            ChatEngine(brain=brain, mode="llm")
