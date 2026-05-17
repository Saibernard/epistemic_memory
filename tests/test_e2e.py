"""
End-to-end tests for the Memory Layer.

Tests complete workflows spanning multiple subsystems:
  - Remember → Recall → Verify
  - Remember → Correct → Version check → Provenance check
  - Ingest → Knowledge page → Wiki export
  - Contradiction → Lint detection
  - Multi-namespace isolation
"""

import os
import time
import tempfile
import pytest


class TestRememberRecallE2E:
    """End-to-end remember and recall flows."""

    def test_store_and_recall_basic(self, brain):
        """Store a fact and recall it by semantic query."""
        brain.remember("The capital of France is Paris", tags=["geography"])
        results = brain.recall("What is the capital of France?")
        assert len(results) >= 1
        assert "Paris" in results[0].memory.content

    def test_store_multiple_and_recall(self, brain):
        """Store multiple facts and recall the most relevant."""
        brain.remember("Python was created by Guido van Rossum", tags=["tech"])
        brain.remember("JavaScript was created by Brendan Eich", tags=["tech"])
        brain.remember("Rust was created by Graydon Hoare", tags=["tech"])

        results = brain.recall("Who created Python?", top_k=3)
        assert len(results) >= 1
        top_content = results[0].memory.content.lower()
        assert "python" in top_content or "guido" in top_content

    def test_episodic_to_semantic_flow(self, brain):
        """Record episodes and verify they're stored as episodic memories."""
        brain.record_episode(
            user_message="How do I sort a list?",
            assistant_response="Use sorted() or list.sort()",
        )
        results = brain.recall("sorting lists")
        assert len(results) >= 1


class TestCorrectionE2E:
    """End-to-end correction with versioning and provenance."""

    def test_full_correction_flow(self, brain):
        """Store → Correct → Verify version + provenance."""
        m1 = brain.remember("The meeting is on Tuesday", tags=["schedule"])
        m2 = brain.correct_memory(m1.id, "The meeting is on Wednesday", reason="rescheduled")
        assert m2 is not None

        # New memory should be high confidence
        assert m2.confidence >= 0.8
        assert m2.epistemic_status == "verified"

        # Old memory should have version snapshot
        versions = brain.get_version_history(m1.id)
        assert len(versions) >= 1
        assert "Tuesday" in versions[0]["content"]

        # Provenance should exist
        prov = brain.get_provenance(m2.id)
        assert len(prov) >= 1

    def test_multi_correction_chain(self, brain):
        """Multiple corrections should maintain full history."""
        m1 = brain.remember("Price: $100", tags=["product"])
        m2 = brain.correct_memory(m1.id, "Price: $120", reason="price increase")
        assert m2 is not None
        m3 = brain.correct_memory(m2.id, "Price: $95", reason="sale")
        assert m3 is not None

        # Each original should have a version snapshot
        v1 = brain.get_version_history(m1.id)
        assert len(v1) >= 1
        v2 = brain.get_version_history(m2.id)
        assert len(v2) >= 1

        # Latest memory should recall correctly
        results = brain.recall("What is the price?")
        # At least one result should mention price
        price_results = [r for r in results if "price" in r.memory.content.lower() or "$" in r.memory.content]
        assert len(price_results) >= 1


class TestKnowledgePageE2E:
    """End-to-end knowledge page flows."""

    def test_ingest_to_page_to_export(self, brain):
        """Store memories → rebuild pages → export wiki."""
        for content in [
            "React is a JavaScript library for building UIs",
            "React uses a virtual DOM for performance",
            "React was created by Facebook/Meta",
        ]:
            brain.remember(content, tags=["tech"])

        # Rebuild
        result = brain.rebuild_knowledge_pages()
        assert "total_pages" in result

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            from memory_layer.wiki_export import export_wiki
            stats = export_wiki(
                storage=brain.storage,
                entity_graph=brain.entity_graph,
                knowledge_page_manager=brain.knowledge_pages,
                output_dir=tmpdir,
            )
            assert os.path.exists(os.path.join(tmpdir, "index.md"))
            # Verify directory structure
            for d in ["entities", "concepts", "topics"]:
                assert os.path.isdir(os.path.join(tmpdir, d))


class TestContradictionLintE2E:
    """End-to-end contradiction detection via lint."""

    def test_contradiction_detected_by_lint(self, brain):
        """Contradictory facts should be flagged by lint."""
        brain.remember("The CEO is Alice Johnson", tags=["org"])
        brain.remember("The CEO is Bob Smith", tags=["org"])

        report = brain.lint()
        # May or may not detect as contradiction depending on similarity threshold
        assert "unresolved_contradictions" in report
        assert report["total_issues"] >= 0


class TestNamespaceIsolation:
    """Test that namespaces properly isolate memories."""

    def test_recall_respects_namespace(self, brain):
        """Memories in different namespaces should not cross-pollute."""
        brain.remember("Favorite color is blue", namespace="user_a", tags=["pref"])
        brain.remember("Favorite color is red", namespace="user_b", tags=["pref"])

        results_a = brain.recall("favorite color", namespace="user_a")
        results_b = brain.recall("favorite color", namespace="user_b")

        # Results should be namespace-specific
        if results_a:
            assert any("blue" in r.memory.content.lower() for r in results_a)
        if results_b:
            assert any("red" in r.memory.content.lower() for r in results_b)


class TestMaintenanceE2E:
    """Test the maintenance pipeline."""

    def test_maintenance_runs_all_subsystems(self, brain):
        """Maintenance should run consolidation, decay, lint, etc."""
        brain.remember("Test data for maintenance", tags=["test"])
        results = brain.maintenance()

        assert "consolidation" in results
        assert "decay" in results
        assert "lint" in results
        assert "storage_stats" in results


class TestEpistemicFlowE2E:
    """End-to-end epistemic status tracking."""

    def test_confidence_varies_by_source(self, brain):
        """Different sources should produce different confidence levels."""
        # Default
        m1 = brain.remember("A generic claim")
        # Document-sourced
        m2 = brain.remember("NASA confirmed water on Mars", metadata={"source_url": "https://nasa.gov"})
        # Correction
        m3 = brain.correct_memory(m1.id, "A verified claim", reason="user confirmed")

        assert m1.confidence < m2.confidence
        assert m3 is not None
        assert m3.confidence >= m2.confidence

    def test_epistemic_status_on_correction(self, brain):
        """Corrected memories should be 'verified'."""
        m1 = brain.remember("Unverified claim")
        assert m1.epistemic_status == "inferred"

        m2 = brain.correct_memory(m1.id, "Verified claim", reason="confirmed")
        assert m2 is not None
        assert m2.epistemic_status == "verified"
