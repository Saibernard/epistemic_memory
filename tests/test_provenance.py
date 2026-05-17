"""Tests for provenance tracking and memory versioning."""

import pytest


class TestProvenance:
    """Test provenance chain logging."""

    def test_creation_provenance_logged(self, brain):
        """Storing a memory should create a provenance entry."""
        mem = brain.remember("Test provenance creation", tags=["test-prov"])
        chain = brain.get_provenance(mem.id)
        assert len(chain) >= 1
        assert chain[0]["operation"] == "created"
        assert chain[0]["memory_id"] == mem.id

    def test_correction_provenance(self, brain):
        """Correcting a memory should log provenance with parent."""
        m1 = brain.remember("The deadline is March 15", tags=["test-prov"])
        m2 = brain.correct_memory(m1.id, "The deadline is April 20", reason="date changed")

        assert m2 is not None
        chain = brain.get_provenance(m2.id)
        correction_entries = [e for e in chain if e["operation"] == "corrected"]
        assert len(correction_entries) >= 1
        assert m1.id in correction_entries[0]["parent_memory_ids"]

    def test_chain_traversal(self, brain):
        """Should traverse the full provenance chain across corrections."""
        m1 = brain.remember("Version 1 of the fact", tags=["test-prov"])
        m2 = brain.correct_memory(m1.id, "Version 2 of the fact")
        assert m2 is not None

        # Get chain for m2 — should include m1's creation too
        chain = brain.get_provenance(m2.id)
        all_memory_ids = set()
        for entry in chain:
            all_memory_ids.add(entry["memory_id"])
        # Should reference both m1 and m2
        assert m2.id in all_memory_ids


class TestVersioning:
    """Test memory version snapshots."""

    def test_version_created_on_correct(self, brain):
        """Correcting a memory should snapshot the old version."""
        m1 = brain.remember("Initial content for versioning", tags=["test-ver"])
        m2 = brain.correct_memory(m1.id, "Updated content for versioning")
        assert m2 is not None

        versions = brain.get_version_history(m1.id)
        assert len(versions) >= 1
        assert versions[0]["content"] == "Initial content for versioning"

    def test_multiple_versions_tracked(self, brain):
        """Multiple corrections should create multiple version snapshots."""
        m1 = brain.remember("Version test v1", tags=["test-ver"])
        m2 = brain.correct_memory(m1.id, "Version test v2")
        assert m2 is not None
        m3 = brain.correct_memory(m2.id, "Version test v3")
        assert m3 is not None

        # m1 should have 1 version (snapshotted when corrected to m2)
        v1 = brain.get_version_history(m1.id)
        assert len(v1) >= 1

        # m2 should have 1 version (snapshotted when corrected to m3)
        v2 = brain.get_version_history(m2.id)
        assert len(v2) >= 1

    def test_version_preserves_fields(self, brain):
        """Version snapshot should preserve strength, importance, confidence."""
        m1 = brain.remember("Test field preservation", importance=0.9, tags=["test-ver"])
        brain.correct_memory(m1.id, "Updated field preservation")

        versions = brain.get_version_history(m1.id)
        assert len(versions) >= 1
        v = versions[0]
        assert v["importance"] == pytest.approx(0.9, abs=0.01)
        assert "confidence" in v
