"""Tests for epistemic status and confidence tracking."""

import pytest


class TestEpistemic:
    """Test confidence and epistemic status assignment."""

    def test_default_confidence(self, brain):
        """Default memories should have ~0.5 confidence."""
        mem = brain.remember("A generic fact", tags=["test-epi"])
        assert 0.4 <= mem.confidence <= 0.6
        assert mem.epistemic_status == "inferred"

    def test_document_source_confidence(self, brain):
        """Memories with source URLs should get higher confidence."""
        mem = brain.remember(
            "Data from official report",
            metadata={"source_url": "https://example.com/report"},
            tags=["test-epi"],
        )
        assert mem.confidence >= 0.7

    def test_correction_boosts_confidence(self, brain):
        """Corrections (user-verified) should get highest confidence."""
        m1 = brain.remember("Original claim", tags=["test-epi"])
        m2 = brain.correct_memory(m1.id, "Corrected claim", reason="user verified")
        assert m2 is not None
        assert m2.confidence >= 0.8
        assert m2.epistemic_status == "verified"

    def test_confidence_persists(self, brain):
        """Confidence should survive store/retrieve cycle."""
        mem = brain.remember(
            "Persistent confidence test",
            metadata={"source_file": "report.pdf"},
            tags=["test-epi"],
        )
        retrieved = brain.storage.get_memory(mem.id)
        assert retrieved is not None
        assert retrieved.confidence == pytest.approx(mem.confidence, abs=0.01)
        assert retrieved.epistemic_status == mem.epistemic_status

    def test_semantic_memory_confidence(self, brain):
        """Semantic memories should get slightly higher confidence."""
        from memory_layer.models import MemoryType
        mem = brain.remember(
            "Python is a programming language",
            memory_type=MemoryType.SEMANTIC,
            tags=["test-epi"],
        )
        assert mem.confidence >= 0.6
