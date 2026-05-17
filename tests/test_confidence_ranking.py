"""
Tests that the stored epistemic signals (confidence + epistemic_status) are
ACTUALLY USED — both written and read back into retrieval ranking.

These guard the claim in the paper that self-knowledge is operational, not
decorative. They are fully local (no LLM, no network): ranking tests use
exact-match queries so they are deterministic under any embedding backend,
including the hash fallback used in CI.
"""

import pytest

from memory_layer.models import Memory, MemoryType


def _recall_one(brain, query, mem_id, **kw):
    """Return the RecallResult for a specific memory id, or None."""
    for r in brain.recall(query, top_k=10, **kw):
        if r.memory.id == mem_id:
            return r
    return None


class TestConfidenceAffectsRanking:
    """The epistemic modifier must change retrieval scores (read path)."""

    def test_low_confidence_is_demoted(self, brain):
        content = "The deployment server is named atlas-prod-01"
        m = brain.remember(content, tags=["t"])
        # Force the stored reliability low + uncertain.
        m.confidence = 0.2
        m.epistemic_status = "uncertain"
        brain.storage.update_memory(m)

        base = _recall_one(brain, content, m.id, use_epistemic=False)
        epi = _recall_one(brain, content, m.id, use_epistemic=True)
        assert base is not None and epi is not None
        # With the epistemic modifier on, an unreliable memory scores lower
        # even though its query relevance is identical.
        assert epi.composite_score < base.composite_score

    def test_high_confidence_is_preserved(self, brain):
        content = "The CI pipeline uses GitHub Actions"
        m = brain.remember(content, tags=["t"])
        m.confidence = 1.0
        m.epistemic_status = "verified"
        brain.storage.update_memory(m)

        base = _recall_one(brain, content, m.id, use_epistemic=False)
        epi = _recall_one(brain, content, m.id, use_epistemic=True)
        assert base is not None and epi is not None
        # verified + confidence 1.0 => modifier == 1.0 => score unchanged.
        assert epi.composite_score == pytest.approx(base.composite_score, rel=0.05)

    def test_contradicted_scores_below_verified(self, brain):
        # Same memory, same query — only the stored epistemic state changes
        # between the two recalls, so any score delta is purely the modifier.
        content = "The primary datacenter is in Oregon"
        m = brain.remember(content, tags=["t"])

        m.confidence = 0.9
        m.epistemic_status = "verified"
        brain.storage.update_memory(m)
        r_verified = _recall_one(brain, content, m.id)

        m = brain.storage.get_memory(m.id)
        m.confidence = 0.2
        m.epistemic_status = "contradicted"
        brain.storage.update_memory(m)
        r_contra = _recall_one(brain, content, m.id)

        assert r_verified is not None and r_contra is not None
        assert r_contra.composite_score < r_verified.composite_score


class TestStatusActivation:
    """The previously-dead 'contradicted' and 'uncertain' statuses must fire."""

    def test_correction_marks_old_contradicted(self, brain):
        m1 = brain.remember("The API runs on port 8080", tags=["t"])
        m2 = brain.correct_memory(m1.id, "The API runs on port 9090", reason="user fix")
        assert m2 is not None and m2.epistemic_status == "verified"

        old = brain.storage.get_memory(m1.id)
        assert old is not None
        assert old.epistemic_status == "contradicted"
        assert old.confidence <= 0.3

    def test_consolidation_flags_uncertain_from_weak_sources(self, brain):
        # A cluster of weak / conflicting episodes -> the synthesized memory
        # should be flagged uncertain (avg confidence < 0.6 or a bad source).
        cluster = [
            Memory(content="User might prefer Vim", memory_type=MemoryType.EPISODIC,
                   confidence=0.4, epistemic_status="inferred"),
            Memory(content="User seemed unsure about Vim", memory_type=MemoryType.EPISODIC,
                   confidence=0.45, epistemic_status="uncertain"),
            Memory(content="User opened Vim once", memory_type=MemoryType.EPISODIC,
                   confidence=0.5, epistemic_status="inferred"),
        ]
        sem = brain.consolidation._extract_semantic(cluster)
        assert sem is not None
        assert sem.epistemic_status == "uncertain"
        assert sem.confidence <= 0.85

    def test_consolidation_stays_inferred_from_strong_sources(self, brain):
        cluster = [
            Memory(content="User ships with GitHub Actions", memory_type=MemoryType.EPISODIC,
                   confidence=0.8, epistemic_status="inferred"),
            Memory(content="User configured GitHub Actions CI", memory_type=MemoryType.EPISODIC,
                   confidence=0.8, epistemic_status="inferred"),
            Memory(content="User relies on GitHub Actions", memory_type=MemoryType.EPISODIC,
                   confidence=0.75, epistemic_status="inferred"),
        ]
        sem = brain.consolidation._extract_semantic(cluster)
        assert sem is not None
        assert sem.epistemic_status == "inferred"
        assert 0.6 <= sem.confidence <= 0.85
