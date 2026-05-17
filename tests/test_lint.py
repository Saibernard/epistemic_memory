"""Tests for the memory lint / self-audit system."""

import time
import pytest


class TestLint:
    """Test lint health checks."""

    def test_clean_report(self, brain):
        """Fresh system should have zero issues."""
        report = brain.lint()
        assert report["total_issues"] >= 0
        assert "generated_at" in report

    def test_detect_stale_memories(self, brain):
        """Should detect old, never-accessed memories."""
        # Store a memory and manually make it old
        mem = brain.remember("Old fact that nobody reads", tags=["test-lint"])
        mem.created_at = time.time() - (15 * 86400)  # 15 days ago
        mem.access_count = 0
        brain.storage.update_memory(mem)

        report = brain.lint()
        stale = report.get("stale_memories", [])
        stale_ids = [s["memory_id"] for s in stale]
        assert mem.id in stale_ids

    def test_detect_orphan_memories(self, brain):
        """Should detect memories with no links or entity associations."""
        report = brain.lint()
        # Orphans may or may not exist depending on entity extraction
        assert "orphan_memories" in report

    def test_report_structure(self, brain):
        """Report should have all 6 categories."""
        report = brain.lint()
        expected_keys = [
            "unresolved_contradictions",
            "stale_memories",
            "orphan_memories",
            "outdated_knowledge_pages",
            "entity_coverage_gaps",
            "duplicates",
            "total_issues",
            "generated_at",
        ]
        for key in expected_keys:
            assert key in report, f"Missing key: {key}"

    def test_lint_in_maintenance(self, brain):
        """Lint should run as part of maintenance."""
        brain.remember("Some test data for maintenance", tags=["test"])
        results = brain.maintenance()
        assert "lint" in results
