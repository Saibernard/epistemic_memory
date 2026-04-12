"""
Memory Linter / Self-Audit System.

Runs 6 health checks across the memory system to detect issues:

  1. Unresolved contradictions — active contradicting memory pairs
  2. Stale memories — never accessed, older than threshold
  3. Orphan memories — no links or entity associations
  4. Outdated knowledge pages — pages older than newest linked memory
  5. Entity coverage gaps — frequently mentioned entities with no page
  6. Duplicates — active memory pairs with cosine similarity > 0.95

Returns a structured LintReport for display in the testbench or CLI.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional, List, Dict, Any

from .models import LintReport, LinkType


class MemoryLinter:
    """
    Self-audit system for memory health.

    Checks for common issues that degrade retrieval quality over time.
    """

    def __init__(self, storage, entity_graph=None, knowledge_pages=None, graph=None):
        self.storage = storage
        self.entity_graph = entity_graph
        self.knowledge_pages = knowledge_pages
        self.graph = graph

    def lint(self) -> Dict[str, Any]:
        """
        Run all 6 lint checks and return a structured report.
        """
        report = LintReport()

        report.unresolved_contradictions = self._check_contradictions()
        report.stale_memories = self._check_stale()
        report.orphan_memories = self._check_orphans()
        report.outdated_knowledge_pages = self._check_outdated_pages()
        report.entity_coverage_gaps = self._check_coverage_gaps()
        report.duplicates = self._check_duplicates()

        report.total_issues = (
            len(report.unresolved_contradictions)
            + len(report.stale_memories)
            + len(report.orphan_memories)
            + len(report.outdated_knowledge_pages)
            + len(report.entity_coverage_gaps)
            + len(report.duplicates)
        )
        report.generated_at = time.time()

        return report.model_dump()

    def _check_contradictions(self) -> List[Dict[str, Any]]:
        """Find active memory pairs linked by 'contradicts' that are both still current."""
        issues = []
        try:
            links = self.storage.get_links_by_type(LinkType.CONTRADICTS, limit=500)
        except (AttributeError, Exception):
            return issues

        for link in links:
            src = self.storage.get_memory(link.source_id)
            tgt = self.storage.get_memory(link.target_id)
            if (
                src and tgt
                and src.is_active and tgt.is_active
                and getattr(src, "is_current", True)
                and getattr(tgt, "is_current", True)
            ):
                issues.append({
                    "type": "contradiction",
                    "memory_a_id": src.id,
                    "memory_a_content": src.content[:200],
                    "memory_b_id": tgt.id,
                    "memory_b_content": tgt.content[:200],
                    "link_weight": link.weight,
                })
        return issues

    def _check_stale(self) -> List[Dict[str, Any]]:
        """Find active memories that have never been accessed and are old."""
        issues = []
        try:
            stale = self.storage.get_stale_memories(max_age_days=14)
        except (AttributeError, Exception):
            return issues

        for mem in stale[:50]:
            age_days = (time.time() - mem.created_at) / 86400
            issues.append({
                "type": "stale",
                "memory_id": mem.id,
                "content": mem.content[:200],
                "age_days": round(age_days, 1),
                "strength": mem.strength,
            })
        return issues

    def _check_orphans(self) -> List[Dict[str, Any]]:
        """Find active memories with zero links and zero entity associations."""
        issues = []
        try:
            orphans = self.storage.get_orphan_memories()
        except (AttributeError, Exception):
            return issues

        for mem in orphans[:50]:
            issues.append({
                "type": "orphan",
                "memory_id": mem.id,
                "content": mem.content[:200],
                "memory_type": mem.memory_type.value,
                "created_at": mem.created_at,
            })
        return issues

    def _check_outdated_pages(self) -> List[Dict[str, Any]]:
        """Find knowledge pages where the newest linked memory is newer than the page."""
        issues = []
        if not self.knowledge_pages:
            return issues

        try:
            outdated = self.knowledge_pages.get_outdated_pages()
        except (AttributeError, Exception):
            return issues

        for page in outdated[:20]:
            issues.append({
                "type": "outdated_page",
                "page_id": page.page_id,
                "title": page.title,
                "page_type": page.page_type,
                "last_updated": page.last_updated,
                "version": page.version,
            })
        return issues

    def _check_coverage_gaps(self) -> List[Dict[str, Any]]:
        """Find entities with 3+ mentions but no knowledge page."""
        issues = []
        if not self.entity_graph:
            return issues

        try:
            conn = self.entity_graph.storage._conn()
            entities = conn.execute(
                "SELECT entity_id, name, entity_type, mention_count "
                "FROM entity_nodes WHERE mention_count >= 3 "
                "ORDER BY mention_count DESC LIMIT 50"
            ).fetchall()
        except Exception:
            return issues

        for eid, name, etype, count in entities:
            page = self.storage.get_knowledge_page_by_entity(eid)
            if not page:
                issues.append({
                    "type": "coverage_gap",
                    "entity_id": eid,
                    "entity_name": name,
                    "entity_type": etype,
                    "mention_count": count,
                })
        return issues

    def _check_duplicates(self) -> List[Dict[str, Any]]:
        """Find active memory pairs with very high cosine similarity (>0.95)."""
        issues = []
        try:
            pairs = self.storage.get_memories_with_embeddings(min_strength=0.0)
        except Exception:
            return issues

        if len(pairs) < 2 or len(pairs) > 5000:
            return issues

        # Build matrix for batch comparison
        memories = [p[0] for p in pairs]
        embeddings = [p[1] for p in pairs]

        matrix = np.vstack(embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = matrix / norms

        # Compare in blocks to limit memory usage
        block_size = 500
        n = len(memories)
        found_pairs: set = set()

        for i in range(0, n, block_size):
            block = normalized[i:i + block_size]
            sims = block @ normalized.T
            for bi in range(len(block)):
                gi = i + bi
                for j in range(gi + 1, n):
                    if sims[bi, j] > 0.95:
                        pair_key = tuple(sorted([memories[gi].id, memories[j].id]))
                        if pair_key not in found_pairs:
                            found_pairs.add(pair_key)
                            issues.append({
                                "type": "duplicate",
                                "memory_a_id": memories[gi].id,
                                "memory_a_content": memories[gi].content[:200],
                                "memory_b_id": memories[j].id,
                                "memory_b_content": memories[j].content[:200],
                                "similarity": round(float(sims[bi, j]), 4),
                            })
                            if len(issues) >= 50:
                                return issues
        return issues
