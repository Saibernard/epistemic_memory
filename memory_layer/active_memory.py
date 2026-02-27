"""
LLM-Driven Active Memory Management for the Memory Layer.

Goes beyond formula-driven decay and consolidation to add intelligent
judgment about what to keep, merge, and forget. Runs periodically
(e.g. every 100 operations or on-demand).

Four core operations:
1. Curate: Identify duplicates, outdated, and trivial memories
2. Promote: Flag frequently-recalled facts for abstraction
3. Adaptive Importance: Re-evaluate stale memories' relevance
4. Conflict Resolution: Resolve CONTRADICTS links
"""

from __future__ import annotations

import json
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import MemoryManager
    from .enrichment import EnrichmentPipeline

from .models import Memory, MemoryType, MemoryLink, LinkType
from .storage import MemoryStorage


_CURATE_PROMPT = """Review these memories stored for a user. Identify:
1. MERGE: memories that say the same thing differently (return pairs of IDs to merge)
2. OUTDATED: memories that are clearly stale or no longer relevant (return IDs)
3. TRIVIAL: memories that are too vague or unimportant (return IDs to deprioritize)

Memories:
{memories}

Respond ONLY with JSON:
{{
  "merge": [[id1, id2], ...],
  "outdated": [id1, ...],
  "trivial": [id1, ...]
}}"""

_PROMOTE_PROMPT = """These facts have been recalled multiple times, suggesting they represent stable patterns. Which ones should be promoted to higher-level knowledge?

Facts:
{facts}

For each promotable fact, explain what stable pattern or trait it reveals.
Respond ONLY with JSON:
{{
  "promote": [
    {{"id": "...", "pattern": "The user consistently..."}}
  ]
}}"""

_IMPORTANCE_PROMPT = """These memories have never been recalled in 7+ days. Rate their likely future relevance (0.0-1.0).

Memories:
{memories}

Respond ONLY with JSON:
{{
  "ratings": [
    {{"id": "...", "importance": 0.3, "reason": "..."}}
  ]
}}"""

_CONFLICT_PROMPT = """These memory pairs contradict each other. For each pair, determine which is more likely current/correct.

Pairs:
{pairs}

Respond ONLY with JSON:
{{
  "resolutions": [
    {{"keep_id": "...", "supersede_id": "...", "reason": "..."}}
  ]
}}"""


class ActiveMemoryManager:
    """
    LLM-driven memory curation that runs periodically to maintain
    memory health and quality.
    """

    def __init__(
        self,
        storage: MemoryStorage,
        enrichment: "EnrichmentPipeline",
        trigger_interval: int = 100,
    ):
        self.storage = storage
        self.enrichment = enrichment
        self.trigger_interval = trigger_interval
        self._last_run: float = 0
        self._operation_count: int = 0

    @property
    def has_llm(self) -> bool:
        return self.enrichment is not None and self.enrichment.has_llm

    def should_run(self) -> bool:
        self._operation_count += 1
        return self._operation_count >= self.trigger_interval

    def reset_counter(self):
        self._operation_count = 0
        self._last_run = time.time()

    def run_all(self, manager: "MemoryManager" = None) -> Dict[str, Any]:
        """Run all active memory management operations."""
        results = {
            "curate": self.curate(),
            "promote": self.promote(),
            "reassess_importance": self.reassess_importance(),
            "resolve_conflicts": self.resolve_conflicts(),
            "timestamp": time.time(),
        }
        self.reset_counter()
        return results

    def curate(self) -> Dict[str, Any]:
        """
        Load recent memories and ask the LLM to identify duplicates,
        outdated, and trivial entries.
        """
        stats = {"merged": 0, "outdated": 0, "trivial": 0}

        if not self.has_llm:
            return stats

        recent = self.storage.get_all_memories(active_only=True)
        recent.sort(key=lambda m: m.created_at, reverse=True)
        recent = recent[:50]

        if len(recent) < 5:
            return stats

        mem_lines = []
        for m in recent:
            mem_lines.append(f"- [{m.id[:8]}] ({m.memory_type.value}, imp={m.importance:.1f}): {m.content[:150]}")

        prompt = _CURATE_PROMPT.format(memories="\n".join(mem_lines))

        try:
            raw = self.enrichment.generate(prompt, max_tokens=500)
            result = _parse_json(raw)
            if not result:
                return stats

            id_map = {m.id[:8]: m for m in recent}

            for pair in result.get("merge", []):
                if len(pair) == 2:
                    m1 = _find_memory(pair[0], id_map)
                    m2 = _find_memory(pair[1], id_map)
                    if m1 and m2:
                        if m1.access_count >= m2.access_count:
                            m2.is_current = False
                            m2.strength = 0.3
                            m2.metadata["merged_into"] = m1.id
                            self.storage.update_memory(m2)
                        else:
                            m1.is_current = False
                            m1.strength = 0.3
                            m1.metadata["merged_into"] = m2.id
                            self.storage.update_memory(m1)
                        stats["merged"] += 1

            for mid_prefix in result.get("outdated", []):
                m = _find_memory(mid_prefix, id_map)
                if m:
                    m.is_current = False
                    m.strength = max(0.1, m.strength * 0.3)
                    self.storage.update_memory(m)
                    stats["outdated"] += 1

            for mid_prefix in result.get("trivial", []):
                m = _find_memory(mid_prefix, id_map)
                if m:
                    m.importance = max(0.1, m.importance * 0.5)
                    self.storage.update_memory(m)
                    stats["trivial"] += 1

        except Exception as e:
            print(f"  ! Active memory curate error: {e}")

        return stats

    def promote(self) -> Dict[str, Any]:
        """
        Find frequently-recalled Level-0 memories and flag them for
        promotion to higher abstraction levels.
        """
        stats = {"promoted": 0}

        candidates = [
            m for m in self.storage.get_all_memories(active_only=True)
            if m.access_count >= 3 and m.abstraction_level == 0
        ]

        if not candidates or not self.has_llm:
            return stats

        candidates.sort(key=lambda m: m.access_count, reverse=True)
        candidates = candidates[:20]

        fact_lines = []
        for m in candidates:
            fact_lines.append(
                f"- [{m.id[:8]}] (recalled {m.access_count}x): {m.content[:150]}"
            )

        prompt = _PROMOTE_PROMPT.format(facts="\n".join(fact_lines))

        try:
            raw = self.enrichment.generate(prompt, max_tokens=400)
            result = _parse_json(raw)
            if not result:
                return stats

            id_map = {m.id[:8]: m for m in candidates}

            for item in result.get("promote", []):
                m = _find_memory(item.get("id", ""), id_map)
                if m:
                    m.metadata["promote_pattern"] = item.get("pattern", "")
                    m.metadata["promote_flagged"] = time.time()
                    self.storage.update_memory(m)
                    stats["promoted"] += 1

        except Exception as e:
            print(f"  ! Active memory promote error: {e}")

        return stats

    def reassess_importance(self) -> Dict[str, Any]:
        """
        Re-evaluate importance for memories never recalled in 7+ days.
        """
        stats = {"reassessed": 0}

        if not self.has_llm:
            return stats

        cutoff = time.time() - (7 * 86400)
        stale = [
            m for m in self.storage.get_all_memories(active_only=True)
            if m.access_count == 0 and m.created_at < cutoff
        ]

        if not stale:
            return stats

        stale = stale[:30]
        mem_lines = []
        for m in stale:
            mem_lines.append(
                f"- [{m.id[:8]}] (imp={m.importance:.1f}): {m.content[:150]}"
            )

        prompt = _IMPORTANCE_PROMPT.format(memories="\n".join(mem_lines))

        try:
            raw = self.enrichment.generate(prompt, max_tokens=500)
            result = _parse_json(raw)
            if not result:
                return stats

            id_map = {m.id[:8]: m for m in stale}

            for rating in result.get("ratings", []):
                m = _find_memory(rating.get("id", ""), id_map)
                if m:
                    new_imp = float(rating.get("importance", m.importance))
                    m.importance = max(0.05, min(1.0, new_imp))
                    self.storage.update_memory(m)
                    stats["reassessed"] += 1

        except Exception as e:
            print(f"  ! Active memory reassess error: {e}")

        return stats

    def resolve_conflicts(self) -> Dict[str, Any]:
        """
        Load all CONTRADICTS link pairs and ask the LLM which is more
        likely current/correct.
        """
        stats = {"resolved": 0}

        if not self.has_llm:
            return stats

        contradict_links = self.storage.get_links_by_type(
            LinkType.CONTRADICTS, limit=50,
        )

        if not contradict_links:
            return stats

        all_ids = set()
        for link in contradict_links:
            all_ids.add(link.source_id)
            all_ids.add(link.target_id)

        loaded = self.storage.get_memories_by_ids(list(all_ids))
        mem_map = {m.id: m for m in loaded if m.is_active}

        pair_lines = []
        for link in contradict_links[:10]:
            m1 = mem_map.get(link.source_id)
            m2 = mem_map.get(link.target_id)
            if m1 and m2:
                pair_lines.append(
                    f"Pair: [{m1.id[:8]}] \"{m1.content[:100]}\" "
                    f"vs [{m2.id[:8]}] \"{m2.content[:100]}\""
                )

        if not pair_lines:
            return stats

        prompt = _CONFLICT_PROMPT.format(pairs="\n".join(pair_lines))

        try:
            raw = self.enrichment.generate(prompt, max_tokens=400)
            result = _parse_json(raw)
            if not result:
                return stats

            all_mems = {m.id[:8]: m for m in loaded if m.is_active}

            for res in result.get("resolutions", []):
                supersede_m = _find_memory(res.get("supersede_id", ""), all_mems)
                if supersede_m:
                    supersede_m.is_current = False
                    supersede_m.strength = 0.3
                    supersede_m.epistemic_status = "contradicted"
                    supersede_m.confidence = round(supersede_m.confidence * 0.5, 3)
                    supersede_m.metadata["conflict_resolved"] = time.time()
                    supersede_m.metadata["resolution_reason"] = res.get("reason", "")
                    self.storage.update_memory(supersede_m)
                    stats["resolved"] += 1

        except Exception as e:
            print(f"  ! Active memory conflict resolution error: {e}")

        return stats


def _parse_json(text: str) -> Optional[Dict]:
    """Safely parse JSON from LLM output."""
    if not text:
        return None
    text = text.strip()
    # Find JSON object in output
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def _find_memory(prefix: str, id_map: Dict[str, Memory]) -> Optional[Memory]:
    """Find a memory by its ID prefix in the map."""
    prefix = str(prefix).strip()
    if prefix in id_map:
        return id_map[prefix]
    for key, mem in id_map.items():
        if key.startswith(prefix) or prefix.startswith(key):
            return mem
    return None
