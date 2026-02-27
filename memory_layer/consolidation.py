"""
Memory Consolidation Engine for the Memory Layer.

This is the "sleep" process of the memory system. Just like the human
brain consolidates short-term memories into long-term knowledge during
sleep, this engine:

1. Finds clusters of related episodic memories
2. Synthesises genuine semantic knowledge via LLM (or keyword fallback)
3. Creates new semantic (long-term) memories
4. Weakens source episodes to compress the memory store
5. Links everything together

When an LLM backend is available (Ollama / OpenAI-compat / enrichment
pipeline), the summary is a real synthesis — not a template.  Falls back
to keyword extraction when no LLM is reachable.
"""

import uuid
import time
import numpy as np
from typing import List, Dict, Optional, Set, Tuple, TYPE_CHECKING
from collections import defaultdict

from .models import Memory, MemoryType, MemoryLink, LinkType
from .storage import MemoryStorage

if TYPE_CHECKING:
    from .embeddings import EmbeddingEngine
    from .enrichment import EnrichmentPipeline


_LLM_CONSOLIDATION_PROMPT = (
    "You are a knowledge synthesis engine. Given a set of related "
    "conversation snippets, produce a single concise factual summary "
    "(under 150 words) that captures the key knowledge, user preferences, "
    "decisions, and patterns. Do NOT include meta-commentary like "
    "'based on the conversations'. Just state the facts.\n\n"
    "Snippets:\n{snippets}\n\nSynthesized knowledge:"
)

_LLM_TRAIT_PROMPT = (
    "Given these observed patterns about a user, identify the stable trait "
    "or preference they reveal. Respond with a single concise statement "
    "(under 80 words) describing the trait. No meta-commentary.\n\n"
    "Patterns:\n{patterns}\n\nTrait:"
)

_LLM_IDENTITY_PROMPT = (
    "Given these personality traits and preferences, distill the core value "
    "or identity aspect they reflect. Respond with a single sentence "
    "(under 50 words). No meta-commentary.\n\n"
    "Traits:\n{traits}\n\nCore value:"
)


class ConsolidationEngine:
    """
    Converts episodic memories into semantic knowledge through
    clustering and LLM-powered synthesis (with keyword fallback).
    """

    def __init__(
        self,
        storage: MemoryStorage,
        embeddings: "EmbeddingEngine",
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.5,
        enrichment: "EnrichmentPipeline" = None,
    ):
        self.storage = storage
        self.embeddings = embeddings
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.enrichment = enrichment

    def run_consolidation(self) -> Dict:
        """
        Run a full consolidation cycle.

        Skips episodes that have already been consolidated in a previous
        cycle to avoid creating redundant semantic memories.

        Returns statistics about what was consolidated.
        """
        stats = {
            "episodes_analyzed": 0,
            "clusters_found": 0,
            "semantic_memories_created": 0,
            "links_created": 0,
            "episodes_weakened": 0,
        }

        already_consolidated = self.storage.get_consolidated_episode_ids()

        episodes_with_emb = self.storage.get_memories_with_embeddings(
            memory_type=MemoryType.EPISODIC
        )

        episodes_with_emb = [
            (mem, emb) for mem, emb in episodes_with_emb
            if mem.id not in already_consolidated
        ]

        if len(episodes_with_emb) < self.min_cluster_size:
            return stats

        stats["episodes_analyzed"] = len(episodes_with_emb)

        clusters = self._cluster_memories(episodes_with_emb)
        stats["clusters_found"] = len(clusters)

        for cluster in clusters:
            semantic_memory = self._extract_semantic(cluster)
            if semantic_memory:
                semantic_memory.embedding = self.embeddings.embed(
                    semantic_memory.content
                )
                if self._is_duplicate_semantic(semantic_memory):
                    continue
                self.storage.store_memory(semantic_memory)
                stats["semantic_memories_created"] += 1

                for episode in cluster:
                    link = MemoryLink(
                        source_id=episode.id,
                        target_id=semantic_memory.id,
                        link_type=LinkType.DERIVED,
                        weight=0.8,
                    )
                    self.storage.store_link(link)
                    stats["links_created"] += 1

                    episode.strength = max(0.15, episode.strength * 0.4)
                    self.storage.update_memory(episode)
                    stats["episodes_weakened"] += 1

                self.storage.log_consolidation(
                    consolidation_id=str(uuid.uuid4()),
                    source_ids=[e.id for e in cluster],
                    result_id=semantic_memory.id,
                    strategy="llm_synthesis" if self._has_llm() else "keyword_extraction",
                )

        return stats

    def _has_llm(self) -> bool:
        return self.enrichment is not None and self.enrichment.has_llm

    def _is_duplicate_semantic(self, candidate: Memory) -> bool:
        """Check if a near-identical semantic memory already exists."""
        if not candidate.embedding:
            return False
        try:
            existing = self.storage.get_all_memories(
                memory_type=MemoryType.SEMANTIC, active_only=True,
            )
            qv = np.array(candidate.embedding, dtype=np.float32)
            qn = np.linalg.norm(qv)
            if qn == 0:
                return False
            for ex in existing:
                if not ex.embedding:
                    continue
                ev = np.array(ex.embedding, dtype=np.float32)
                en = np.linalg.norm(ev)
                if en == 0:
                    continue
                sim = float(np.dot(ev, qv) / (en * qn))
                if sim >= 0.92:
                    return True
        except Exception:
            pass
        return False

    def _cluster_memories(
        self,
        memories_with_emb: List[Tuple[Memory, np.ndarray]],
    ) -> List[List[Memory]]:
        """
        Cluster memories by embedding similarity using centroid-aware
        greedy approach. Each candidate is compared to the running
        cluster centroid for tighter clusters.
        """
        if not memories_with_emb:
            return []

        used: set = set()
        clusters: List[List[Memory]] = []

        for i, (mem_i, emb_i) in enumerate(memories_with_emb):
            if i in used:
                continue

            cluster = [mem_i]
            cluster_embs = [emb_i]
            centroid = emb_i.copy()
            used.add(i)

            for j, (mem_j, emb_j) in enumerate(memories_with_emb):
                if j in used:
                    continue

                norm_product = np.linalg.norm(centroid) * np.linalg.norm(emb_j)
                if norm_product == 0:
                    continue
                similarity = float(np.dot(centroid, emb_j) / norm_product)

                if similarity >= self.similarity_threshold:
                    cluster.append(mem_j)
                    cluster_embs.append(emb_j)
                    centroid = np.mean(cluster_embs, axis=0)
                    used.add(j)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        return clusters

    def _extract_semantic(self, cluster: List[Memory]) -> Optional[Memory]:
        """
        Extract a semantic memory from a cluster of episodes.

        Uses LLM synthesis when available, falls back to keyword-based
        template summarization.
        """
        if not cluster:
            return None

        contents = [m.content for m in cluster]
        all_tags: List[str] = []
        for m in cluster:
            all_tags.extend(m.tags)

        top_keywords = self._extract_keywords(contents)

        if self._has_llm():
            summary = self._llm_synthesize(contents)
        else:
            summary = self._keyword_summarize(cluster, top_keywords)

        avg_importance = sum(m.importance for m in cluster) / len(cluster)
        unique_tags = list(set(all_tags))

        # Epistemic: a synthesized memory is *derived* knowledge. It inherits
        # the average confidence of its sources (capped, since synthesis can
        # introduce error) and is flagged "uncertain" when its sources conflict
        # or are weak. Fully local — works with the keyword fallback too.
        avg_confidence = sum(getattr(m, "confidence", 0.5) for m in cluster) / len(cluster)
        source_statuses = {getattr(m, "epistemic_status", "inferred") for m in cluster}
        if ({"contradicted", "uncertain"} & source_statuses) or avg_confidence < 0.6:
            epistemic_status = "uncertain"
        else:
            epistemic_status = "inferred"

        return Memory(
            memory_type=MemoryType.SEMANTIC,
            content=summary,
            importance=min(1.0, avg_importance + 0.2),
            strength=1.0,
            confidence=round(min(0.85, avg_confidence), 3),
            epistemic_status=epistemic_status,
            tags=unique_tags + ["consolidated"],
            source_episode_ids=[m.id for m in cluster],
            metadata={
                "source_count": len(cluster),
                "keywords": top_keywords,
                "consolidation_time": time.time(),
                "synthesis": "llm" if self._has_llm() else "keyword",
            },
        )

    def _llm_synthesize(self, contents: List[str]) -> str:
        """Use the enrichment LLM to produce a genuine synthesis."""
        snippets = "\n---\n".join(c[:500] for c in contents[:10])
        prompt = _LLM_CONSOLIDATION_PROMPT.format(snippets=snippets)
        result = self.enrichment.generate(prompt, max_tokens=300)
        if result and len(result.strip()) > 20:
            return result.strip()
        return self._keyword_summarize_from_contents(contents)

    def _keyword_summarize(
        self, cluster: List[Memory], keywords: List[str]
    ) -> str:
        best_episode = max(
            cluster,
            key=lambda m: m.access_count * 0.5 + m.importance * 0.5,
        )
        summary = f"Consolidated knowledge from {len(cluster)} interactions. "
        if keywords:
            summary += f"Key themes: {', '.join(keywords[:5])}. "
        summary += f"Core context: {best_episode.content[:300]}"
        return summary

    def _keyword_summarize_from_contents(self, contents: List[str]) -> str:
        keywords = self._extract_keywords(contents)
        best = max(contents, key=len)
        summary = f"Consolidated knowledge from {len(contents)} interactions. "
        if keywords:
            summary += f"Key themes: {', '.join(keywords[:5])}. "
        summary += f"Core context: {best[:300]}"
        return summary

    _STOP_WORDS = frozenset({
        'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'and',
        'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
        'neither', 'each', 'every', 'all', 'any', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
        'than', 'too', 'very', 'just', 'because', 'if', 'when', 'that',
        'this', 'it', 'i', 'you', 'he', 'she', 'we', 'they', 'my',
        'your', 'his', 'her', 'its', 'our', 'their', 'what', 'which',
        'who', 'whom', 'how', 'where', 'there', 'here', 'about', 'up',
        'out', 'then', 'them', 'these', 'those', 'me', 'him', 'us',
        'user', 'assistant', 'message', 'response',
    })

    def _extract_keywords(self, contents: List[str]) -> List[str]:
        word_counts: Dict[str, int] = defaultdict(int)
        for content in contents:
            words = content.lower().split()
            for word in words:
                word = word.strip('.,!?;:()[]{}"\'-')
                if len(word) > 2 and word not in self._STOP_WORDS:
                    word_counts[word] += 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top = [w for w, c in sorted_words[:10] if c >= 2]
        if not top:
            top = [w for w, _ in sorted_words[:5]]
        return top

    # ══════════════════════════════════════════════
    #  MULTI-LEVEL ABSTRACTION CONSOLIDATION (Phase 2A)
    # ══════════════════════════════════════════════

    def consolidate_level_0_to_1(self) -> Dict:
        """
        Level 0 (raw facts) -> Level 1 (patterns).
        This is the standard consolidation (episodic -> semantic), now
        setting abstraction_level=1 on results.
        """
        stats = self.run_consolidation()

        # Retroactively set abstraction_level on newly created semantics
        if stats["semantic_memories_created"] > 0:
            all_sem = self.storage.get_all_memories(
                memory_type=MemoryType.SEMANTIC, active_only=True,
            )
            for m in all_sem:
                if (
                    m.abstraction_level == 0
                    and "consolidated" in m.tags
                ):
                    m.abstraction_level = 1
                    self.storage.update_memory(m)
        return stats

    def consolidate_level_1_to_2(self) -> Dict:
        """
        Level 1 (patterns) -> Level 2 (traits/preferences).
        Clusters Level-1 semantic memories by theme and uses LLM to
        identify stable traits.
        """
        stats = {
            "patterns_analyzed": 0,
            "traits_created": 0,
            "links_created": 0,
        }

        level1_mems = [
            m for m in self.storage.get_all_memories(
                memory_type=MemoryType.SEMANTIC, active_only=True,
            )
            if m.abstraction_level == 1
        ]

        if len(level1_mems) < self.min_cluster_size:
            return stats

        level1_with_emb = []
        for m in level1_mems:
            if m.embedding:
                level1_with_emb.append(
                    (m, np.array(m.embedding, dtype=np.float32))
                )

        stats["patterns_analyzed"] = len(level1_with_emb)
        clusters = self._cluster_memories(level1_with_emb)

        for cluster in clusters:
            trait_mem = self._synthesize_trait(cluster)
            if trait_mem:
                trait_mem.embedding = self.embeddings.embed(trait_mem.content)
                self.storage.store_memory(trait_mem)
                stats["traits_created"] += 1

                for src in cluster:
                    link = MemoryLink(
                        source_id=src.id,
                        target_id=trait_mem.id,
                        link_type=LinkType.ABSTRACTS,
                        weight=0.9,
                    )
                    self.storage.store_link(link)
                    stats["links_created"] += 1

                self.storage.log_consolidation(
                    consolidation_id=str(uuid.uuid4()),
                    source_ids=[m.id for m in cluster],
                    result_id=trait_mem.id,
                    strategy="level_1_to_2",
                )

        return stats

    def consolidate_level_2_to_3(self) -> Dict:
        """
        Level 2 (traits) -> Level 3 (identity/values).
        Clusters Level-2 trait memories and distills core values.
        Triggered rarely.
        """
        stats = {
            "traits_analyzed": 0,
            "identity_created": 0,
            "links_created": 0,
        }

        level2_mems = [
            m for m in self.storage.get_all_memories(
                memory_type=MemoryType.SEMANTIC, active_only=True,
            )
            if m.abstraction_level == 2
        ]

        if len(level2_mems) < 2:
            return stats

        level2_with_emb = []
        for m in level2_mems:
            if m.embedding:
                level2_with_emb.append(
                    (m, np.array(m.embedding, dtype=np.float32))
                )

        stats["traits_analyzed"] = len(level2_with_emb)
        clusters = self._cluster_memories(level2_with_emb)

        for cluster in clusters:
            identity_mem = self._synthesize_identity(cluster)
            if identity_mem:
                identity_mem.embedding = self.embeddings.embed(identity_mem.content)
                self.storage.store_memory(identity_mem)
                stats["identity_created"] += 1

                for src in cluster:
                    link = MemoryLink(
                        source_id=src.id,
                        target_id=identity_mem.id,
                        link_type=LinkType.ABSTRACTS,
                        weight=0.95,
                    )
                    self.storage.store_link(link)
                    stats["links_created"] += 1

                self.storage.log_consolidation(
                    consolidation_id=str(uuid.uuid4()),
                    source_ids=[m.id for m in cluster],
                    result_id=identity_mem.id,
                    strategy="level_2_to_3",
                )

        return stats

    def run_multi_level_consolidation(self) -> Dict:
        """Run all consolidation levels in order."""
        results = {}
        results["level_0_to_1"] = self.consolidate_level_0_to_1()
        results["level_1_to_2"] = self.consolidate_level_1_to_2()
        results["level_2_to_3"] = self.consolidate_level_2_to_3()
        return results

    def _synthesize_trait(self, cluster: List[Memory]) -> Optional[Memory]:
        """Synthesize a Level-2 trait from Level-1 patterns."""
        if not cluster:
            return None

        contents = [m.content for m in cluster]
        all_tags = []
        for m in cluster:
            all_tags.extend(m.tags)

        if self._has_llm():
            patterns = "\n---\n".join(c[:300] for c in contents[:8])
            prompt = _LLM_TRAIT_PROMPT.format(patterns=patterns)
            summary = self.enrichment.generate(prompt, max_tokens=200)
            if not summary or len(summary.strip()) < 15:
                summary = self._keyword_summarize_from_contents(contents)
        else:
            summary = f"User trait: {self._extract_keywords(contents)[:3]}"

        avg_importance = sum(m.importance for m in cluster) / len(cluster)

        return Memory(
            memory_type=MemoryType.SEMANTIC,
            content=summary.strip(),
            importance=min(1.0, avg_importance + 0.3),
            strength=1.0,
            tags=list(set(all_tags + ["trait", "consolidated"])),
            source_episode_ids=[m.id for m in cluster],
            abstraction_level=2,
            metadata={
                "source_count": len(cluster),
                "consolidation_time": time.time(),
                "synthesis": "llm" if self._has_llm() else "keyword",
            },
        )

    def _synthesize_identity(self, cluster: List[Memory]) -> Optional[Memory]:
        """Synthesize a Level-3 identity/value from Level-2 traits."""
        if not cluster:
            return None

        contents = [m.content for m in cluster]
        all_tags = []
        for m in cluster:
            all_tags.extend(m.tags)

        if self._has_llm():
            traits = "\n---\n".join(c[:200] for c in contents[:6])
            prompt = _LLM_IDENTITY_PROMPT.format(traits=traits)
            summary = self.enrichment.generate(prompt, max_tokens=100)
            if not summary or len(summary.strip()) < 10:
                summary = f"Core value reflected in: {', '.join(self._extract_keywords(contents)[:3])}"
        else:
            summary = f"Core value: {self._extract_keywords(contents)[:2]}"

        avg_importance = sum(m.importance for m in cluster) / len(cluster)

        return Memory(
            memory_type=MemoryType.SEMANTIC,
            content=summary.strip(),
            importance=min(1.0, avg_importance + 0.4),
            strength=1.0,
            tags=list(set(all_tags + ["identity", "consolidated"])),
            source_episode_ids=[m.id for m in cluster],
            abstraction_level=3,
            metadata={
                "source_count": len(cluster),
                "consolidation_time": time.time(),
                "synthesis": "llm" if self._has_llm() else "keyword",
            },
        )
