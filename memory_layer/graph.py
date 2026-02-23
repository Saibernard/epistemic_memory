"""
Associative Memory Graph for the Memory Layer.

Manages the web of connections between memories. This is inspired by
how human memory works - recalling "birthday" might activate memories
of "cake", "friends", "that surprise party in 2019", etc.

When a new memory is stored:
  1. Find similar existing memories (by embedding similarity)
  2. Create weighted links between them
  3. Boost links for shared tags and temporal proximity

When recalling:
  1. Find the most relevant memories
  2. Traverse their links to find associated context
  3. Deeper associations have lower activation (spreading activation)
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple, TYPE_CHECKING

from .models import Memory, MemoryLink, LinkType
from .storage import MemoryStorage

if TYPE_CHECKING:
    from .embeddings import EmbeddingEngine


class AssociativeGraph:
    """
    Manages associations between memories using a graph structure.

    Think of it as the brain's neural network of memories -
    activating one node spreads activation to connected nodes.
    """

    def __init__(
        self,
        storage: MemoryStorage,
        embeddings: "EmbeddingEngine",
        similarity_threshold: float = 0.4,
        max_links_per_memory: int = 10,
    ):
        self.storage = storage
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.max_links_per_memory = max_links_per_memory

    def create_associations(
        self,
        memory: Memory,
        candidates: List[Tuple[Memory, np.ndarray]],
    ) -> List[MemoryLink]:
        """
        Discover and create associations between a new memory and a set
        of pre-filtered candidate memories (typically from a FAISS search).

        Links are weighted by:
        - Embedding cosine similarity (semantic relatedness)
        - Shared tags (topical overlap)
        - Temporal proximity (events close in time)
        """
        if memory.embedding is None or not candidates:
            return []

        query_emb = np.array(memory.embedding, dtype=np.float32)
        links: List[MemoryLink] = []
        scored: list = []

        for existing_mem, existing_emb in candidates:
            if existing_mem.id == memory.id:
                continue

            norm_product = np.linalg.norm(query_emb) * np.linalg.norm(existing_emb)
            if norm_product == 0:
                continue
            similarity = float(np.dot(query_emb, existing_emb) / norm_product)

            shared_tags = set(memory.tags) & set(existing_mem.tags)
            tag_boost = len(shared_tags) * 0.1

            time_diff = abs(memory.created_at - existing_mem.created_at)
            temporal_boost = 0.1 if time_diff < 3600 else 0.0

            total_score = similarity + tag_boost + temporal_boost

            if total_score >= self.similarity_threshold:
                scored.append((
                    existing_mem, total_score, similarity, temporal_boost > 0
                ))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:self.max_links_per_memory]

        for existing_mem, score, similarity, is_temporal in scored:
            if is_temporal and similarity > 0.3:
                link_type = LinkType.TEMPORAL
            else:
                link_type = LinkType.SIMILAR

            link = MemoryLink(
                source_id=memory.id,
                target_id=existing_mem.id,
                link_type=link_type,
                weight=min(1.0, score),
            )

            self.storage.store_link(link)
            links.append(link)

        return links

    def get_associated_memories(
        self,
        memory_id: str,
        depth: int = 1,
        max_results: int = 10,
    ) -> List[Tuple[Memory, float]]:
        """
        Traverse the association graph from a starting memory.

        Uses breadth-first traversal with spreading activation -
        each hop reduces the activation weight, so direct associations
        are stronger than indirect ones.
        """
        visited: Set[str] = {memory_id}
        results: Dict[str, float] = {}

        current_layer = [memory_id]
        decay_per_hop = 0.6

        for d in range(depth):
            next_layer: list = []
            layer_weight = decay_per_hop ** d

            _skip_types = {LinkType.SUPERSEDED, LinkType.CONTRADICTS}
            for mid in current_layer:
                links = self.storage.get_links_for(mid)

                for link in links:
                    if link.link_type in _skip_types:
                        continue

                    neighbor_id = (
                        link.target_id if link.source_id == mid else link.source_id
                    )

                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        weight = link.weight * layer_weight
                        results[neighbor_id] = max(
                            results.get(neighbor_id, 0), weight
                        )
                        next_layer.append(neighbor_id)

            current_layer = next_layer

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        # Batch-load memory objects instead of N individual queries
        top_ids = [mid for mid, _ in sorted_results[:max_results]]
        loaded = {m.id: m for m in self.storage.get_memories_by_ids(top_ids)}

        associated: List[Tuple[Memory, float]] = []
        for mem_id, weight in sorted_results[:max_results]:
            memory = loaded.get(mem_id)
            if memory and memory.is_active:
                associated.append((memory, weight))

        return associated

    def find_contradictions(
        self,
        memory: Memory,
        candidates: List[Tuple[Memory, np.ndarray]],
        base_threshold: float = 0.72,
    ) -> List[Tuple[Memory, float]]:
        """
        Find existing memories that might contradict the new memory, using
        a pre-filtered candidate set (typically from FAISS search).

        Heuristic: memories with very high embedding similarity but
        different content may represent updated/conflicting information.

        Skips pairs that share the same source document (multi-chunk
        ingestion produces high-similarity sibling chunks that are
        complementary, not contradictory).

        ``base_threshold`` should be calibrated to the embedding model —
        higher-dimensional models compress the similarity space and need
        a higher threshold to avoid false positives.

        Returns list of (memory, similarity) tuples.
        """
        if memory.embedding is None or not candidates:
            return []

        query_emb = np.array(memory.embedding, dtype=np.float32)
        new_source = memory.metadata.get("source_file") if memory.metadata else None
        contradictions: List[Tuple[Memory, float]] = []

        for existing_mem, existing_emb in candidates:
            if existing_mem.id == memory.id:
                continue

            # Same-source protection: sibling chunks from the same document
            # are complementary parts, never contradictions.
            if new_source:
                old_source = (existing_mem.metadata or {}).get("source_file")
                if old_source and old_source == new_source:
                    continue

            norm_product = np.linalg.norm(query_emb) * np.linalg.norm(existing_emb)
            if norm_product == 0:
                continue
            similarity = float(np.dot(query_emb, existing_emb) / norm_product)

            if similarity > base_threshold and existing_mem.content != memory.content:
                if (existing_mem.memory_type == memory.memory_type
                        and existing_mem.is_active):
                    contradictions.append((existing_mem, similarity))

        contradictions.sort(key=lambda x: x[1], reverse=True)
        return contradictions

    def check_duplicate(
        self,
        content: str,
        candidates: List[Tuple[Memory, np.ndarray]],
        embedding: List[float],
        threshold: float = 0.99,
    ) -> Optional[Memory]:
        """
        Check whether *content* is an exact duplicate of an existing memory.

        Returns the existing Memory if a duplicate is found, else None.
        """
        if not candidates:
            return None

        query_emb = np.array(embedding, dtype=np.float32)
        content_lower = content.strip().lower()

        for mem, emb in candidates:
            norm_product = np.linalg.norm(query_emb) * np.linalg.norm(emb)
            if norm_product == 0:
                continue
            sim = float(np.dot(query_emb, emb) / norm_product)
            if sim >= threshold and mem.content.strip().lower() == content_lower:
                return mem

        return None
