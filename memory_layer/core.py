"""
Memory Manager - The Core Orchestrator of the Memory Layer.

This is the "brain" that coordinates all memory subsystems:
- Working Memory (current context buffer)
- Episodic Memory (specific events and interactions)
- Semantic Memory (extracted facts and knowledge)
- Procedural Memory (learned patterns and workflows)
- Associative Graph (web of connections between memories)
- Decay Engine (forgetting curve + spaced repetition)
- Consolidation Engine (episodic → semantic promotion)

Usage:
    from memory_layer import MemoryManager
    
    brain = MemoryManager(db_path="my_brain.db")
    
    # Store
    brain.remember("User prefers dark mode", importance=0.8, tags=["preference"])
    
    # Recall
    results = brain.recall("What UI preferences does the user have?")
    for r in results:
        print(f"  {r.memory.content} (relevance={r.relevance_score:.2f})")
    
    # Record interaction
    brain.record_episode(
        user_message="How do I sort a list in Python?",
        assistant_response="Use sorted() or list.sort()...",
        feedback="positive"
    )
"""

import atexit
import os
import time
import threading
import re
import math
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from .models import (
    Memory,
    MemoryType,
    MemoryLink,
    LinkType,
    WorkingMemoryItem,
    RecallResult,
    MemoryStats,
    ProvenanceEntry,
    MemoryVersion,
)
from .storage_protocol import StorageBackend
from .storage_factory import create_storage
from .embeddings import create_embedding_engine
from .decay import DecayEngine
from .graph import AssociativeGraph
from .consolidation import ConsolidationEngine
from .faiss_index import MemoryIndex, FAISS_AVAILABLE
from .enrichment import EnrichmentPipeline
from .reranker import create_reranker
from .entity_graph import EntityGraph
from .temporal import (
    extract_temporal_refs, temporal_relevance, has_temporal_intent,
    has_historical_intent,
)
from .active_memory import ActiveMemoryManager
from .graph_reasoner import GraphReasoner, ReasoningChain
from .predictive import PredictiveCache
from .reasoning_engine import ReasoningEngine
from .knowledge_pages import KnowledgePageManager
from .lint import MemoryLinter


# ══════════════════════════════════════════════
#  Auto-load .env from project root (API keys, config overrides).
#  Only sets vars that aren't already in the environment.
# ══════════════════════════════════════════════
def _load_dotenv() -> None:
    for candidate in (
        pathlib.Path.cwd() / ".env",
        pathlib.Path(__file__).resolve().parent.parent / ".env",
    ):
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip("\"'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
            break

_load_dotenv()


# ══════════════════════════════════════════════
#  TUNING CONSTANTS (collected here for easy tuning)
# ══════════════════════════════════════════════

# --- Contradiction / auto-replace thresholds ---
# Base thresholds calibrated for 768d local embeddings.
# Higher-dimensional models (e.g. 3072d Gemini) compress the cosine
# similarity space, so these are scaled up dynamically at runtime via
# _scale_sim_threshold().
_BASE_SIM_THRESHOLDS = {
    "default": 0.82,
    "med":     0.78,   # >60 % token overlap
    "high":    0.76,   # >70 % token overlap
    "exact":   0.73,   # >75 % token overlap
}
_BASE_CONTRADICT_THRESHOLD = 0.72   # pre-filter in find_contradictions


def _scale_sim_threshold(base: float, dim: int) -> float:
    """
    Scale a similarity threshold based on embedding dimensionality.

    Higher-dimensional models produce inflated cosine similarities between
    unrelated texts.  We add a small offset so that the effective threshold
    stays discriminative regardless of model.

    Calibration anchor: 768d → offset 0 (thresholds as-is).
    """
    if dim <= 768:
        return base
    offset = min(0.12, 0.04 * (dim / 768 - 1))
    return min(0.97, base + offset)

# --- Recall composite weights ---
# Semantic (embedding cosine sim) is the primary retrieval signal.
# Cross-encoder reranker is the most accurate relevance judge.
# Other signals are secondary differentiators.
W_SEMANTIC    = 0.55
W_LEXICAL     = 0.03
W_PHRASE      = 0.03
W_ENTITY      = 0.03
W_IDENTIFIER  = 0.03
W_BREVITY     = 0.02
W_FTS         = 0.07
W_TYPE        = 0.03
W_ENTITY_GR   = 0.04
W_TEMPORAL    = 0.06     # Phase 1A: temporal relevance signal
W_ABSTRACTION = 0.03     # Phase 2A: abstraction level alignment
W_RERANK      = 0.35

# --- Epistemic reliability modifier (fully local, no LLM) ---
# A memory's *stored* self-assessed trustworthiness scales its query
# relevance, so an unreliable memory is demoted even when it matches the
# query well. confidence in [0,1] maps to a multiplier in
# [W_CONFIDENCE_FLOOR, 1.0]; epistemic_status applies a categorical penalty
# on top. This is what makes confidence/epistemic_status affect retrieval
# rather than being write-only metadata.
W_CONFIDENCE_FLOOR = 0.7
_EPISTEMIC_STATUS_PENALTY: Dict[str, float] = {
    "verified": 1.0,
    "inferred": 1.0,
    "uncertain": 0.85,
    "contradicted": 0.5,
}

# --- Memory type weights for scoring ---
TYPE_WEIGHTS: Dict[str, float] = {
    "semantic": 1.5,
    "procedural": 1.3,
    "episodic": 1.0,
}

# --- Recall intent boosts ---
INTENT_BOOST = 0.04

# --- Passage indexing ---
PASSAGE_CHAR_THRESHOLD = 400

# --- FAISS candidate pool for graph operations ---
GRAPH_CANDIDATE_K = 200


class MemoryManager:
    """
    The unified interface to the entire memory system.

    Initialize once, then use .remember(), .recall(), .record_episode()
    for all memory operations. The system handles embedding, linking,
    decay, consolidation, and all the biological memory dynamics
    automatically behind the scenes.
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        embedding_model: str = None,
        embedding_mode: str = "local",
        llm_extract: bool = False,
        llm_extract_model: str = "gpt-4o-mini",
        openai_api_key: str = None,
        working_memory_size: int = 20,
        auto_consolidate: bool = True,
        consolidation_interval: int = 50,
        decay_interval: int = 100,
        default_namespace: str = "default",
        storage: StorageBackend = None,
        storage_backend: str = None,
        enrichment_backend: str = "auto",
        enrichment_model: str = "phi3:mini",
        enrichment_url: str = "http://localhost:11434",
        reranker_mode: str = "auto",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reasoning_engine: str = "none",
        reasoning_model: str = None,
    ):
        embedding_mode = os.environ.get("MEMORY_EMBEDDING_MODE", embedding_mode)
        llm_extract = os.environ.get("MEMORY_LLM_EXTRACT", "1" if llm_extract else "0") == "1"
        enrichment_backend = os.environ.get("MEMORY_ENRICHMENT_BACKEND", enrichment_backend)
        enrichment_model = os.environ.get("MEMORY_ENRICHMENT_MODEL", enrichment_model)
        enrichment_url = os.environ.get("MEMORY_ENRICHMENT_URL", enrichment_url)
        reranker_mode = os.environ.get("MEMORY_RERANKER", reranker_mode)
        reasoning_engine = os.environ.get("MEMORY_REASONING_ENGINE", reasoning_engine)
        reasoning_model = os.environ.get("MEMORY_REASONING_MODEL", reasoning_model) or reasoning_model

        print("Initializing Memory Layer...")

        if storage is not None:
            self.storage = storage
        elif storage_backend is not None:
            self.storage = create_storage(storage_backend, sqlite_path=db_path)
        else:
            self.storage = create_storage("sqlite", sqlite_path=db_path)
        self.embeddings = create_embedding_engine(
            mode=embedding_mode,
            model_name=embedding_model,
            api_key=openai_api_key,
        )
        self.embedding_mode = embedding_mode
        self.decay_engine = DecayEngine()
        self.graph = AssociativeGraph(self.storage, self.embeddings)
        self.enrichment = EnrichmentPipeline(
            backend=enrichment_backend,
            model=enrichment_model,
            base_url=enrichment_url,
        )
        print(f"  + Enrichment: {self.enrichment.backend_name}")

        self.llm_extractor = None
        if llm_extract:
            try:
                from .llm_extract import LLMFactExtractor
                self.llm_extractor = LLMFactExtractor(
                    model=llm_extract_model,
                    api_key=openai_api_key,
                )
            except Exception as e:
                print(f"  ! OpenAI extraction unavailable ({e}), trying local variant...")
                try:
                    from .llm_extract import LocalFactExtractor
                    if self.enrichment.has_llm:
                        self.llm_extractor = LocalFactExtractor(enrichment=self.enrichment)
                    else:
                        print(f"  ! LLM extraction disabled: no local LLM available either")
                except Exception as e2:
                    print(f"  ! LLM extraction disabled: {e2}")

        self.consolidation = ConsolidationEngine(
            self.storage, self.embeddings, enrichment=self.enrichment,
        )

        self.reranker = create_reranker(mode=reranker_mode, model_name=reranker_model)
        if self.reranker.available:
            print(f"  + Reranker: cross-encoder ({reranker_model})")
        else:
            print(f"  + Reranker: disabled (install sentence-transformers for neural reranking)")

        self.entity_graph = EntityGraph(self.storage)
        ent_count = self.entity_graph.get_entity_count()
        if ent_count > 0:
            print(f"  + Entity graph: {ent_count} entities, {self.entity_graph.get_link_count()} links")

        self.active_memory = ActiveMemoryManager(
            storage=self.storage,
            enrichment=self.enrichment,
            trigger_interval=100,
        )
        self.graph_reasoner = GraphReasoner(manager=self)
        self.predictive_cache = PredictiveCache(manager=self)
        self.reasoning = ReasoningEngine(
            manager=self,
            mode=reasoning_engine,
            model=reasoning_model,
        )
        self.knowledge_pages = KnowledgePageManager(
            storage=self.storage,
            entity_graph=self.entity_graph,
            enrichment=self.enrichment,
        )
        self.linter = MemoryLinter(
            storage=self.storage,
            entity_graph=self.entity_graph,
            knowledge_pages=self.knowledge_pages,
            graph=self.graph,
        )

        index_base = db_path.rsplit(".", 1)[0] if "." in db_path else db_path
        dim = self.embeddings.dimension
        self.memory_index = MemoryIndex(
            dimension=dim, index_path=f"{index_base}_mem_idx",
        )
        self.passage_index = MemoryIndex(
            dimension=dim, index_path=f"{index_base}_pass_idx",
        )

        self.working_memory_size = working_memory_size
        self.auto_consolidate = auto_consolidate
        self.consolidation_interval = consolidation_interval
        self.decay_interval = decay_interval
        self.default_namespace = default_namespace

        self._operation_count = 0
        self._lock = threading.Lock()
        self._bg_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="memory_bg")
        self._shutting_down = False
        atexit.register(self.shutdown)

        self._contradiction_cache: Optional[Dict[str, float]] = None
        self._contradiction_cache_dirty = True

        self._last_episode_id: Optional[str] = None

        self._check_embedding_migration()
        self._ensure_faiss_indices()
        self._startup_health_check()

        if FAISS_AVAILABLE:
            print(f"  + FAISS index: {self.memory_index.size} memories, "
                  f"{self.passage_index.size} passages")
        else:
            print(f"  ! FAISS not installed — using numpy fallback (pip install faiss-cpu)")
        print(f"  + Embedding mode: {embedding_mode}")
        if self.llm_extractor:
            print(f"  + LLM extraction: enabled ({llm_extract_model})")
        print(f"  + Database: {db_path}")
        print(f"  + Memory Layer ready!\n")

    # ══════════════════════════════════════════════
    #  PRIMARY INTERFACE
    # ══════════════════════════════════════════════

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        namespace: str = None,
        embedding: List[float] = None,
    ) -> Memory:
        """
        Store a new memory.

        The memory will be:
        1. (Optional) LLM-extracted into atomic facts
        2. Analyzed for importance signals
        3. Embedded for semantic search
        4. Checked for contradictions with existing knowledge
        5. Stored persistently in SQLite
        6. Automatically linked to related existing memories

        When LLM extraction is enabled and the content contains multiple
        facts, each fact is stored as a separate memory for precise recall.
        Uses batch embedding for speed when multiple facts are extracted.
        The first memory object is returned (the "primary" memory).

        If *embedding* is provided (e.g. from a multimodal Gemini embed),
        it is used directly and the text-embed step is skipped.
        """
        namespace = namespace or self.default_namespace

        # Input validation
        if not content or not content.strip():
            raise ValueError("Cannot store empty content")
        content = content.strip()
        if len(content) > 100_000:
            raise ValueError(f"Content too large ({len(content)} chars, max 100000)")
        if self._shutting_down:
            raise RuntimeError("MemoryManager is shutting down")

        if (
            embedding is None
            and self.llm_extractor is not None
            and self.llm_extractor.should_extract(content)
        ):
            facts = self.llm_extractor.extract(content)
            if len(facts) > 1:
                return self._batch_store_extracted_facts(
                    facts=facts,
                    original_content=content,
                    memory_type=memory_type,
                    base_importance=importance,
                    base_tags=tags,
                    base_metadata=metadata,
                    namespace=namespace,
                )

        return self._store_single_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            metadata=metadata,
            namespace=namespace,
            embedding=embedding,
        )

    def _batch_store_extracted_facts(
        self,
        facts: List[Dict[str, Any]],
        original_content: str,
        memory_type: MemoryType,
        base_importance: float,
        base_tags: List[str] = None,
        base_metadata: Dict[str, Any] = None,
        namespace: str = "default",
    ) -> Memory:
        """
        Batch-store LLM-extracted facts with a single embedding API call.

        This is 5-20x faster than storing each fact individually because:
        - One batch embedding call instead of N serial calls
        - One FAISS save at the end instead of N

        Also stores the original chunk as a source_chunk and extracts
        temporal references and entity relationships from the LLM output.
        """
        # Phase 1B: Store original chunk as source chunk
        import uuid as _uuid
        source_chunk_id = str(_uuid.uuid4())
        source_file = (base_metadata or {}).get("source_file")
        chunk_index = (base_metadata or {}).get("chunk_index", 0)
        self.storage.store_source_chunk(
            source_chunk_id, original_content,
            source_file=source_file, chunk_index=chunk_index,
        )

        # Phase 1A: Extract temporal + relationship data from LLM output
        llm_event_dates = facts[0].pop("_event_dates", []) if facts else []
        llm_relationships = facts[0].pop("_relationships", []) if facts else []

        # Regex fallback for temporal extraction
        from .temporal import extract_temporal_refs
        if not llm_event_dates:
            refs = extract_temporal_refs(original_content)
            llm_event_dates = [
                {"date": r.resolved_date, "type": r.ref_type,
                 "description": r.description}
                for r in refs if r.resolved_date
            ]

        now = time.time()
        fact_contents = [f["content"] for f in facts]
        memories: List[Memory] = []
        for fact in facts:
            fact.pop("_event_dates", None)
            fact.pop("_relationships", None)
            fact_tags = list(set((base_tags or []) + fact.get("tags", [])))
            fact_importance = max(base_importance, fact.get("importance", 0.5))
            fact_importance = self._detect_importance(fact["content"], fact_importance)
            fact_meta = dict(base_metadata or {})
            fact_meta["llm_extracted"] = True
            fact_meta["original_content_preview"] = original_content[:200]

            memory = Memory(
                memory_type=memory_type,
                content=fact["content"],
                importance=fact_importance,
                tags=fact_tags,
                metadata=fact_meta,
                strength=1.0,
                access_count=0,
                namespace=namespace,
                document_date=now,
                event_dates=llm_event_dates if llm_event_dates else None,
            )
            memories.append(memory)

        embeddings = self.embeddings.embed_batch(fact_contents)

        for memory, emb in zip(memories, embeddings):
            memory.embedding = emb
            self.storage.store_memory(memory)
            self.memory_index.add(memory.id, emb)
            # Phase 1B: Link each memory to its source chunk
            self.storage.link_memory_to_chunk(memory.id, source_chunk_id)

        self.memory_index.save()

        for memory in memories:
            candidates = self._get_faiss_candidates(memory.embedding)
            self.graph.create_associations(memory, candidates)
            entity_text = memory.content
            self.entity_graph.index_memory(memory.id, entity_text)

        # Phase 1D: Store extracted entity relationships
        self._store_extracted_relationships(
            llm_relationships, memories[0].id if memories else None, now,
        )

        self._invalidate_contradiction_cache()
        self._increment_operations()
        return memories[0]

    def _store_extracted_relationships(
        self,
        relationships: List[Dict[str, Any]],
        memory_id: Optional[str],
        timestamp: float,
    ):
        """Store LLM-extracted entity relationship triples."""
        import uuid as _uuid
        for rel in relationships:
            subject = rel.get("subject", "").strip()
            relation = rel.get("relation", "").strip().upper()
            obj = rel.get("object", "").strip()
            if not subject or not relation or not obj:
                continue

            import hashlib
            src_eid = hashlib.md5(subject.lower().encode()).hexdigest()[:16]
            tgt_eid = hashlib.md5(obj.lower().encode()).hexdigest()[:16]

            self.entity_graph._ensure_entity(src_eid, subject, "concept")
            self.entity_graph._ensure_entity(tgt_eid, obj, "concept")

            self.storage.supersede_entity_relationships(
                src_eid, tgt_eid, relation,
            )

            self.storage.store_entity_relationship({
                "id": str(_uuid.uuid4()),
                "source_entity_id": src_eid,
                "target_entity_id": tgt_eid,
                "relation_type": relation,
                "context": rel.get("temporal", ""),
                "reasoning": None,
                "document_date": timestamp,
                "event_date": None,
                "valid_from": timestamp,
                "valid_until": None,
                "is_current": 1,
                "memory_id": memory_id,
                "confidence": 1.0,
                "created_at": timestamp,
            })

    def _store_single_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        namespace: str = "default",
        embedding: List[float] = None,
    ) -> Memory:
        """Internal: store a single memory (after any extraction has occurred)."""
        importance = self._detect_importance(content, importance)

        working = self.storage.get_working_memory(limit=5) if hasattr(self.storage, "get_working_memory") else []
        context_str = " | ".join(w.content for w in working) if working else ""

        enriched = self.enrichment.enrich(content, context=context_str)

        meta = metadata or {}
        if enriched and enriched != content:
            meta["enriched_content"] = enriched

        # Phase 1A: Extract temporal references
        now = time.time()
        temporal_refs = extract_temporal_refs(content)
        event_dates = [
            {"date": r.resolved_date, "type": r.ref_type,
             "description": r.description}
            for r in temporal_refs if r.resolved_date
        ] or None

        confidence, epistemic_status = self._compute_initial_confidence(
            content, meta, memory_type,
        )

        memory = Memory(
            memory_type=memory_type,
            content=content,
            importance=importance,
            tags=tags or [],
            metadata=meta,
            strength=1.0,
            access_count=0,
            namespace=namespace,
            document_date=now,
            event_dates=event_dates,
            confidence=confidence,
            epistemic_status=epistemic_status,
        )

        if embedding is not None:
            memory.embedding = embedding
        else:
            embed_text = enriched if enriched else content
            memory.embedding = self.embeddings.embed(embed_text)

        # Use FAISS to narrow candidates instead of full table scan
        candidates = self._get_faiss_candidates(memory.embedding)

        # Dedup: if an identical memory already exists, reinforce it instead
        existing = self.graph.check_duplicate(content, candidates, memory.embedding)
        if existing:
            existing = self.decay_engine.reinforce(existing, boost=0.1)
            self.storage.update_memory(existing)
            return existing

        # Check for contradictions against FAISS candidates only.
        # Pass a model-aware base threshold so that high-dimensional
        # embeddings don't trigger false positives in the pre-filter.
        dim = self.embeddings.dimension
        contradict_base = _scale_sim_threshold(
            _BASE_CONTRADICT_THRESHOLD, dim,
        )
        contradiction_pairs = self.graph.find_contradictions(
            memory, candidates, base_threshold=contradict_base,
        )

        # Secondary: catch updates missed by cosine threshold using word overlap.
        # "CEO is John Smith" → "CEO is Jane Doe" has 0.56 Jaccard and only 0.62
        # cosine — both below thresholds, but clearly an update of the same fact.
        seen_ids = {m.id for m, _ in contradiction_pairs}
        new_tokens = set(re.findall(r"[a-z0-9_]+", content.lower()))
        if new_tokens and len(content) < 500:
            min_jaccard = 0.4 if len(content) < 120 else 0.5
            for cand_mem, cand_emb in candidates:
                if cand_mem.id == memory.id or cand_mem.id in seen_ids:
                    continue
                new_source = memory.metadata.get("source_file") if memory.metadata else None
                if new_source and (cand_mem.metadata or {}).get("source_file") == new_source:
                    continue
                old_tokens = set(re.findall(r"[a-z0-9_]+", cand_mem.content.lower()))
                union = new_tokens | old_tokens
                jaccard = len(new_tokens & old_tokens) / len(union) if union else 0
                if jaccard >= min_jaccard and new_tokens != old_tokens:
                    emb = np.array(cand_emb, dtype=np.float32)
                    q = np.array(memory.embedding, dtype=np.float32)
                    sim = float(np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-9))
                    if sim >= 0.4:
                        contradiction_pairs.append((cand_mem, sim))
                        seen_ids.add(cand_mem.id)

        superseded_ids = []
        soft_contradictions = []
        now = time.time()

        st_default = _scale_sim_threshold(_BASE_SIM_THRESHOLDS["default"], dim)
        st_med     = _scale_sim_threshold(_BASE_SIM_THRESHOLDS["med"], dim)
        st_high    = _scale_sim_threshold(_BASE_SIM_THRESHOLDS["high"], dim)
        st_exact   = _scale_sim_threshold(_BASE_SIM_THRESHOLDS["exact"], dim)

        for old_mem, sim in contradiction_pairs:
            old_tokens = set(re.findall(r"[a-z0-9_]+", old_mem.content.lower()))
            union = new_tokens | old_tokens
            overlap = len(new_tokens & old_tokens) / len(union) if union else 0

            is_update = (
                (overlap >= 0.4 and len(content) < 120)
                or (overlap >= 0.5 and len(content) < 500)
                or sim >= st_default
            )

            if is_update:
                # Snapshot old version before superseding
                self._snapshot_version(old_mem, reason="superseded")

                # Phase 1E: Append-only versioning — keep old memory queryable
                old_mem.metadata["superseded_by"] = memory.id
                old_mem.metadata["valid_until"] = now
                if not old_mem.metadata.get("valid_from"):
                    old_mem.metadata["valid_from"] = old_mem.created_at

                old_mem.strength = 0.3  # reduced but not zeroed
                old_mem.is_active = True  # stay queryable for history
                old_mem.is_current = False  # mark as non-current
                # Epistemic: this memory was overridden by a contradicting update
                old_mem.epistemic_status = "contradicted"
                old_mem.confidence = round(min(old_mem.confidence, 0.3), 3)
                self.storage.update_memory(old_mem)

                # Remove from FAISS to avoid polluting primary retrieval,
                # but the memory remains in SQLite for historical queries
                try:
                    self.memory_index.remove(old_mem.id)
                except Exception:
                    pass

                superseded_ids.append(old_mem.id)
                memory.importance = max(memory.importance, old_mem.importance)
                print(
                    f"  ⟳ Superseded: \"{old_mem.content[:60]}...\" "
                    f"→ \"{memory.content[:60]}\" (sim={sim:.3f}, overlap={overlap:.2f})"
                )
            else:
                soft_contradictions.append(old_mem)

        memory.metadata["valid_from"] = now
        if superseded_ids:
            memory.metadata["replaces"] = superseded_ids
            memory.tags = list(set(memory.tags + ["updated"]))
        if soft_contradictions:
            memory.metadata["contradicts"] = [c.id for c in soft_contradictions]
            # Epistemic: an unresolved contradiction makes both sides uncertain
            # (neither was confidently superseded). User-verified memories are
            # left untouched.
            if memory.epistemic_status != "verified":
                memory.epistemic_status = "uncertain"
                memory.confidence = round(min(memory.confidence, 0.6), 3)
            for c in soft_contradictions:
                if c.epistemic_status not in ("verified", "contradicted"):
                    c.epistemic_status = "uncertain"
                    c.confidence = round(min(c.confidence, 0.6), 3)
                    self.storage.update_memory(c)

        # Atomic transaction: store memory + links + source chunks together
        links_to_store = []
        for old_id in superseded_ids:
            links_to_store.append(MemoryLink(
                source_id=old_id,
                target_id=memory.id,
                link_type=LinkType.SUPERSEDED,
                weight=1.0,
            ))
        if soft_contradictions:
            for c in soft_contradictions:
                links_to_store.append(MemoryLink(
                    source_id=memory.id,
                    target_id=c.id,
                    link_type=LinkType.CONTRADICTS,
                    weight=0.9,
                ))

        sc_id = None
        if len(content) > 200:
            import uuid as _uuid
            sc_id = str(_uuid.uuid4())

        try:
            with self.storage.transaction() as conn:
                self.storage.store_memory_on_conn(conn, memory)
                for link in links_to_store:
                    self.storage.store_link_on_conn(conn, link)
                if sc_id:
                    source_file = meta.get("source_file")
                    conn.execute(
                        "INSERT OR IGNORE INTO source_chunks "
                        "(id, content, source_file, chunk_index, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (sc_id, content, source_file, 0, time.time()),
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_source_map "
                        "(memory_id, chunk_id) VALUES (?, ?)",
                        (memory.id, sc_id),
                    )
        except Exception as e:
            print(f"  ✗ Atomic store failed for '{content[:50]}...': {e}")
            raise

        # FAISS updates (outside transaction — rebuilt from SQLite on mismatch)
        self.memory_index.add(memory.id, memory.embedding)

        # FTS index (best-effort, non-critical)
        fts_text = enriched if enriched else content
        self.storage.fts_index_memory(memory.id, fts_text)

        self._build_passages(memory)
        self.graph.create_associations(memory, candidates)

        entity_text = enriched if enriched else content
        self.entity_graph.index_memory(memory.id, entity_text)

        self.memory_index.save()
        if len(content) > PASSAGE_CHAR_THRESHOLD:
            self.passage_index.save()

        self._invalidate_contradiction_cache()
        self._increment_operations()

        self.reasoning.enqueue(memory)

        # Log provenance
        try:
            source_url = meta.get("source_url", meta.get("source_file", ""))
            parent_ids = superseded_ids if superseded_ids else []
            self.storage.store_provenance(ProvenanceEntry(
                memory_id=memory.id,
                parent_memory_ids=parent_ids,
                operation="created",
                reason="supersedes prior" if superseded_ids else "",
                source_url=source_url or "",
            ))
        except Exception:
            pass

        # Propagate to knowledge pages (background)
        try:
            self._bg_pool.submit(
                self.knowledge_pages.propagate_from_memory,
                memory.id, entity_text,
            )
        except Exception:
            pass

        return memory

    def remember_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        memory_type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.7,
        namespace: str = None,
    ) -> List[Memory]:
        """
        Ingest a batch of document chunks with sliding window context.

        Each chunk is enriched using its neighboring chunks as context,
        enabling entity resolution across chunk boundaries (e.g., resolving
        "that framework" -> "React" from neighboring chunks).

        Args:
            chunks: List of dicts with "content", "tags", "metadata" keys
                    (as returned by DocumentIngestor.extract_and_chunk).
            memory_type: Memory type for all chunks.
            importance: Base importance for all chunks.
            namespace: Memory namespace.

        Returns:
            List of stored Memory objects.
        """
        namespace = namespace or self.default_namespace
        contents = [c["content"] for c in chunks]
        window_size = 2  # chunks before and after

        memories = []
        for i, chunk in enumerate(chunks):
            # Build sliding window context from neighboring chunks
            start = max(0, i - window_size)
            end = min(len(contents), i + window_size + 1)
            neighbors = []
            for j in range(start, end):
                if j != i:
                    neighbors.append(contents[j][:500])

            sliding_context = " | ".join(neighbors) if neighbors else ""

            # Enrich with sliding window context
            enriched = self.enrichment.enrich(
                chunk["content"], context=sliding_context,
            )

            meta = dict(chunk.get("metadata", {}))
            if enriched and enriched != chunk["content"]:
                meta["enriched_content"] = enriched
            meta["sliding_window_enriched"] = True

            mem = self.remember(
                content=chunk["content"],
                memory_type=memory_type,
                importance=importance,
                tags=chunk.get("tags", []),
                metadata=meta,
                namespace=namespace,
            )
            memories.append(mem)

        return memories

    def recall(
        self,
        query: str,
        memory_types: List[MemoryType] = None,
        top_k: int = 5,
        min_strength: float = 0.1,
        min_confidence: float = 0.05,
        include_associations: bool = True,
        tags: List[str] = None,
        namespace: str = None,
        include_history: bool = False,
        reasoning: bool = False,
        use_epistemic: bool = True,
    ) -> List[RecallResult]:
        """
        Two-stage recall pipeline:

        **Stage 1 — FAISS coarse retrieval** (fast, O(log n)):
            Pull a broad set of candidate memory IDs from both the memory
            and passage FAISS indices.

        **Stage 2 — Fine re-ranking** (precise, applied to candidates only):
            Composite scoring: semantic + BM25/IDF lexical + phrase-match +
            entity overlap + intent boosts + strength/importance modifiers.

        Final:
            Reinforce recalled memories (spaced repetition) and follow
            association links for deeper context.
        """
        namespace = namespace or self.default_namespace
        query_embedding = self.embeddings.embed(query)
        query_emb_np = np.array(query_embedding, dtype=np.float32)
        query_tokens = self._tokenize(query)
        query_has_identifier = any(
            any(ch.isdigit() for ch in t) or re.fullmatch(r"[a-z]+[_-]?\d+", t)
            for t in query_tokens
        )

        query_lower = query.lower()
        is_temporal = has_temporal_intent(query)
        is_historical = has_historical_intent(query) or include_history
        is_preference = bool(re.search(
            r"\b(prefer|favorite|favourite|like|dislike|opinion|choice|hate|love)\b",
            query_lower,
        ))

        contradiction_penalties = self._get_contradiction_penalties()

        expanded_queries = self._expand_query(query)

        # ══════════════════════════════════════════════
        #  STAGE 1: FAISS coarse retrieval + FTS5
        # ══════════════════════════════════════════════
        total_indexed = max(self.memory_index.size, 1)
        if total_indexed <= 5000:
            candidate_k = total_indexed
        else:
            candidate_k = max(top_k * 20, 500)

        # Phase 3B: Check predictive cache for pre-fetched candidates
        cached = self.predictive_cache.check_cache(query_embedding)

        mem_hits = self.memory_index.search(query_embedding, k=candidate_k)
        candidate_ids: Dict[str, float] = {}
        # Seed with cache hits if available
        if cached:
            for mid, score in cached:
                candidate_ids[mid] = max(candidate_ids.get(mid, 0.0), score)
        for mid, score in mem_hits:
            candidate_ids[mid] = max(candidate_ids.get(mid, 0.0), score)

        for alt_q in expanded_queries[1:]:
            try:
                alt_emb = self.embeddings.embed(alt_q)
                alt_hits = self.memory_index.search(alt_emb, k=min(candidate_k, 50))
                for mid, score in alt_hits:
                    discounted = score * 0.85
                    candidate_ids[mid] = max(candidate_ids.get(mid, 0.0), discounted)
            except Exception:
                pass

        pass_candidate_k = min(self.passage_index.size, candidate_k * 2) if self.passage_index.size > 0 else 0
        if pass_candidate_k > 0:
            pass_hits = self.passage_index.search(query_embedding, k=pass_candidate_k)
            passage_best: Dict[str, float] = {}
            for pkey, score in pass_hits:
                parent_mid = pkey.split("::")[0] if "::" in pkey else pkey
                passage_best[parent_mid] = max(passage_best.get(parent_mid, 0.0), score)
                if parent_mid not in candidate_ids:
                    candidate_ids[parent_mid] = 0.0
        else:
            passage_best = {}

        fts_boost: Dict[str, float] = {}
        if hasattr(self.storage, "fts_search"):
            for eq in expanded_queries:
                fts_hits = self.storage.fts_search(eq, limit=candidate_k)
                for mid, bm25_score in fts_hits:
                    normalized = max(0.0, 1.0 + bm25_score * 0.1)
                    fts_boost[mid] = max(fts_boost.get(mid, 0.0), min(1.0, normalized))
                    if mid not in candidate_ids:
                        candidate_ids[mid] = 0.0

        entity_boost: Dict[str, float] = {}
        try:
            entity_hits = self.entity_graph.expand_from_query(query, limit=candidate_k)
            for mid, escore in entity_hits:
                entity_boost[mid] = min(1.0, escore)
                if mid not in candidate_ids:
                    candidate_ids[mid] = 0.0
        except Exception:
            pass

        if not candidate_ids:
            return []

        # ══════════════════════════════════════════════
        #  Load candidate Memory objects from storage
        # ══════════════════════════════════════════════
        types_to_search = set(memory_types) if memory_types else None
        tag_set = set(tags) if tags else None
        loaded = self.storage.get_memories_by_ids(list(candidate_ids.keys()))
        candidates: List[Memory] = []
        for mem in loaded:
            if not mem.is_active:
                continue
            # Phase 1E: Filter non-current unless historical query
            if not mem.is_current and not is_historical:
                continue
            if types_to_search and mem.memory_type not in types_to_search:
                continue
            if namespace and mem.namespace != namespace:
                continue
            if tag_set and not (tag_set & set(mem.tags)):
                continue
            candidates.append(mem)

        if not candidates:
            return []

        # ══════════════════════════════════════════════
        #  STAGE 2: Fine re-ranking
        # ══════════════════════════════════════════════
        doc_token_sets: list = []
        df: Dict[str, int] = {}
        for memory in candidates:
            tokens = set(self._tokenize(memory.content))
            doc_token_sets.append(tokens)
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        total_docs = len(candidates)

        raw_results: list = []

        for i, memory in enumerate(candidates):
            full_sim = max(0.0, candidate_ids.get(memory.id, 0.0))
            pass_sim = passage_best.get(memory.id, 0.0)
            if pass_sim > 0 and full_sim > 0:
                semantic = 0.6 * max(full_sim, pass_sim) + 0.4 * min(full_sim, pass_sim)
            else:
                semantic = max(full_sim, pass_sim)

            effective_strength = self.decay_engine.compute_current_strength(memory)
            if effective_strength < min_strength:
                continue

            mem_tokens = doc_token_sets[i]
            lexical_score = self._idf_lexical_score(
                query_tokens, mem_tokens, df, total_docs
            )
            phrase_score = self._phrase_match_score(query, memory.content)
            entity_boost_val = self._entity_overlap_boost(query_tokens, mem_tokens, df)
            identifier_boost = self._identifier_match_boost(query_tokens, mem_tokens)

            content_len = len(memory.content)
            brevity_bonus = 0.0
            if content_len < 150:
                brevity_bonus = 0.08
            elif content_len < 300:
                brevity_bonus = 0.04

            fts_score = fts_boost.get(memory.id, 0.0)
            type_weight = TYPE_WEIGHTS.get(memory.memory_type.value, 1.0)
            ent_graph_score = entity_boost.get(memory.id, 0.0)

            # Phase 1A: Temporal relevance scoring
            temporal_score = 0.0
            if is_temporal:
                temporal_score = temporal_relevance(
                    query, memory.event_dates, memory.document_date,
                )

            # Phase 2A: Abstraction level alignment
            abstraction_score = 0.0
            if is_preference and memory.abstraction_level >= 2:
                abstraction_score = 0.5 + memory.abstraction_level * 0.15
            elif not is_preference and memory.abstraction_level == 0:
                abstraction_score = 0.3

            base_relevance = (
                W_SEMANTIC    * semantic
                + W_LEXICAL     * lexical_score
                + W_PHRASE      * phrase_score
                + W_ENTITY      * entity_boost_val
                + W_IDENTIFIER  * identifier_boost
                + W_BREVITY     * brevity_bonus
                + W_FTS         * fts_score
                + W_TYPE        * (type_weight - 1.0)
                + W_ENTITY_GR   * ent_graph_score
                + W_TEMPORAL    * temporal_score
                + W_ABSTRACTION * abstraction_score
            )

            content_lower = memory.content.lower()
            if is_temporal and re.search(
                r"(date|when|year|month|ago|time|yesterday|last)", content_lower
            ):
                base_relevance += INTENT_BOOST
            if is_preference and re.search(
                r"(prefer|favorite|favourite|like|dislike|love|hate)", content_lower
            ):
                base_relevance += INTENT_BOOST

            if query_has_identifier and identifier_boost == 0.0:
                base_relevance *= 0.80

            confidence = max(0.0, min(1.0, base_relevance))

            strength_mod = 0.75 + 0.25 * effective_strength
            importance_mod = 0.65 + 0.35 * memory.importance
            contradiction_penalty = contradiction_penalties.get(memory.id, 1.0)

            # Phase 1E: Penalize non-current memories rather than filtering them
            history_penalty = 1.0
            if not memory.is_current:
                history_penalty = 0.4 if is_historical else 0.15

            superseded_penalty = 1.0
            if memory.metadata.get("superseded_by") and memory.is_current:
                superseded_penalty = 0.5

            # Epistemic reliability modifier from the memory's *stored*
            # confidence + epistemic_status (distinct from the relevance-derived
            # `confidence` local above). Toggleable for ablation.
            if use_epistemic:
                stored_conf = max(0.0, min(1.0, memory.confidence))
                epistemic_mod = (
                    W_CONFIDENCE_FLOOR
                    + (1.0 - W_CONFIDENCE_FLOOR) * stored_conf
                ) * _EPISTEMIC_STATUS_PENALTY.get(memory.epistemic_status, 1.0)
            else:
                epistemic_mod = 1.0

            composite = (
                confidence * strength_mod * importance_mod
                * contradiction_penalty * superseded_penalty
                * history_penalty * epistemic_mod
            )

            raw_results.append({
                "memory": memory,
                "relevance": base_relevance,
                "lexical_score": lexical_score,
                "confidence": confidence,
                "effective_strength": effective_strength,
                "epistemic_mod": epistemic_mod,
                "composite": composite,
            })

        # ══════════════════════════════════════════════
        #  STAGE 3: Neural reranking + final sort
        # ══════════════════════════════════════════════
        raw_results.sort(key=lambda x: x["composite"], reverse=True)
        raw_results = [
            r for r in raw_results if r["confidence"] >= min_confidence
        ]

        rerank_pool = raw_results[:max(top_k * 3, 15)]

        if self.reranker.available and len(rerank_pool) > 1:
            texts = [r["memory"].content for r in rerank_pool]
            try:
                ce_scores = self.reranker.score_pairs(query, texts)
                ce_min = min(ce_scores) if ce_scores else 0.0
                ce_max = max(ce_scores) if ce_scores else 1.0
                ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

                for r, ce_raw in zip(rerank_pool, ce_scores):
                    ce_norm = (ce_raw - ce_min) / ce_range
                    old_composite = r["composite"]
                    r["composite"] = (
                        old_composite * (1.0 - W_RERANK) + ce_norm * W_RERANK
                    )
                    r["relevance"] = (
                        r["relevance"] * (1.0 - W_RERANK) + ce_norm * W_RERANK
                    )
                    r["confidence"] = max(0.0, min(1.0, r["relevance"]))

                rerank_pool.sort(key=lambda x: x["composite"], reverse=True)
            except Exception:
                pass

        raw_results = rerank_pool[:top_k]

        recall_results: List[RecallResult] = []

        for r in raw_results:
            memory = r["memory"]

            memory = self.decay_engine.reinforce(memory)
            self.storage.update_memory(memory)

            associations: List[str] = []
            if include_associations:
                associated = self.graph.get_associated_memories(
                    memory.id, depth=2, max_results=3
                )
                associations = [m.id for m, w in associated]

            recall_results.append(RecallResult(
                memory=memory,
                relevance_score=r["relevance"],
                effective_strength=r["effective_strength"],
                confidence=r["confidence"],
                lexical_score=r["lexical_score"],
                composite_score=r["composite"],
                associations=associations,
            ))

        if self._operation_count % 25 == 0:
            self.memory_index.save()
            self.passage_index.save()

        # Phase 3B: Trigger predictive pre-fetching for next query
        if recall_results:
            working = self.storage.get_working_memory(limit=3)
            context = " | ".join(w.content for w in working) if working else ""
            self.predictive_cache.predict_and_cache(
                query, recall_results, context,
            )

        # Phase 3A: Multi-hop graph reasoning
        if reasoning and recall_results:
            seed_memories = [r.memory for r in recall_results]
            chain = self.graph_reasoner.reason(query, seed_memories)
            if chain.nodes:
                for r in recall_results:
                    r.memory.metadata["_reasoning_chain"] = {
                        "synthesis": chain.synthesis,
                        "hops": chain.hops_taken,
                        "chain_length": len(chain.nodes),
                        "confidence": chain.confidence,
                        "nodes": [
                            {"content": n.content[:150], "hop": n.hop,
                             "link_type": n.link_type, "reason": n.reason}
                            for n in chain.nodes
                        ],
                    }
                    break  # Only attach to first result

        return recall_results

    def synthesize(
        self,
        topic: str,
        top_k: int = 20,
        store_result: bool = False,
        namespace: str = None,
    ) -> Dict[str, Any]:
        """
        Synthesize knowledge about a topic from all relevant memories.

        Recalls memories, groups them by type and relevance, and produces
        a coherent synthesis. If store_result=True, saves the synthesis
        as a high-abstraction semantic memory.

        Returns:
            {
                "topic": str,
                "synthesis": str,
                "source_count": int,
                "sources": [{"id": str, "content": str, "score": float, "type": str}],
                "stored_memory_id": str or None,
            }
        """
        namespace = namespace or self.default_namespace

        results = self.recall(
            query=topic,
            top_k=top_k,
            namespace=namespace,
            include_associations=True,
            reasoning=True,
        )

        if not results:
            return {
                "topic": topic,
                "synthesis": f"No memories found related to: {topic}",
                "source_count": 0,
                "sources": [],
                "stored_memory_id": None,
            }

        sources = []
        for r in results:
            sources.append({
                "id": r.memory.id,
                "content": r.memory.content[:500],
                "score": round(r.composite_score, 4),
                "type": r.memory.memory_type.value,
                "tags": r.memory.tags,
            })

        synthesis_text = None
        if self.enrichment.has_llm:
            memory_texts = []
            for r in results[:15]:
                memory_texts.append(f"- [{r.memory.memory_type.value}] {r.memory.content[:300]}")

            prompt = (
                f"Synthesize the following memories about '{topic}' into a coherent, "
                f"well-organized summary. Group related information together. "
                f"Highlight key facts, patterns, and any contradictions.\n\n"
                f"Memories:\n" + "\n".join(memory_texts) + "\n\n"
                f"Synthesis:"
            )
            try:
                synthesis_text = self.enrichment.generate(prompt, max_tokens=800)
            except Exception as e:
                print(f"  ! Synthesis LLM failed ({self.enrichment.backend_name}): {e}")
                synthesis_text = None

        if not synthesis_text:
            grouped = {"semantic": [], "episodic": [], "procedural": []}
            for r in results:
                mtype = r.memory.memory_type.value
                grouped.get(mtype, grouped["semantic"]).append(r.memory.content[:200])

            parts = []
            if grouped["semantic"]:
                parts.append("**Facts & Knowledge:**\n" + "\n".join(f"- {c}" for c in grouped["semantic"][:10]))
            if grouped["procedural"]:
                parts.append("**Procedures & Patterns:**\n" + "\n".join(f"- {c}" for c in grouped["procedural"][:5]))
            if grouped["episodic"]:
                parts.append("**Related Interactions:**\n" + "\n".join(f"- {c}" for c in grouped["episodic"][:5]))

            synthesis_text = "\n\n".join(parts) if parts else "Could not synthesize."

        stored_id = None
        if store_result and synthesis_text:
            synth_memory = self.remember(
                content=f"[Synthesis: {topic}] {synthesis_text[:2000]}",
                memory_type=MemoryType.SEMANTIC,
                importance=0.8,
                tags=["synthesis", "auto-generated"],
                metadata={
                    "synthesis_topic": topic,
                    "source_count": len(results),
                    "source_ids": [r.memory.id for r in results[:10]],
                },
                namespace=namespace,
            )
            stored_id = synth_memory.id

        return {
            "topic": topic,
            "synthesis": synthesis_text,
            "source_count": len(results),
            "sources": sources,
            "stored_memory_id": stored_id,
        }

    _QUERY_EXPAND_PROMPT = (
        "Given this search query, generate 2-3 alternative phrasings that "
        "would match the same information in a memory database. Return ONLY "
        "the alternatives, one per line. No numbering, no explanation.\n\n"
        "Query: {query}\n\nAlternatives:"
    )

    def _expand_query(self, query: str) -> List[str]:
        """
        Generate alternative query phrasings to broaden recall.

        Uses the enrichment LLM if available, otherwise falls back to
        simple synonym / rewrite heuristics.
        """
        alternatives = [query]

        if self.enrichment.has_llm:
            try:
                prompt = self._QUERY_EXPAND_PROMPT.format(query=query)
                raw = self.enrichment.generate(prompt, max_tokens=150)
                for line in raw.strip().split("\n"):
                    line = line.strip().lstrip("0123456789.-) ")
                    if line and len(line) > 5 and line != query:
                        alternatives.append(line)
            except Exception:
                pass

        if len(alternatives) < 2:
            alternatives.extend(self._heuristic_expand(query))

        return alternatives[:4]

    @staticmethod
    def _heuristic_expand(query: str) -> List[str]:
        """Simple keyword-based query rewrites as LLM fallback."""
        expansions = []
        q = query.lower()

        synonym_map = {
            "prefer": "like",
            "like": "prefer",
            "use": "work with",
            "work with": "use",
            "favorite": "preferred",
            "preferred": "favorite",
            "how": "what way",
            "where": "location of",
            "when": "date of",
        }
        for old, new in synonym_map.items():
            if old in q:
                expansions.append(q.replace(old, new, 1))
                break

        words = query.split()
        if len(words) >= 3:
            expansions.append(" ".join(words[1:]))

        return expansions

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def _idf_lexical_score(
        self,
        query_tokens: List[str],
        memory_tokens: set,
        doc_freq: Dict[str, int],
        total_docs: int,
    ) -> float:
        if not query_tokens or not memory_tokens or total_docs <= 0:
            return 0.0

        def token_idf(token: str) -> float:
            d = doc_freq.get(token, 0)
            return math.log((1 + total_docs) / (1 + d)) + 1.0

        q_unique = set(query_tokens)
        denom = sum(token_idf(t) for t in q_unique)
        if denom <= 0:
            return 0.0
        numer = sum(token_idf(t) for t in q_unique if t in memory_tokens)
        return numer / denom

    def _entity_overlap_boost(
        self,
        query_tokens: List[str],
        memory_tokens: set,
        doc_freq: Dict[str, int],
    ) -> float:
        informative = []
        for t in query_tokens:
            if len(t) >= 4 or any(ch.isdigit() for ch in t):
                if doc_freq.get(t, 0) <= max(2, int(0.1 * max(1, len(doc_freq)))):
                    informative.append(t)
        if not informative:
            informative = [t for t in query_tokens if len(t) >= 4]
        if not informative:
            return 0.0
        matched = sum(1 for t in informative if t in memory_tokens)
        return min(1.0, matched / max(1, len(informative)))

    def _phrase_match_score(self, query: str, content: str) -> float:
        query_words = query.lower().split()
        content_lower = content.lower()

        if len(query_words) < 2:
            return 0.0

        phrases: List[str] = []
        for n in range(2, min(5, len(query_words) + 1)):
            for start in range(len(query_words) - n + 1):
                phrase = " ".join(query_words[start : start + n])
                if len(phrase) > 4:
                    phrases.append(phrase)

        if not phrases:
            return 0.0

        matched = sum(1 for p in phrases if p in content_lower)
        return min(1.0, matched / len(phrases))

    def _identifier_match_boost(self, query_tokens: List[str], memory_tokens: set) -> float:
        id_like = [
            t for t in query_tokens
            if any(ch.isdigit() for ch in t) or re.fullmatch(r"[a-z]+[_-]?\d+", t)
        ]
        if not id_like:
            return 0.0
        return 1.0 if any(t in memory_tokens for t in id_like) else 0.0

    # ── Contradiction penalty cache ──

    def _invalidate_contradiction_cache(self):
        with self._lock:
            self._contradiction_cache_dirty = True

    def _get_contradiction_penalties(self) -> Dict[str, float]:
        with self._lock:
            if not self._contradiction_cache_dirty and self._contradiction_cache is not None:
                return self._contradiction_cache

        penalties = self._build_contradiction_penalty_map()
        with self._lock:
            self._contradiction_cache = penalties
            self._contradiction_cache_dirty = False
        return penalties

    def _build_contradiction_penalty_map(self) -> Dict[str, float]:
        """
        Penalize memories contradicted by newer active memories.

        Uses targeted query to load only CONTRADICTS links instead of
        the entire link table — O(contradictions) not O(all links).
        """
        penalties: Dict[str, float] = {}

        if hasattr(self.storage, "get_links_by_type"):
            contra_links = self.storage.get_links_by_type(LinkType.CONTRADICTS)
        else:
            links = self.storage.get_all_links()
            contra_links = [l for l in links if l.link_type == LinkType.CONTRADICTS]

        if not contra_links:
            return penalties

        involved_ids = set()
        for link in contra_links:
            involved_ids.add(link.source_id)
            involved_ids.add(link.target_id)

        memory_map = {
            m.id: m
            for m in self.storage.get_memories_by_ids(list(involved_ids))
            if m.is_active
        }

        for link in contra_links:
            src = memory_map.get(link.source_id)
            tgt = memory_map.get(link.target_id)
            if not src or not tgt:
                continue
            if src.created_at >= tgt.created_at:
                penalties[tgt.id] = min(penalties.get(tgt.id, 1.0), 0.55)
            else:
                penalties[src.id] = min(penalties.get(src.id, 1.0), 0.55)
        return penalties

    def record_episode(
        self,
        user_message: str,
        assistant_response: str,
        feedback: str = None,
        importance: float = 0.5,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        namespace: str = None,
    ) -> Memory:
        """
        Record a complete interaction episode.

        Designed for AI assistants to log conversations. Creates a
        CAUSAL link to the previous episode, forming a temporal chain
        that enables "what happened before/after X?" queries.

        Feedback adjusts importance:
        - "positive": Slightly boosted
        - "negative": Boosted more
        - "correction": Boosted most
        """
        namespace = namespace or self.default_namespace

        self.add_to_working_memory(user_message, role="user")
        self.add_to_working_memory(assistant_response, role="assistant")

        content = f"User: {user_message}\nAssistant: {assistant_response}"

        if feedback == "positive":
            importance = min(1.0, importance + 0.2)
        elif feedback == "negative":
            importance = min(1.0, importance + 0.3)
        elif feedback == "correction":
            importance = min(1.0, importance + 0.4)

        meta = metadata or {}
        meta["feedback"] = feedback
        meta["user_message"] = user_message
        meta["assistant_response"] = assistant_response

        episode = self.remember(
            content=content,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            tags=tags or [],
            metadata=meta,
            namespace=namespace,
        )

        if self._last_episode_id is not None:
            causal_link = MemoryLink(
                source_id=self._last_episode_id,
                target_id=episode.id,
                link_type=LinkType.CAUSAL,
                weight=0.7,
            )
            self.storage.store_link(causal_link)

        self._last_episode_id = episode.id
        return episode

    def learn_procedure(
        self,
        name: str,
        description: str,
        steps: List[str] = None,
        tags: List[str] = None,
        namespace: str = None,
    ) -> Memory:
        """Store a procedural memory (a learned workflow or pattern)."""
        namespace = namespace or self.default_namespace

        content = f"Procedure: {name}\n{description}"
        if steps:
            content += "\nSteps:\n" + "\n".join(
                f"  {i+1}. {s}" for i, s in enumerate(steps)
            )

        return self.remember(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            importance=0.7,
            tags=(tags or []) + ["procedure", name.lower().replace(" ", "_")],
            metadata={"name": name, "steps": steps or []},
            namespace=namespace,
        )

    def reinforce_memory(self, memory_id: str, boost: float = 0.2) -> Optional[Memory]:
        """Manually reinforce a specific memory."""
        memory = self.storage.get_memory(memory_id)
        if memory is None:
            return None

        memory = self.decay_engine.reinforce(memory, boost=boost)
        self.storage.update_memory(memory)
        return memory

    def correct_memory(
        self,
        memory_id: str,
        new_content: str,
        reason: str = "",
    ) -> Optional[Memory]:
        """
        Correct/update a memory using temporal versioning.

        The old version is preserved with valid_until set and strength
        reduced (not deleted). A superseded link connects old → new.
        """
        old_memory = self.storage.get_memory(memory_id)
        if old_memory is None:
            return None

        # Snapshot before correction
        self._snapshot_version(old_memory, reason=reason or "corrected")

        now = time.time()

        new_memory = self.remember(
            content=new_content,
            memory_type=old_memory.memory_type,
            importance=min(1.0, old_memory.importance + 0.3),
            tags=old_memory.tags + ["corrected"],
            metadata={
                **{k: v for k, v in old_memory.metadata.items()
                   if k not in ("superseded_by", "valid_until", "enriched_content")},
                "corrected_from": memory_id,
                "correction_reason": reason,
                "correction_time": now,
                "valid_from": now,
            },
            namespace=old_memory.namespace,
        )

        # Set high confidence on corrections (user-verified)
        new_memory.confidence = 0.9
        new_memory.epistemic_status = "verified"
        self.storage.update_memory(new_memory)

        old_memory.metadata["superseded_by"] = new_memory.id
        old_memory.metadata["valid_until"] = now
        if not old_memory.metadata.get("valid_from"):
            old_memory.metadata["valid_from"] = old_memory.created_at
        old_memory.strength = max(0.2, old_memory.strength * 0.5)
        # Epistemic: the corrected-away memory is now contradicted
        old_memory.epistemic_status = "contradicted"
        old_memory.confidence = round(min(old_memory.confidence, 0.3), 3)
        self.storage.update_memory(old_memory)

        link = MemoryLink(
            source_id=memory_id,
            target_id=new_memory.id,
            link_type=LinkType.SUPERSEDED,
            weight=1.0,
        )
        self.storage.store_link(link)

        # Log provenance
        try:
            self.storage.store_provenance(ProvenanceEntry(
                memory_id=new_memory.id,
                parent_memory_ids=[memory_id],
                operation="corrected",
                reason=reason,
            ))
        except Exception:
            pass

        self._invalidate_contradiction_cache()
        return new_memory

    def forget_memory(
        self, memory_id: str, hard: bool = False
    ) -> bool:
        """
        Forget a specific memory.

        hard=False: soft-delete (deactivate) — reversible
        hard=True:  permanent delete — irreversible
        """
        memory = self.storage.get_memory(memory_id)
        if memory is None:
            return False
        self.memory_index.remove(memory_id)
        self.entity_graph.remove_memory(memory_id)
        self.storage.forget_memory(memory_id, hard=hard)
        self.memory_index.save()
        self._invalidate_contradiction_cache()
        return True

    # ══════════════════════════════════════════════
    #  WORKING MEMORY
    # ══════════════════════════════════════════════

    def add_to_working_memory(
        self,
        content: str,
        role: str = "user",
        metadata: Dict[str, Any] = None,
    ):
        item = WorkingMemoryItem(
            content=content,
            role=role,
            metadata=metadata or {},
        )
        self.storage.store_working_item(item)
        self.storage.trim_working_memory(keep_last=self.working_memory_size)

    def get_working_context(self) -> List[WorkingMemoryItem]:
        return self.storage.get_working_memory(limit=self.working_memory_size)

    def clear_working_memory(self):
        self.storage.clear_working_memory()

    # ══════════════════════════════════════════════
    #  MAINTENANCE
    # ══════════════════════════════════════════════

    def consolidate(self, multi_level: bool = True) -> Dict:
        """
        Manually trigger memory consolidation and index new memories.
        When multi_level=True, runs L0->L1->L2->L3 consolidation.
        """
        if multi_level:
            stats = self.consolidation.run_multi_level_consolidation()
        else:
            stats = self.consolidation.run_consolidation()
        self._index_unindexed_memories()
        return stats

    def _index_unindexed_memories(self):
        """Add any memories missing from FAISS/entity graph (e.g. after consolidation)."""
        indexed_ids = set(self.memory_index._id_to_idx.keys()) if hasattr(self.memory_index, '_id_to_idx') else set()
        for mem, emb in self.storage.get_memories_with_embeddings():
            if mem.id not in indexed_ids:
                self.memory_index.add(mem.id, mem.embedding)
                self.entity_graph.index_memory(mem.id, mem.content)
        self.memory_index.save()

    def run_decay(self) -> Dict:
        """Manually run decay on all memories."""
        return self.decay_engine.apply_decay_to_all(self.storage)

    def maintenance(self) -> Dict[str, Any]:
        """
        Run all maintenance tasks: consolidation, decay, pruning,
        integrity repair. Call periodically or before shutdown.
        """
        results: Dict[str, Any] = {}

        # Consolidation
        try:
            results["consolidation"] = self.consolidate(multi_level=True)
        except Exception as e:
            results["consolidation"] = {"error": str(e)}

        # Decay
        try:
            results["decay"] = self.run_decay()
        except Exception as e:
            results["decay"] = {"error": str(e)}

        # Prune reasoning
        try:
            pruned = self.storage.prune_reasoning_conclusions(max_per_type=200)
            results["reasoning_pruned"] = pruned
        except Exception as e:
            results["reasoning_pruned"] = {"error": str(e)}

        # Clean reasoning queue
        try:
            cleaned = self.storage.prune_processed_reasoning_queue(keep_hours=48)
            results["queue_cleaned"] = cleaned
        except Exception as e:
            results["queue_cleaned"] = {"error": str(e)}

        # Integrity repair
        try:
            results["repair"] = self.storage.repair()
        except Exception as e:
            results["repair"] = {"error": str(e)}

        # FAISS sync
        try:
            db_count = self.storage.count_active_memories_with_embeddings()
            idx_count = self.memory_index.size
            if db_count != idx_count:
                self._rebuild_faiss_indices()
                self.memory_index.save()
                self.passage_index.save()
                results["faiss_rebuilt"] = True
            else:
                results["faiss_rebuilt"] = False
        except Exception as e:
            results["faiss_rebuilt"] = {"error": str(e)}

        # Lint / self-audit
        try:
            results["lint"] = self.linter.lint()
        except Exception as e:
            results["lint"] = {"error": str(e)}

        # Storage stats
        results["storage_stats"] = self.storage.get_storage_stats()

        return results

    # ══════════════════════════════════════════════
    #  KNOWLEDGE PAGES / LINT / PROVENANCE / VERSIONS
    # ══════════════════════════════════════════════

    def lint(self) -> Dict[str, Any]:
        """Run all memory health checks. Returns structured report."""
        return self.linter.lint()

    def get_knowledge_pages(self, page_type: str = None) -> list:
        """Get all knowledge pages."""
        return [p.model_dump() for p in self.knowledge_pages.get_all_pages(page_type)]

    def get_knowledge_page(self, page_id: str) -> Optional[Dict[str, Any]]:
        """Get a single knowledge page by ID."""
        page = self.knowledge_pages.get_page(page_id)
        return page.model_dump() if page else None

    def rebuild_knowledge_pages(self) -> Dict[str, Any]:
        """Rebuild all knowledge pages from scratch."""
        return self.knowledge_pages.rebuild_all_pages()

    def get_provenance(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get provenance chain for a memory."""
        entries = self.storage.get_provenance_chain(memory_id)
        return [e.model_dump() for e in entries]

    def get_version_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get version history for a memory."""
        versions = self.storage.get_version_history(memory_id)
        return [v.model_dump() for v in versions]

    def compound_recall(
        self,
        query: str,
        top_k: int = 10,
        store_result: bool = True,
    ) -> Dict[str, Any]:
        """
        Compounding recall: synthesize an answer from memories, then
        store the synthesis as a new higher-level memory.
        """
        result = self.synthesize(topic=query, top_k=top_k, store_result=store_result)

        if store_result and result.get("memory_id"):
            # Log provenance for the compound memory
            source_ids = result.get("source_memory_ids", [])
            try:
                self.storage.store_provenance(ProvenanceEntry(
                    memory_id=result["memory_id"],
                    parent_memory_ids=source_ids,
                    operation="derived",
                    reason=f"compound recall: {query[:100]}",
                ))
            except Exception:
                pass

        return result

    def _compute_initial_confidence(
        self, content: str, meta: Dict[str, Any], memory_type: MemoryType,
    ) -> tuple:
        """Compute initial confidence and epistemic status for a new memory."""
        confidence = 0.5
        status = "inferred"

        if meta.get("corrected_from") or meta.get("correction_reason"):
            confidence = 0.9
            status = "verified"
        elif meta.get("source_url") or meta.get("source_file"):
            confidence = 0.8
            status = "inferred"
        elif memory_type == MemoryType.SEMANTIC:
            confidence = 0.7
            status = "inferred"

        return confidence, status

    def _snapshot_version(self, memory: Memory, reason: str = ""):
        """Create a version snapshot of a memory before modifying it."""
        try:
            self.storage.store_memory_version(MemoryVersion(
                memory_id=memory.id,
                content=memory.content,
                strength=memory.strength,
                importance=memory.importance,
                confidence=memory.confidence,
                change_reason=reason,
            ))
        except Exception:
            pass

    def get_stats(self, namespace: str = None) -> MemoryStats:
        """Get statistics about the memory system's current state."""
        counts = self.storage.get_counts(namespace=namespace)

        return MemoryStats(
            total_memories=counts.get("total", 0),
            episodic_count=counts.get("episodic", 0),
            semantic_count=counts.get("semantic", 0),
            procedural_count=counts.get("procedural", 0),
            total_links=counts.get("links", 0),
            working_memory_size=counts.get("working", 0),
            avg_strength=self.storage.get_avg_strength(),
            avg_importance=self.storage.get_avg_importance(),
            consolidation_count=self.storage.get_consolidation_count(),
            oldest_memory_age_hours=self.storage.get_oldest_memory_age_hours(),
            most_accessed_memory_id=self.storage.get_most_accessed_memory_id(),
        )

    # ══════════════════════════════════════════════
    #  LLM CONTEXT FORMATTING
    # ══════════════════════════════════════════════

    def format_for_llm(
        self,
        query: str,
        token_budget: int = 4000,
        include_working_memory: bool = True,
        top_k: int = 15,
        namespace: str = None,
        format_mode: str = "standard",
    ) -> str:
        """
        Recall relevant memories and format them into a ready-to-use
        LLM system/context prompt that fits within a token budget.

        format_mode:
            "standard" -- existing behavior
            "reasoned" -- peer card at top, then reasoning conclusions,
                          then raw memories for remaining budget

        Returns a string that can be injected directly into an LLM's
        system message or context window. Handles:
        - Recall + ranking
        - Deduplication of overlapping content
        - Token budget packing (greedy, highest-relevance first)
        - Working memory (recent conversation) integration
        - Type-aware formatting
        """
        chars_per_token = 4
        char_budget = token_budget * chars_per_token

        sections: list = []
        used_chars = 0

        if format_mode == "reasoned" and self.reasoning.enabled:
            reasoned_block = self._format_reasoned_section(char_budget // 3)
            if reasoned_block:
                sections.append(reasoned_block)
                used_chars += len(reasoned_block)

        if include_working_memory:
            working = self.get_working_context()
            if working:
                wm_lines = []
                for item in working[-10:]:
                    prefix = "User" if item.role == "user" else "Assistant"
                    wm_lines.append(f"  {prefix}: {item.content[:200]}")
                wm_block = "## Recent Conversation\n" + "\n".join(wm_lines)
                if len(wm_block) < char_budget * 0.3:
                    sections.append(wm_block)
                    used_chars += len(wm_block)

        results = self.recall(
            query=query,
            top_k=top_k,
            namespace=namespace,
            include_associations=False,
        )

        if not results:
            if sections:
                return "\n\n".join(sections)
            return ""

        import time as _time
        now = _time.time()

        # Phase 1B: Load source chunks for recalled memories
        result_ids = [r.memory.id for r in results]
        source_chunks = self.storage.get_source_chunks(result_ids)

        seen_content = set()
        categorized: Dict[str, list] = {
            "semantic": [],
            "procedural": [],
            "episodic": [],
        }
        has_recent_updates = False

        for r in results:
            content = r.memory.content.strip()
            content = re.sub(r"(\S)\n{2,}(\S)", r"\1 \2", content)
            content_key = content[:100].lower()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            age_seconds = now - r.memory.created_at
            is_recent = age_seconds < 3600  # created in last hour
            is_updated = any(
                t in ("updated", "test-update", "correction")
                for t in (r.memory.tags or [])
            )
            is_llm_extracted = (r.memory.metadata or {}).get("llm_extracted", False)
            from_document = bool((r.memory.metadata or {}).get("source_file"))

            if is_recent and not from_document:
                has_recent_updates = True

            # Phase 1B: Attach source chunk for context
            source_chunk = source_chunks.get(r.memory.id)

            # Phase 1E: Historical tag
            is_historical = not r.memory.is_current

            mtype = r.memory.memory_type.value
            entry = {
                "content": content,
                "score": r.composite_score,
                "importance": r.memory.importance,
                "tags": r.memory.tags,
                "is_recent": is_recent,
                "is_updated": is_updated,
                "from_document": from_document,
                "age_seconds": age_seconds,
                "source_chunk": source_chunk,
                "is_historical": is_historical,
                "abstraction_level": r.memory.abstraction_level,
            }
            categorized.get(mtype, categorized["episodic"]).append(entry)

        # Sort each category: recent updates first, then by score
        for mtype in categorized:
            categorized[mtype].sort(
                key=lambda e: (
                    -(1 if e["is_recent"] and not e["from_document"] else 0),
                    -(1 if e["is_updated"] else 0),
                    -e["score"],
                )
            )

        if has_recent_updates:
            preamble = (
                "**IMPORTANT: Some memories below are recent updates. "
                "When information conflicts, ALWAYS prefer memories marked "
                "[RECENT UPDATE] over older document excerpts.**"
            )
            sections.append(preamble)
            used_chars += len(preamble)

        type_labels = {
            "semantic": "Known Facts & Knowledge",
            "procedural": "Learned Procedures & Workflows",
            "episodic": "Relevant Past Interactions",
        }

        for mtype in ["semantic", "procedural", "episodic"]:
            entries = categorized[mtype]
            if not entries:
                continue

            label = type_labels[mtype]
            block_lines = [f"## {label}"]

            for entry in entries:
                content = entry["content"]
                remaining = char_budget - used_chars - 10
                if remaining <= 0:
                    break
                if len(content) > remaining:
                    content = content[:remaining] + "…"

                prefix = ""
                if entry.get("is_historical"):
                    prefix = "[HISTORICAL] "
                elif entry["is_recent"] and not entry["from_document"]:
                    prefix = "[RECENT UPDATE] "
                elif entry["is_updated"]:
                    prefix = "[UPDATED] "

                # Phase 2A: Level prefix for higher abstractions
                level = entry.get("abstraction_level", 0)
                if level >= 3:
                    prefix += "[CORE IDENTITY] "
                elif level == 2:
                    prefix += "[TRAIT] "
                elif level == 1:
                    prefix += "[PATTERN] "

                line = f"- {prefix}{content}"
                if entry["tags"]:
                    line += f"  [{', '.join(entry['tags'][:3])}]"

                # Phase 1B: Inject source chunk for deeper context
                src = entry.get("source_chunk")
                if src and len(src) < remaining - len(line) - 50:
                    truncated_src = src[:300] + "…" if len(src) > 300 else src
                    line += f"\n  Source: {truncated_src}"

                block_lines.append(line)
                used_chars += len(line) + 1

            if len(block_lines) > 1:
                sections.append("\n".join(block_lines))

        return "\n\n".join(sections)

    def _format_reasoned_section(self, char_budget: int) -> str:
        """Build a reasoning-first context block for format_mode='reasoned'."""
        lines = []
        used = 0

        peer_card = self.reasoning._get_existing_peer_card()
        if peer_card:
            card_line = f"**User Profile:** {peer_card.content[:char_budget // 3]}"
            lines.append(card_line)
            used += len(card_line)

        conclusions = self.storage.get_all_memories(
            memory_type=MemoryType.SEMANTIC,
            tags=["reasoning"],
            active_only=True,
        )
        conclusions = [
            m for m in conclusions
            if (m.metadata or {}).get("reasoning_type") in ("deductive", "inductive", "abductive")
        ]
        conclusions.sort(key=lambda m: m.abstraction_level, reverse=True)

        if conclusions:
            lines.append("## Reasoning Conclusions")
            for m in conclusions:
                if used >= char_budget:
                    break
                rtype = (m.metadata or {}).get("reasoning_type", "")
                tag = rtype.upper() if rtype else ""
                entry = f"- [{tag}] {m.content[:300]}"
                lines.append(entry)
                used += len(entry)

        return "\n".join(lines) if lines else ""

    # ══════════════════════════════════════════════
    #  FAISS INDEX MANAGEMENT
    # ══════════════════════════════════════════════

    def _get_faiss_candidates(
        self, embedding: List[float], k: int = GRAPH_CANDIDATE_K,
    ) -> List[Tuple[Memory, np.ndarray]]:
        """
        Use FAISS to retrieve the top-k most similar memories and return
        them as (Memory, embedding_ndarray) pairs ready for graph operations.
        """
        hits = self.memory_index.search(embedding, k=k)
        if not hits:
            return []

        hit_ids = [mid for mid, _ in hits]
        loaded = self.storage.get_memories_by_ids(hit_ids)
        mem_map = {m.id: m for m in loaded if m.is_active and m.embedding}

        results: List[Tuple[Memory, np.ndarray]] = []
        for mid, _ in hits:
            mem = mem_map.get(mid)
            if mem and mem.embedding:
                results.append((mem, np.array(mem.embedding, dtype=np.float32)))

        return results

    def _startup_health_check(self):
        """Run integrity checks and auto-repair on startup."""
        report = self.storage.integrity_check()
        if not report["sqlite_ok"]:
            print("  ⚠ SQLite integrity check FAILED — database may be corrupted")
            print("    Attempting repair via VACUUM...")
            try:
                self.storage._conn().execute("VACUUM")
                print("    VACUUM completed — re-checking...")
                report = self.storage.integrity_check()
                if report["sqlite_ok"]:
                    print("    ✓ Database integrity restored")
                else:
                    print("    ✗ Database still corrupt — consider restoring from backup")
            except Exception as e:
                print(f"    ✗ VACUUM failed: {e}")

        stats = self.storage.get_storage_stats()
        print(f"  + Storage: {stats['active_memories']} active, "
              f"{stats['inactive_memories']} inactive, "
              f"{stats['total_links']} links")
        if stats.get("db_size_mb", 0) > 100:
            print(f"  ⚠ Database size is {stats['db_size_mb']}MB — consider pruning old data")

        needs_repair = (
            report["orphaned_links"] > 0
            or report["orphaned_passages"] > 0
            or report["dead_peer_cards"] > 5
        )
        if needs_repair:
            print(f"  ⚡ Auto-repairing: {report['orphaned_links']} orphaned links, "
                  f"{report['orphaned_passages']} orphaned passages, "
                  f"{report['dead_peer_cards']} dead peer cards...")
            repaired = self.storage.repair()
            for k, v in repaired.items():
                if v > 0:
                    print(f"    ✓ Cleaned {v} {k.replace('_', ' ')}")

        # Cap reasoning conclusions to prevent unbounded growth
        pruned = self.storage.prune_reasoning_conclusions(max_per_type=200)
        if pruned > 0:
            print(f"  ⚡ Pruned {pruned} old reasoning conclusions")

        # Clean old processed reasoning queue entries
        cleaned = self.storage.prune_processed_reasoning_queue(keep_hours=48)
        if cleaned > 0:
            print(f"  ⚡ Cleaned {cleaned} old reasoning queue entries")

    def _ensure_faiss_indices(self):
        """
        Load FAISS indices from disk, or rebuild them from SQLite if the
        saved index is missing / stale / out-of-sync with DB.
        """
        mem_loaded = self.memory_index.load()
        pass_loaded = self.passage_index.load()

        if mem_loaded and pass_loaded:
            db_count = self.storage.count_active_memories_with_embeddings()
            idx_count = self.memory_index.size
            if db_count == idx_count:
                return
            print(f"  ⚡ FAISS index stale ({idx_count} indexed vs {db_count} in DB), rebuilding...")
        else:
            print("  ⚡ Building FAISS indices from database...")

        self._rebuild_faiss_indices()
        self.memory_index.save()
        self.passage_index.save()

    def _rebuild_faiss_indices(self):
        """Rebuild both FAISS indices from SQLite data."""
        mem_embs: Dict[str, list] = {}
        for memory, emb in self.storage.get_memories_with_embeddings():
            mem_embs[memory.id] = emb.tolist()
        self.memory_index.build_from_dict(mem_embs)

        pass_embs: Dict[str, list] = {}
        for parent_mid, emb in self.storage.get_all_passage_embeddings():
            idx = sum(1 for k in pass_embs if k.startswith(parent_mid + "::"))
            pkey = f"{parent_mid}::p{idx}"
            pass_embs[pkey] = emb.tolist()
        self.passage_index.build_from_dict(pass_embs)

    # ══════════════════════════════════════════════
    #  EMBEDDING MIGRATION
    # ══════════════════════════════════════════════

    def _check_embedding_migration(self):
        """
        Detect if the embedding model has changed since the database was last
        used and automatically re-embed all existing memories.
        """
        stored_model = self.storage.get_meta("embedding_model")
        stored_dim = self.storage.get_meta("embedding_dimension")
        current_model = self.embeddings.model_name
        current_dim = str(self.embeddings.dimension)

        needs_reembed = False
        if self.storage.has_memories():
            sample_dim = self.storage.get_sample_embedding_dimension()
            if sample_dim is not None and sample_dim != self.embeddings.dimension:
                print(f"  ⚡ Embedding dimension mismatch: {sample_dim}d → {current_dim}d")
                needs_reembed = True
            elif stored_model is not None and stored_model != current_model:
                print(f"  ⚡ Embedding model changed: {stored_model} → {current_model}")
                needs_reembed = True

            if needs_reembed:
                print(f"     Re-embedding all existing memories...")
                self._reembed_all_memories()
                print(f"  ✓ Re-embedding complete!")
        elif stored_model == current_model and stored_dim == current_dim:
            return

        self.storage.set_meta("embedding_model", current_model)
        self.storage.set_meta("embedding_dimension", current_dim)

    def _reembed_all_memories(self):
        """Re-embed every active memory, recreate passages, and rebuild FAISS."""
        self.storage.clear_all_embeddings()
        self.memory_index.clear()
        self.passage_index.clear()
        memories = self.storage.get_all_memories(active_only=True)

        for i, memory in enumerate(memories):
            memory.embedding = self.embeddings.embed(memory.content)
            self.storage.update_memory(memory)
            self._build_passages(memory)

            if (i + 1) % 50 == 0:
                print(f"     ... re-embedded {i + 1}/{len(memories)} memories")

        self._rebuild_faiss_indices()
        self.memory_index.save()
        self.passage_index.save()

    # ══════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ══════════════════════════════════════════════

    def _build_passages(self, memory: Memory):
        """
        Create passage-level embeddings for long content so retrieval
        can match on *any* section, not just the first ~256 tokens.

        Shared by _store_single_memory and _reembed_all_memories.
        """
        if len(memory.content) <= PASSAGE_CHAR_THRESHOLD:
            return

        all_passages: list = []

        passage_pairs = self.embeddings.embed_passages(memory.content)
        if passage_pairs:
            for idx, (chunk, emb) in enumerate(passage_pairs):
                all_passages.append({
                    "chunk_index": idx,
                    "content_preview": chunk[:500],
                    "embedding": emb,
                })

        key_sents = self._extract_key_sentences(memory.content)
        if key_sents:
            sent_embs = self.embeddings.embed_batch(key_sents)
            base_idx = len(all_passages)
            for si, (sent, emb) in enumerate(zip(key_sents, sent_embs)):
                all_passages.append({
                    "chunk_index": base_idx + si,
                    "content_preview": sent[:500],
                    "embedding": emb,
                })

        if all_passages:
            self.storage.store_passages(memory.id, all_passages)
            for p in all_passages:
                passage_key = f"{memory.id}::p{p['chunk_index']}"
                self.passage_index.add(passage_key, p["embedding"])

    def _extract_key_sentences(
        self, content: str, max_sentences: int = 15
    ) -> List[str]:
        """
        Extract fact-bearing sentences from content for fine-grained retrieval.
        Prioritises sentences containing names, numbers, dates, or opinion keywords.
        """
        raw = re.split(r"[.!?\n]+", content)
        sentences = [s.strip() for s in raw if len(s.strip()) > 25]

        if len(sentences) <= 3:
            return []

        def _fact_score(sent: str) -> float:
            score = 0.0
            if re.search(r"\d", sent):
                score += 1.0
            caps = re.findall(r"\b[A-Z][a-z]+\b", sent)
            score += min(2.0, len(caps) * 0.5)
            if re.search(
                r"\b(name|born|live|work|prefer|favorite|married|child|"
                r"moved|started|bought|enjoy|visit|love|hate|play|study)\b",
                sent.lower(),
            ):
                score += 1.0
            if len(sent) > 50:
                score += 0.5
            return score

        scored = [(s, _fact_score(s)) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, sc in scored[:max_sentences] if sc > 0]

    def _detect_importance(self, content: str, base_importance: float) -> float:
        """Auto-detect importance from content signals."""
        content_lower = content.lower()
        boost = 0.0

        high_signals = [
            "important", "critical", "remember this", "never forget",
            "always remember", "key point", "essential", "crucial",
            "must remember", "don't forget",
        ]
        for signal in high_signals:
            if signal in content_lower:
                boost += 0.3
                break

        correction_signals = [
            "actually", "correction", "wrong", "mistake",
            "not correct", "update", "changed", "no longer",
        ]
        for signal in correction_signals:
            if signal in content_lower:
                boost += 0.2
                break

        preference_signals = [
            "prefer", "like", "dislike", "hate", "love",
            "favorite", "always use", "never use",
        ]
        for signal in preference_signals:
            if signal in content_lower:
                boost += 0.15
                break

        return min(1.0, base_importance + boost)

    def _increment_operations(self):
        """Track operations and trigger auto-maintenance when thresholds are hit."""
        if self._shutting_down:
            return
        with self._lock:
            self._operation_count += 1

            if (
                self.auto_consolidate
                and self._operation_count % self.consolidation_interval == 0
            ):
                self._bg_pool.submit(self._safe_consolidate)

            if self._operation_count % self.decay_interval == 0:
                self._bg_pool.submit(self._safe_decay)

            if self.active_memory.should_run():
                self._bg_pool.submit(self._safe_active_management)

            self.reasoning.maybe_dream()

    def _safe_active_management(self):
        """Thread-safe wrapper for background active memory management."""
        try:
            self.active_memory.run_all(manager=self)
        except Exception as e:
            import sys
            print(f"  ! Background active memory error: {e}", file=sys.stderr)

    def _safe_consolidate(self):
        """Thread-safe wrapper for background consolidation."""
        try:
            stats = self.consolidation.run_consolidation()
            if stats.get("semantic_memories_created", 0) > 0:
                self._index_unindexed_memories()
        except Exception as e:
            import sys
            print(f"  ! Background consolidation error: {e}", file=sys.stderr)

    def _safe_decay(self):
        """Thread-safe wrapper for background decay."""
        try:
            self.decay_engine.apply_decay_to_all(self.storage)
        except Exception as e:
            import sys
            print(f"  ! Background decay error: {e}", file=sys.stderr)

    # ══════════════════════════════════════════════
    #  PRODUCTION: HEALTH CHECK, SHUTDOWN, BACKUP
    # ══════════════════════════════════════════════

    def health_check(self) -> Dict[str, Any]:
        """
        Run a full health check. Returns a report suitable for
        monitoring dashboards and health-check endpoints.
        """
        report: Dict[str, Any] = {"status": "ok", "issues": []}

        # Database integrity
        try:
            db_report = self.storage.integrity_check()
            report["database"] = db_report
            if not db_report["sqlite_ok"]:
                report["status"] = "degraded"
                report["issues"].append("SQLite integrity check failed")
        except Exception as e:
            report["status"] = "error"
            report["issues"].append(f"Database check failed: {e}")

        # Storage stats
        try:
            report["storage"] = self.storage.get_storage_stats()
        except Exception:
            report["storage"] = {}

        # FAISS sync check
        try:
            db_count = self.storage.count_active_memories_with_embeddings()
            idx_count = self.memory_index.size
            report["faiss_synced"] = db_count == idx_count
            if not report["faiss_synced"]:
                report["status"] = "degraded"
                report["issues"].append(
                    f"FAISS out of sync: {idx_count} indexed vs {db_count} in DB"
                )
        except Exception:
            report["faiss_synced"] = False

        # Reasoning engine
        try:
            report["reasoning"] = self.reasoning.get_stats()
        except Exception:
            report["reasoning"] = {"enabled": False}

        # Thread pool
        report["background_pool"] = {
            "shutting_down": self._shutting_down,
        }

        return report

    def backup(self, dest_path: str = None) -> str:
        """Create a database backup. Returns the backup file path."""
        # Save FAISS indices first
        try:
            self.memory_index.save()
            self.passage_index.save()
        except Exception:
            pass
        return self.storage.backup(dest_path)

    def shutdown(self):
        """
        Graceful shutdown: flush indices, wait for background tasks,
        save state to disk. Called automatically via atexit.
        """
        if self._shutting_down:
            return
        self._shutting_down = True

        # Save FAISS indices
        try:
            self.memory_index.save()
            self.passage_index.save()
        except Exception:
            pass

        # Drain the thread pool
        try:
            self._bg_pool.shutdown(wait=True, cancel_futures=False)
        except TypeError:
            # Python < 3.9 doesn't support cancel_futures
            self._bg_pool.shutdown(wait=True)

        # Save entity graph
        try:
            self.entity_graph.save()
        except Exception:
            pass
