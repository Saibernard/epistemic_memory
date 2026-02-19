"""
Data models for the Memory Layer.

Defines all memory types, link types, and request/response structures.
These models mirror how human cognitive science categorizes memory:
  - Episodic: specific events and interactions
  - Semantic: extracted facts and knowledge  
  - Procedural: learned patterns and workflows
"""

import uuid
import time
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


# ============================================================
# CORE ENUMS
# ============================================================

class MemoryType(str, Enum):
    """Types of memory in the system, inspired by cognitive science."""
    EPISODIC = "episodic"       # Specific events/interactions (like autobiographical memory)
    SEMANTIC = "semantic"       # Extracted facts/knowledge (like encyclopedic knowledge)
    PROCEDURAL = "procedural"   # Learned patterns/workflows (like muscle memory)


class LinkType(str, Enum):
    """Types of associations between memories."""
    SIMILAR = "similar"           # Content similarity
    TEMPORAL = "temporal"         # Close in time
    CAUSAL = "causal"             # One caused/led to another
    DERIVED = "derived"           # Semantic memory derived from episodic
    CONTRADICTS = "contradicts"   # Conflicting information
    REINFORCES = "reinforces"     # Supporting/corroborating information
    SUPERSEDED = "superseded"     # This memory was replaced by a newer version
    ABSTRACTS = "abstracts"       # Higher-level abstraction of child memories


class EpistemicStatus(str, Enum):
    """How confident we are in a memory's truth value."""
    VERIFIED = "verified"         # User-confirmed or correction-sourced
    INFERRED = "inferred"         # Extracted or derived, not explicitly confirmed
    UNCERTAIN = "uncertain"       # Low corroboration or conflicting signals
    CONTRADICTED = "contradicted" # Superseded or lost a conflict resolution


# ============================================================
# CORE DATA MODELS
# ============================================================

class Memory(BaseModel):
    """
    The fundamental atom of the memory system.
    
    Every memory has:
    - Content (what is remembered)
    - Embedding (vector representation for semantic search)
    - Strength (decays over time, strengthens on recall)
    - Importance (salience - how critical is this memory)
    - Temporal metadata (when created, last accessed, how often)
    - Type classification (episodic, semantic, procedural)
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType
    content: str
    embedding: Optional[List[float]] = None
    
    # Temporal
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    
    # Dynamics
    strength: float = 1.0       # 0.0 = forgotten, 1.0 = vivid. Decays over time.
    importance: float = 0.5     # 0.0 = trivial, 1.0 = critical. Set by content signals.
    confidence: float = 0.5     # 0.0 = unreliable, 1.0 = highly trusted
    epistemic_status: str = "inferred"  # verified | inferred | uncertain | contradicted
    
    # Organization
    namespace: str = "default"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    source_episode_ids: List[str] = Field(default_factory=list)  # For derived memories
    is_active: bool = True

    # Temporal grounding (Phase 1A)
    document_date: Optional[float] = None       # when the source content was authored
    event_dates: Optional[List[Dict[str, Any]]] = None  # [{date, type, description}]

    # Versioning (Phase 1E)
    is_current: bool = True                     # False = superseded but still queryable

    # Abstraction hierarchy (Phase 2A)
    abstraction_level: int = 0                  # 0=facts, 1=patterns, 2=traits, 3=identity


class MemoryLink(BaseModel):
    """
    An association between two memories.
    
    Memories don't exist in isolation - they form a web of connections.
    When you recall one memory, associated memories are activated too.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    link_type: LinkType
    weight: float = 0.5         # Strength of association (0.0 - 1.0)
    created_at: float = Field(default_factory=time.time)


class WorkingMemoryItem(BaseModel):
    """
    Item in working memory - the current short-term context buffer.
    
    Like the human brain's working memory, this holds the immediate
    context of the current conversation/interaction.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    role: str = "user"          # 'user', 'assistant', 'system'
    created_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# RECALL RESULTS
# ============================================================

class RecallResult(BaseModel):
    """Result of a memory recall operation."""
    memory: Memory
    relevance_score: float      # How semantically relevant to the query
    effective_strength: float   # Current strength after applying decay
    confidence: float = 0.0     # Final confidence after hybrid ranking
    lexical_score: float = 0.0  # Token overlap score
    composite_score: float = 0.0
    associations: List[str] = Field(default_factory=list)  # IDs of associated memories


class MemoryStats(BaseModel):
    """Statistics about the memory system's health and state."""
    total_memories: int = 0
    episodic_count: int = 0
    semantic_count: int = 0
    procedural_count: int = 0
    total_links: int = 0
    working_memory_size: int = 0
    avg_strength: float = 0.0
    avg_importance: float = 0.0
    consolidation_count: int = 0
    oldest_memory_age_hours: float = 0.0
    most_accessed_memory_id: Optional[str] = None


# ============================================================
# API REQUEST MODELS
# ============================================================

class RememberRequest(BaseModel):
    """Request to store a new memory."""
    content: str = Field(..., min_length=1, max_length=10000)
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = Field(0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    namespace: str = "default"
    source: str = "api"


class RecallRequest(BaseModel):
    """Request to recall memories."""
    query: str = Field(..., min_length=1, max_length=2000)
    memory_types: Optional[List[MemoryType]] = None  # None = search all types
    tags: Optional[List[str]] = None
    top_k: int = Field(5, ge=1, le=200)
    min_strength: float = Field(0.1, ge=0.0, le=1.0)
    min_confidence: float = Field(0.05, ge=0.0, le=1.0)
    include_associations: bool = True
    namespace: str = "default"
    reasoning: bool = False
    include_history: bool = False
    diversity: bool = False


class EpisodeRequest(BaseModel):
    """Request to record an interaction episode."""
    user_message: str = Field(..., min_length=1, max_length=8000)
    assistant_response: str = Field(..., min_length=1, max_length=8000)
    feedback: Optional[str] = None   # 'positive', 'negative', 'correction'
    importance: float = Field(0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    namespace: str = "default"


class ReinforceRequest(BaseModel):
    """Request to strengthen a memory that was useful."""
    memory_id: str
    boost: float = Field(0.2, ge=0.0, le=1.0)


class CorrectRequest(BaseModel):
    """Request to correct/update a memory."""
    memory_id: str
    new_content: str = Field(..., min_length=1, max_length=10000)
    reason: str = ""


class ForgetRequest(BaseModel):
    """Request to forget/delete a memory."""
    memory_id: str
    hard_delete: bool = False


class DocumentUploadResponse(BaseModel):
    """Response from document upload/ingestion."""
    filename: str
    total_chunks: int
    memories_created: int
    document_type: str
    tags: List[str] = Field(default_factory=list)
    memory_ids: List[str] = Field(default_factory=list)
    text_length: int = 0
    status: str = "success"


# ============================================================
# KNOWLEDGE PAGES, PROVENANCE, VERSIONING, LINT
# ============================================================

class KnowledgePage(BaseModel):
    """
    A wiki-style knowledge page synthesized from memories about an entity or concept.
    Inspired by Karpathy's LLM Wiki pattern — auto-generated and auto-updated.
    """
    page_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str                   # Links to entity_nodes.entity_id
    title: str
    page_type: str = "entity"        # entity | concept | topic
    summary: str = ""                # LLM-synthesized wiki-style summary
    memory_ids: List[str] = Field(default_factory=list)  # Contributing memories
    version: int = 1
    last_updated: float = Field(default_factory=time.time)
    created_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceEntry(BaseModel):
    """
    Audit trail entry tracking the lifecycle of a memory.
    Records every creation, correction, supersession, and consolidation.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str
    parent_memory_ids: List[str] = Field(default_factory=list)
    operation: str                   # created | corrected | superseded | consolidated | derived
    reason: str = ""
    source_url: str = ""
    created_at: float = Field(default_factory=time.time)


class MemoryVersion(BaseModel):
    """
    A point-in-time snapshot of a memory before it was modified.
    Created automatically on corrections and supersessions.
    """
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str
    content: str
    strength: float = 1.0
    importance: float = 0.5
    confidence: float = 0.5
    changed_at: float = Field(default_factory=time.time)
    change_reason: str = ""


class LintReport(BaseModel):
    """Result of a memory system self-audit / lint check."""
    unresolved_contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    stale_memories: List[Dict[str, Any]] = Field(default_factory=list)
    orphan_memories: List[Dict[str, Any]] = Field(default_factory=list)
    outdated_knowledge_pages: List[Dict[str, Any]] = Field(default_factory=list)
    entity_coverage_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    duplicates: List[Dict[str, Any]] = Field(default_factory=list)
    total_issues: int = 0
    generated_at: float = Field(default_factory=time.time)
