"""
🧠 Memory Layer - A Biologically-Inspired Memory System for AI

Give any AI persistent, evolving memory that never forgets.
Memories strengthen with use, decay without it, and automatically
organize into a web of associations.

Usage:
    from memory_layer import MemoryManager

    brain = MemoryManager(db_path="my_brain.db")
    brain.remember("User prefers Python", importance=0.8)
    results = brain.recall("What language does the user prefer?")

Or run as a server:
    python run.py
"""

# ── Prevent OpenMP crash on macOS when FAISS + sentence-transformers
#    both ship their own libomp.  Must be set before any native lib loads.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

# ── Pre-import FAISS (if available) so its libomp loads first ──
try:
    import faiss as _faiss  # noqa: F401
except ImportError:
    pass

from .models import (
    Memory,
    MemoryType,
    MemoryLink,
    LinkType,
    WorkingMemoryItem,
    RecallResult,
    MemoryStats,
    ForgetRequest,
    KnowledgePage,
    ProvenanceEntry,
    MemoryVersion,
    LintReport,
    EpistemicStatus,
)
from .core import MemoryManager
from .embeddings import create_embedding_engine, EmbeddingEngine
from .storage_protocol import StorageBackend
from .storage_factory import create_storage
from .document_ingest import DocumentIngestor
from .enrichment import EnrichmentPipeline
from .reranker import create_reranker
from .entity_graph import EntityGraph
from .passport import export_passport, import_passport, inspect_passport, convert_passport
from .proxy import MemoryProxy
from .temporal import extract_temporal_refs, temporal_relevance, has_temporal_intent
from .active_memory import ActiveMemoryManager
from .graph_reasoner import GraphReasoner
from .predictive import PredictiveCache
from .reasoning_engine import ReasoningEngine
from .llm_extract import LocalFactExtractor
from .chat import ChatEngine
from .knowledge_pages import KnowledgePageManager
from .wiki_export import export_wiki
from .lint import MemoryLinter

__version__ = "0.4.0"
__all__ = [
    "MemoryManager",
    "Memory",
    "MemoryType",
    "MemoryLink",
    "LinkType",
    "WorkingMemoryItem",
    "RecallResult",
    "MemoryStats",
    "ForgetRequest",
    "EmbeddingEngine",
    "create_embedding_engine",
    "StorageBackend",
    "create_storage",
    "DocumentIngestor",
    "EnrichmentPipeline",
    "EntityGraph",
    "create_reranker",
    "MemoryProxy",
    "export_passport",
    "import_passport",
    "inspect_passport",
    "convert_passport",
    "extract_temporal_refs",
    "temporal_relevance",
    "has_temporal_intent",
    "ActiveMemoryManager",
    "GraphReasoner",
    "PredictiveCache",
    "ReasoningEngine",
    "LocalFactExtractor",
    "ChatEngine",
    "KnowledgePage",
    "ProvenanceEntry",
    "MemoryVersion",
    "LintReport",
    "EpistemicStatus",
    "KnowledgePageManager",
    "export_wiki",
    "MemoryLinter",
]
