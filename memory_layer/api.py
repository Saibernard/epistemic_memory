"""
FastAPI REST Interface for the Memory Layer.

This is the universal plug-in API. Any AI application can connect
to this server to gain persistent, evolving memory.

Start the server:
    python run.py

Then any AI can use it:
    POST /remember     → Store a memory
    POST /recall       → Search memories by meaning
    POST /episode      → Log an interaction
    POST /reinforce    → Strengthen a useful memory
    POST /correct      → Fix incorrect information
    POST /forget       → Forget a specific memory
    POST /consolidate  → Trigger memory consolidation
    GET  /context      → Get working memory
    GET  /stats        → System health & statistics
    GET  /health       → Health check
"""

import asyncio
import os
import time
import threading
from collections import defaultdict, deque
from typing import Optional, List
from contextlib import asynccontextmanager
from functools import partial

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    Memory,
    MemoryType,
    MemoryLink,
    WorkingMemoryItem,
    RecallResult,
    MemoryStats,
    RememberRequest,
    RecallRequest,
    EpisodeRequest,
    ReinforceRequest,
    CorrectRequest,
    ForgetRequest,
    DocumentUploadResponse,
)
from .core import MemoryManager
from .document_ingest import DocumentIngestor, SUPPORTED_EXTENSIONS


# ─────────────────────────────────────────────
# APPLICATION LIFECYCLE
# ─────────────────────────────────────────────

_manager: Optional[MemoryManager] = None
_rate_limit_lock = threading.Lock()
_rate_limit_hits = defaultdict(deque)

API_KEY = os.environ.get("MEMORY_API_KEY", "").strip()
RATE_LIMIT_RPM = int(os.environ.get("MEMORY_RATE_LIMIT_RPM", "120"))
cors_raw = os.environ.get("MEMORY_CORS_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in cors_raw.split(",") if o.strip()]
START_TIME = time.time()


def get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        raise HTTPException(
            status_code=500, detail="Memory manager not initialized"
        )
    return _manager


async def _run_sync(func, *args, **kwargs):
    """Run a blocking function in the default executor to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    try:
        from .config import get_db_path, load_config, ensure_home_dir
        ensure_home_dir()
        cfg = load_config()
        db_path = get_db_path()
        embed_mode = cfg.get("embeddings", "mode")
        embed_model = cfg.get("embeddings", "model")
        llm_extract = cfg.getboolean("llm", "extract")
        llm_model = cfg.get("llm", "model")
        backend = cfg.get("storage", "backend", fallback="sqlite")
    except Exception:
        db_path = os.environ.get("MEMORY_DB_PATH", "memory.db")
        embed_mode = os.environ.get("MEMORY_EMBEDDING_MODE", "local")
        embed_model = os.environ.get("MEMORY_EMBEDDING_MODEL", "all-mpnet-base-v2")
        llm_extract = os.environ.get("MEMORY_LLM_EXTRACT", "0") == "1"
        llm_model = "gpt-4o-mini"
        backend = os.environ.get("MEMORY_STORAGE_BACKEND", "sqlite")

    from .storage_factory import create_storage
    storage = create_storage(backend, sqlite_path=db_path)

    _manager = MemoryManager(
        db_path=db_path,
        embedding_model=embed_model,
        embedding_mode=embed_mode,
        llm_extract=llm_extract,
        llm_extract_model=llm_model,
        storage=storage,
    )
    print(f"Memory Layer API ready at http://localhost:8484")
    print(f"   Docs: http://localhost:8484/docs\n")
    yield
    print(f"\nMemory Layer shutting down gracefully...")
    if _manager:
        _manager.shutdown()
    print(f"  Memories persist in: {db_path}")


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="🧠 Memory Layer",
    description="""
## A Biologically-Inspired Memory System for AI

Give any AI persistent, evolving memory that never forgets.

### What makes this different from a vector database?

| Feature | Vector DB | Memory Layer |
|---------|-----------|--------------|
| Storage | Static vectors | Living memories with strength, importance, decay |
| Recall | Similarity search | Similarity + strength + importance + associations |
| Organization | Flat | Multi-type (episodic, semantic, procedural) + graph |
| Over time | Nothing changes | Memories decay, strengthen, consolidate, evolve |
| Learning | None | Episodic → Semantic consolidation (like human sleep) |

### Connect any AI in 3 lines:

```python
import requests

# Store
requests.post("http://localhost:8484/remember", json={"content": "User likes Python"})

# Recall  
requests.post("http://localhost:8484/recall", json={"query": "programming preferences"})
```
    """,
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=False if (ALLOWED_ORIGINS == ["*"]) else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_api_key(request: Request):
    if not API_KEY:
        return
    provided = request.headers.get("x-api-key", "").strip()
    auth = request.headers.get("authorization", "").strip()
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if provided != API_KEY and bearer != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _check_rate_limit(request: Request):
    if RATE_LIMIT_RPM <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60.0
    with _rate_limit_lock:
        q = _rate_limit_hits[client]
        while q and q[0] < window_start:
            q.popleft()
        if len(q) >= RATE_LIMIT_RPM:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)


def _guard(request: Request):
    _check_api_key(request)
    _check_rate_limit(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", summary="Health check", tags=["System"])
async def health():
    basic = {
        "status": "healthy",
        "service": "memory-layer",
        "version": "0.4.0",
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }
    try:
        mgr = get_manager()
        report = await _run_sync(mgr.health_check)
        basic["status"] = report.get("status", "ok")
        basic["details"] = report
    except Exception:
        pass
    return basic


@app.get("/stats", response_model=MemoryStats, summary="System statistics", tags=["System"])
async def get_stats(request: Request, namespace: Optional[str] = None):
    _guard(request)
    return await _run_sync(get_manager().get_stats, namespace=namespace)


@app.get("/storage-stats", summary="Detailed storage statistics", tags=["System"])
async def storage_stats(request: Request):
    _guard(request)
    return await _run_sync(get_manager().storage.get_storage_stats)


@app.post("/maintenance", summary="Run maintenance tasks", tags=["System"])
async def run_maintenance(request: Request):
    """Run consolidation, decay, pruning, and integrity repair."""
    _guard(request)
    return await _run_sync(get_manager().maintenance)


@app.post("/backup", summary="Create database backup", tags=["System"])
async def create_backup(request: Request):
    _guard(request)
    path = await _run_sync(get_manager().backup)
    return {"backup_path": path}


# ── Store & Recall ──────────────────────────

@app.post("/remember", response_model=Memory, summary="Store a memory", tags=["Memory"])
async def remember(payload: RememberRequest, request: Request):
    """
    Store a new memory.

    The memory will be embedded locally, checked for contradictions,
    stored persistently, and automatically linked to related memories.
    """
    _guard(request)
    manager = get_manager()
    memory = await _run_sync(
        manager.remember,
        content=payload.content,
        memory_type=payload.memory_type,
        importance=payload.importance,
        tags=payload.tags,
        metadata=payload.metadata,
        namespace=payload.namespace,
    )
    return memory


def _apply_mmr_results(results: list, target_k: int, lambda_param: float = 0.7) -> list:
    """Maximum Marginal Relevance: balance relevance with diversity."""
    if len(results) <= target_k:
        return results

    import numpy as np

    embeddings = []
    for r in results:
        if r.memory.embedding:
            embeddings.append(np.array(r.memory.embedding, dtype=np.float32))
        else:
            embeddings.append(np.zeros(1))

    selected = [0]
    remaining = list(range(1, len(results)))

    while len(selected) < target_k and remaining:
        best_idx = None
        best_score = -float('inf')

        for idx in remaining:
            relevance = results[idx].composite_score

            max_sim = 0.0
            emb_i = embeddings[idx]
            for sel_idx in selected:
                emb_s = embeddings[sel_idx]
                if len(emb_i) > 1 and len(emb_s) > 1:
                    norm = np.linalg.norm(emb_i) * np.linalg.norm(emb_s)
                    if norm > 0:
                        sim = float(np.dot(emb_i, emb_s) / norm)
                        max_sim = max(max_sim, sim)

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [results[i] for i in selected]


@app.post("/recall", response_model=List[RecallResult], summary="Recall memories", tags=["Memory"])
async def recall(payload: RecallRequest, request: Request):
    """
    Recall memories relevant to a natural language query.

    Results are ranked by: relevance × current_strength × importance.
    Recalled memories are automatically reinforced (spaced repetition).
    """
    _guard(request)
    manager = get_manager()
    effective_top_k = payload.top_k * 3 if payload.diversity else payload.top_k
    results = await _run_sync(
        manager.recall,
        query=payload.query,
        memory_types=payload.memory_types,
        top_k=effective_top_k,
        min_strength=payload.min_strength,
        min_confidence=payload.min_confidence,
        include_associations=payload.include_associations,
        tags=payload.tags,
        namespace=payload.namespace,
        reasoning=payload.reasoning,
        include_history=payload.include_history,
    )
    if payload.diversity and len(results) > 1:
        results = _apply_mmr_results(results, payload.top_k)
    return results


# ── Episodes ────────────────────────────────

@app.post("/episode", response_model=Memory, summary="Record interaction", tags=["Episodes"])
async def record_episode(payload: EpisodeRequest, request: Request):
    """
    Record a user-assistant interaction as an episodic memory.

    Feedback adjusts importance:
    - "positive" → small boost
    - "negative" → bigger boost
    - "correction" → biggest boost
    """
    _guard(request)
    manager = get_manager()
    memory = await _run_sync(
        manager.record_episode,
        user_message=payload.user_message,
        assistant_response=payload.assistant_response,
        feedback=payload.feedback,
        importance=payload.importance,
        tags=payload.tags,
        metadata=payload.metadata,
        namespace=payload.namespace,
    )
    return memory


# ── Memory Management ───────────────────────

@app.post("/reinforce", response_model=Memory, summary="Reinforce memory", tags=["Memory"])
async def reinforce(payload: ReinforceRequest, request: Request):
    _guard(request)
    manager = get_manager()
    memory = await _run_sync(manager.reinforce_memory, payload.memory_id, boost=payload.boost)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@app.post("/correct", response_model=Memory, summary="Correct memory", tags=["Memory"])
async def correct(payload: CorrectRequest, request: Request):
    """Correct/update a memory. Old version is deactivated, new created with boosted importance."""
    _guard(request)
    manager = get_manager()
    memory = await _run_sync(
        manager.correct_memory,
        memory_id=payload.memory_id,
        new_content=payload.new_content,
        reason=payload.reason,
    )
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@app.post("/forget", summary="Forget a memory", tags=["Memory"])
async def forget(payload: ForgetRequest, request: Request):
    """Forget a specific memory (soft-delete by default, hard-delete optional)."""
    _guard(request)
    manager = get_manager()
    ok = await _run_sync(manager.forget_memory, payload.memory_id, hard=payload.hard_delete)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    action = "permanently deleted" if payload.hard_delete else "deactivated"
    return {"status": action, "memory_id": payload.memory_id}


@app.get(
    "/memory/{memory_id}",
    response_model=Memory,
    summary="Get specific memory",
    tags=["Memory"],
)
async def get_memory(memory_id: str, request: Request):
    _guard(request)
    manager = get_manager()
    memory = manager.storage.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@app.delete(
    "/memory/{memory_id}",
    summary="Deactivate memory",
    tags=["Memory"],
)
async def delete_memory(memory_id: str, request: Request):
    _guard(request)
    manager = get_manager()
    memory = manager.storage.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    await _run_sync(manager.forget_memory, memory_id, hard=False)
    return {"status": "deactivated", "memory_id": memory_id}


@app.get("/memories", response_model=List[Memory], summary="List memories", tags=["Memory"])
async def list_memories(
    memory_type: Optional[str] = None,
    active_only: bool = True,
    namespace: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    request: Request = None,
):
    """List all memories, optionally filtered by type, namespace, or tags."""
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if request is not None:
        _guard(request)
    manager = get_manager()
    mt = MemoryType(memory_type) if memory_type else None
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    all_memories = manager.storage.get_all_memories(
        memory_type=mt, active_only=active_only,
        namespace=namespace, tags=tag_list,
    )
    return all_memories[offset: offset + limit]


# ── Working Memory ──────────────────────────

@app.get(
    "/context",
    response_model=List[WorkingMemoryItem],
    summary="Get working memory",
    tags=["Working Memory"],
)
async def get_context(request: Request):
    _guard(request)
    return get_manager().get_working_context()


@app.delete("/working-memory", summary="Clear working memory", tags=["Working Memory"])
async def clear_working_memory(request: Request):
    _guard(request)
    get_manager().clear_working_memory()
    return {"status": "cleared"}


# ── Maintenance ─────────────────────────────

@app.post("/consolidate", summary="Consolidate memories", tags=["Maintenance"])
async def consolidate(request: Request):
    """Trigger memory consolidation (episodic → semantic knowledge)."""
    _guard(request)
    return await _run_sync(get_manager().consolidate)


@app.post("/decay", summary="Run decay cycle", tags=["Maintenance"])
async def run_decay(request: Request):
    """Apply Ebbinghaus forgetting curve to all memories."""
    _guard(request)
    return await _run_sync(get_manager().run_decay)


# ── Document Upload ─────────────────────────

_ingestor = DocumentIngestor()


@app.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload a document",
    tags=["Documents"],
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    importance: float = Form(0.6),
    tags: str = Form(""),
    namespace: str = Form("default"),
):
    """
    Upload a document (PDF, DOCX, TXT, MD, CSV, JSON) and store its
    contents as searchable memories.

    Supported formats: PDF, DOCX, TXT, MD, CSV, JSON (max 10 MB).
    """
    _guard(request)
    manager = get_manager()

    filename = file.filename or "unknown.txt"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    extra_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    try:
        chunks = _ingestor.extract_and_chunk_bytes(
            data=data,
            filename=filename,
            extra_tags=extra_tags,
        )
    except ImportError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from the document.",
        )

    memory_ids = []
    importance = max(0.0, min(1.0, importance))

    for chunk in chunks:
        memory = await _run_sync(
            manager.remember,
            content=chunk["content"],
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            tags=chunk["tags"],
            metadata=chunk["metadata"],
            namespace=namespace,
        )
        memory_ids.append(memory.id)

    text_length = sum(len(c["content"]) for c in chunks)

    return DocumentUploadResponse(
        filename=filename,
        total_chunks=len(chunks),
        memories_created=len(memory_ids),
        document_type=ext.lstrip("."),
        tags=extra_tags + ["document"],
        memory_ids=memory_ids,
        text_length=text_length,
        status="success",
    )


@app.post(
    "/upload/text",
    response_model=DocumentUploadResponse,
    summary="Upload raw text as document",
    tags=["Documents"],
)
async def upload_text_document(
    request: Request,
    content: str = Form(...),
    filename: str = Form("document.txt"),
    importance: float = Form(0.6),
    tags: str = Form(""),
    namespace: str = Form("default"),
):
    """Upload raw text content and store it as chunked memories."""
    _guard(request)
    manager = get_manager()

    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Empty content")

    extra_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    chunks = _ingestor.chunk_text(
        text=content,
        source_filename=filename,
        extra_tags=extra_tags,
    )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No content could be chunked from the input.",
        )

    memory_ids = []
    importance = max(0.0, min(1.0, importance))

    for chunk in chunks:
        memory = await _run_sync(
            manager.remember,
            content=chunk["content"],
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            tags=chunk["tags"],
            metadata=chunk["metadata"],
            namespace=namespace,
        )
        memory_ids.append(memory.id)

    text_length = sum(len(c["content"]) for c in chunks)

    return DocumentUploadResponse(
        filename=filename,
        total_chunks=len(chunks),
        memories_created=len(memory_ids),
        document_type=os.path.splitext(filename)[1].lstrip(".") or "text",
        tags=extra_tags + ["document"],
        memory_ids=memory_ids,
        text_length=text_length,
        status="success",
    )


# ── URL Ingestion ────────────────────────────

class URLIngestRequest(BaseModel):
    url: str
    importance: float = 0.6
    tags: List[str] = []
    namespace: str = "default"


@app.post(
    "/upload/url",
    response_model=DocumentUploadResponse,
    summary="Ingest a web page",
    tags=["Documents"],
)
async def upload_url(payload: URLIngestRequest, request: Request):
    """
    Fetch a web page, extract its main content, and store it as
    searchable memories.
    """
    _guard(request)
    manager = get_manager()

    try:
        chunks = _ingestor.extract_and_chunk_url(
            url=payload.url,
            extra_tags=payload.tags,
        )
    except ImportError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from the URL.",
        )

    memory_ids = []
    importance = max(0.0, min(1.0, payload.importance))

    for chunk in chunks:
        memory = await _run_sync(
            manager.remember,
            content=chunk["content"],
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            tags=chunk["tags"],
            metadata=chunk["metadata"],
            namespace=payload.namespace,
        )
        memory_ids.append(memory.id)

    text_length = sum(len(c["content"]) for c in chunks)
    page_title = chunks[0]["metadata"].get("page_title", payload.url) if chunks else payload.url

    return DocumentUploadResponse(
        filename=page_title,
        total_chunks=len(chunks),
        memories_created=len(memory_ids),
        document_type="url",
        tags=payload.tags + ["url"],
        memory_ids=memory_ids,
        text_length=text_length,
        status="success",
    )


# ── Chat ─────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    mode: str = "local"
    namespace: str = "default"
    top_k: int = 8


class ChatResponse(BaseModel):
    answer: str
    sources: list = []
    has_answer: bool = True
    mode: str = "local"


@app.post("/chat", response_model=ChatResponse, summary="Chat with memory graph", tags=["Chat"])
async def chat(request: Request, payload: ChatRequest):
    """
    Ask a question against the memory graph.

    Modes:
    - local: Pure retrieval, returns ranked memory excerpts (no LLM, free)
    - llm: Uses an LLM to synthesize a natural language answer from memories

    The bot will NOT hallucinate — if the memories don't contain the answer,
    it returns "I don't have that information."
    """
    _guard(request)
    from .chat import ChatEngine

    manager = get_manager()
    try:
        engine = ChatEngine(
            brain=manager,
            mode=payload.mode,
            namespace=payload.namespace,
            top_k=payload.top_k,
        )
    except (ValueError, ImportError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Chat mode '{payload.mode}' unavailable: {e}. "
                   f"Use mode='local' or set the required API key.",
        )

    response = await _run_sync(engine.ask, payload.question)
    return ChatResponse(
        answer=response.answer,
        sources=response.sources,
        has_answer=response.has_answer,
        mode=response.mode,
    )


# ── Memory Passport (Export / Import / Convert) ─────

class ExportRequest(BaseModel):
    namespace: Optional[str] = None
    include_embeddings: bool = False
    include_inactive: bool = False
    passphrase: Optional[str] = None


class ImportOptions(BaseModel):
    passphrase: Optional[str] = None
    target_namespace: Optional[str] = None
    skip_duplicates: bool = True
    reembed: bool = True


@app.post("/passport/export", summary="Export memory passport", tags=["Passport"])
async def passport_export(payload: ExportRequest, request: Request):
    """
    Export all memories as a Universal Memory Passport (JSON).

    Supports optional AES-256 encryption with a user passphrase.
    The passport can be imported into any Memory Layer instance,
    or converted for use with other AI memory systems.
    """
    _guard(request)
    import tempfile
    from .passport import export_passport

    manager = get_manager()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        summary = await _run_sync(
            export_passport,
            manager.storage,
            tmp_path,
            namespace=payload.namespace,
            include_embeddings=payload.include_embeddings,
            include_inactive=payload.include_inactive,
            passphrase=payload.passphrase,
        )

        import json as _json
        with open(tmp_path) as f:
            passport_data = _json.load(f)

        return {"summary": summary, "passport": passport_data}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/passport/import", summary="Import memory passport", tags=["Passport"])
async def passport_import(
    request: Request,
    file: UploadFile = File(...),
    passphrase: str = Form(""),
    target_namespace: str = Form(""),
    skip_duplicates: bool = Form(True),
    reembed: bool = Form(True),
):
    """
    Import memories from a Universal Memory Passport or any supported format.

    Supported formats:
    - Memory Layer Universal Passport (v1, v2)
    - Memory Layer legacy export
    - Mem0 export
    - Zep/Graphiti export
    - ChatGPT memory export
    - Claude Projects export
    """
    _guard(request)
    import tempfile
    from .passport import import_passport

    manager = get_manager()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as f:
        f.write(await file.read())
        tmp_path = f.name

    try:
        summary = await _run_sync(
            import_passport,
            manager.storage,
            tmp_path,
            passphrase=passphrase or None,
            target_namespace=target_namespace or None,
            skip_duplicates=skip_duplicates,
            reembed=reembed,
            embeddings_engine=manager.embeddings if reembed else None,
        )

        if summary.get("memories_imported", 0) > 0:
            manager._rebuild_faiss_indices()

        return {"summary": summary}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/passport/inspect", summary="Inspect a passport file", tags=["Passport"])
async def passport_inspect(
    request: Request,
    file: UploadFile = File(...),
    passphrase: str = Form(""),
):
    """Inspect a passport file without importing it. Returns metadata and statistics."""
    _guard(request)
    import tempfile
    from .passport import inspect_passport

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as f:
        f.write(await file.read())
        tmp_path = f.name

    try:
        info = inspect_passport(tmp_path, passphrase=passphrase or None)
        return info
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
