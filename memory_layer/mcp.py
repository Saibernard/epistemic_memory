#!/usr/bin/env python3
"""
🧠 Memory Layer MCP Server for Cursor

Transport: newline-delimited JSON over stdio (NOT Content-Length framed).
Each message is one JSON object per line, terminated by \n.
"""

import sys
import json
import os
import re
import hashlib
import traceback
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

# ── Trace for debugging ──
# Write to the user's home dir, never the package dir (site-packages may
# be read-only when pip-installed).
TRACE_PATH = os.path.join(
    os.path.expanduser("~"), ".memory-layer", "mcp_trace.log",
)
TRACE_ENABLED = os.environ.get("MCP_TRACE", "0") == "1"
def trace(msg):
    if not TRACE_ENABLED:
        return
    try:
        os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
        with open(TRACE_PATH, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            f.flush()
    except OSError:
        pass

# ── Silence all print() so nothing leaks to stdout ──
import builtins
_real_print = builtins.print
def _silent_print(*args, **kwargs):
    kwargs["file"] = sys.stderr
    kwargs["flush"] = True
    _real_print(*args, **kwargs)
builtins.print = _silent_print
import logging
logging.basicConfig(level=logging.CRITICAL)

# Use standardized config paths; fall back gracefully if config module unavailable
try:
    from memory_layer.config import get_db_path, load_config, ensure_home_dir
    ensure_home_dir()
    _cfg = load_config()
    _default_db = get_db_path()
    _default_embed = _cfg.get("embeddings", "mode")
    _default_extract = _cfg.getboolean("llm", "extract")
except Exception:
    _default_db = os.path.join(
        os.path.expanduser("~"), ".memory-layer", "memory.db",
    )
    _default_embed = "local"
    _default_extract = False

DB_PATH = os.environ.get("MEMORY_DB_PATH", _default_db)
EMBEDDING_MODE = os.environ.get("MEMORY_EMBEDDING_MODE", _default_embed)
LLM_EXTRACT = os.environ.get("MEMORY_LLM_EXTRACT", "1" if _default_extract else "0") == "1"
brain = None

# ── Per-project namespace ─────────────────────────────────────────────
# Claude Code / Cursor launch MCP stdio servers with cwd = the project
# directory, so the project a conversation belongs to is derivable here.
# Every tool call defaults to that project's namespace: project A's
# memories never surface in project B.  User-level facts (preferences,
# identity) live in the reserved "global" namespace, shared everywhere.

GLOBAL_NAMESPACE = "global"

def _derive_project_namespace() -> str:
    override = os.environ.get("MEMORY_NAMESPACE", "").strip()
    if override:
        return override
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd in ("/", home, ""):
        # Not launched from a project directory — use the shared space
        return GLOBAL_NAMESPACE
    base = re.sub(
        r"[^a-z0-9_-]+", "-", os.path.basename(cwd).lower()
    ).strip("-") or "project"
    # Short path hash so two folders named e.g. "app" don't collide
    digest = hashlib.sha1(cwd.encode("utf-8", "replace")).hexdigest()[:6]
    return f"{base}-{digest}"

PROJECT_NAMESPACE = _derive_project_namespace()
MAX_CONTENT_CHARS = int(os.environ.get("MCP_MAX_CONTENT_CHARS", "10000"))
MAX_QUERY_CHARS = int(os.environ.get("MCP_MAX_QUERY_CHARS", "2000"))
MAX_TAGS = int(os.environ.get("MCP_MAX_TAGS", "20"))
MAX_TOP_K = int(os.environ.get("MCP_MAX_TOP_K", "50"))

def get_brain():
    global brain
    if brain is None:
        from memory_layer import MemoryManager
        from memory_layer.storage_factory import create_storage
        try:
            _backend = _cfg.get("storage", "backend", fallback="sqlite")
        except Exception:
            _backend = os.environ.get("MEMORY_STORAGE_BACKEND", "sqlite")
        storage = create_storage(_backend, sqlite_path=DB_PATH)
        brain = MemoryManager(
            db_path=DB_PATH,
            embedding_mode=EMBEDDING_MODE,
            llm_extract=LLM_EXTRACT,
            storage=storage,
        )
        trace(f"brain loaded (embedding={EMBEDDING_MODE}, llm_extract={LLM_EXTRACT}, storage={_backend})")
    return brain


# ── Warm-daemon thin-client routing ───────────────────────────────────
# One long-lived daemon holds the model + FAISS hot; this process then
# serves the hot tools over localhost HTTP in ~10-90ms instead of
# loading its own model.  Any failure falls back to the in-process
# brain, so behaviour is identical — just slower — without the daemon.

_client = None          # cached MemoryClient once verified up
_daemon_checked = False


def get_client():
    """Return a live MemoryClient, or None to use the in-process brain."""
    global _client, _daemon_checked
    if os.environ.get("MEMORY_NO_DAEMON", "0") in ("1", "true", "yes"):
        return None
    if _client is not None:
        return _client
    try:
        from memory_layer.daemon import (
            MemoryClient, ensure_daemon, daemon_matches_db,
        )
        # Never route to a daemon serving a different database
        if not daemon_matches_db(DB_PATH):
            return None
        if not _daemon_checked:
            _daemon_checked = True
            # Fire-and-forget spawn; warm path kicks in once it's up
            ensure_daemon(wait=False)
        c = MemoryClient()
        if c.is_up():
            _client = c
            trace("daemon client active")
            return _client
    except Exception:
        pass
    return None


class _Obj:
    """Attribute-access wrapper so daemon JSON flows through the same
    formatting code as in-process pydantic objects."""

    def __init__(self, d):
        self._d = d or {}

    def __getattr__(self, name):
        v = self._d.get(name)
        if isinstance(v, dict):
            return _Obj(v)
        return v


def _wrap_recall(rows):
    out = []
    for r in rows:
        o = _Obj(r)
        # normalise: metadata may be absent in serialized form
        if isinstance(r.get("memory"), dict) and r["memory"].get("metadata") is None:
            r["memory"]["metadata"] = {}
        out.append(o)
    return out


# ══════════════════════════════════════════
#  STDIO TRANSPORT — newline-delimited JSON
# ══════════════════════════════════════════

def send_msg(msg):
    """Send one JSON message as a single line to stdout."""
    line = json.dumps(msg, separators=(",", ":")) + "\n"
    data = line.encode("utf-8")
    mv = memoryview(data)
    while len(mv) > 0:
        n = os.write(1, mv)
        mv = mv[n:]
    trace(f"SENT id={msg.get('id')}")


_stdin_buf = b""

def recv_msg():
    """Read one JSON message (one line) from stdin using buffered reads."""
    global _stdin_buf
    while b"\n" not in _stdin_buf:
        chunk = os.read(0, 8192)
        if not chunk:
            raise EOFError("stdin closed")
        _stdin_buf += chunk

    line, _stdin_buf = _stdin_buf.split(b"\n", 1)
    line = line.strip()
    if not line:
        return None

    msg = json.loads(line)
    trace(f"RECV method={msg.get('method','?')} id={msg.get('id','?')}")
    return msg


# ══════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════

TOOLS = [
    {
        "name": "memory_remember",
        "description": "Store something in persistent memory. By default the memory is scoped to the CURRENT PROJECT (this working directory). Use scope='global' for user-level facts that apply everywhere (preferences, identity, tools they use). If similar information already exists, the old memory is automatically replaced with the new version.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember"},
                "importance": {"type": "number", "description": "0.0-1.0, default 0.7"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                "scope": {"type": "string", "enum": ["project", "global"], "description": "'project' (default): visible only in this project. 'global': user-level fact visible in every project."},
                "namespace": {"type": "string", "description": "Advanced: explicit namespace override (default: this project's namespace)"}
            },
            "required": ["content"]
        }
    },
    {
        "name": "memory_recall",
        "description": "Search persistent memory for relevant information: user preferences, past decisions, project context. Searches the CURRENT PROJECT's memories plus shared global (user-level) memories by default.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "top_k": {"type": "integer", "description": "Max results, default 5"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
                "include_global": {"type": "boolean", "description": "Also search shared user-level memories (default: true)"},
                "namespace": {"type": "string", "description": "Advanced: explicit namespace override (default: this project's namespace)"},
                "reasoning": {"type": "boolean", "description": "Enable multi-hop graph reasoning for deeper answers"},
                "include_history": {"type": "boolean", "description": "Include superseded/historical versions of memories"},
                "diversity": {"type": "boolean", "description": "Apply MMR diversity to reduce redundant results"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_forget",
        "description": "Forget/delete a specific memory by ID. Use this to remove incorrect or unwanted memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "The ID of the memory to forget"},
                "hard_delete": {"type": "boolean", "description": "If true, permanently delete (default: false = soft-delete)"}
            },
            "required": ["memory_id"]
        }
    },
    {
        "name": "memory_record_episode",
        "description": "Record a summary of the current conversation for future reference.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Conversation summary"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"},
                "namespace": {"type": "string", "description": "Memory namespace (default: 'default')"}
            },
            "required": ["summary"]
        }
    },
    {
        "name": "memory_stats",
        "description": "Get memory system statistics.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "memory_ingest_document",
        "description": "Upload a document file into memory. Supports PDF, DOCX, TXT, MD, CSV, JSON. The document is automatically chunked and each chunk stored as a searchable memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the document file"},
                "importance": {"type": "number", "description": "0.0-1.0, default 0.6"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                "namespace": {"type": "string", "description": "Memory namespace (default: 'default')"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "memory_ingest_url",
        "description": "Fetch a web page (any URL: docs sites, Confluence, wikis, blog posts, GitHub READMEs) and store its content as searchable memories. The page is automatically fetched, cleaned, chunked, and stored.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The web page URL to ingest"},
                "importance": {"type": "number", "description": "0.0-1.0, default 0.6"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                "namespace": {"type": "string", "description": "Memory namespace (default: 'default')"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "memory_health",
        "description": "Run a health check on the memory system. Returns database integrity status, FAISS sync status, storage stats, and any issues found.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "memory_maintenance",
        "description": "Run all maintenance tasks: consolidation, decay, pruning of old reasoning conclusions, integrity repair, and FAISS sync. Use periodically to keep the memory system healthy.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "memory_backup",
        "description": "Create a backup of the memory database. Returns the path to the backup file.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "memory_synthesize",
        "description": "Synthesize knowledge about a topic from all stored memories. Creates a coherent summary from relevant memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The topic to synthesize knowledge about"},
                "store_result": {"type": "boolean", "description": "If true, store the synthesis as a new memory (default: false)"},
                "namespace": {"type": "string", "description": "Memory namespace (default: 'default')"}
            },
            "required": ["topic"]
        }
    }
]

def _parse_tags(args):
    tags = args.get("tags", [])
    if not isinstance(tags, list):
        raise ValueError("tags must be a list of strings")
    return [str(t)[:80] for t in tags[:MAX_TAGS]]


def handle_tool(name, args):
    # Default namespace = this project (derived from cwd at launch)
    ns = str(args.get("namespace", PROJECT_NAMESPACE)).strip() or PROJECT_NAMESPACE

    if name == "memory_remember":
        content = str(args.get("content", "")).strip()
        if not content:
            raise ValueError("content is required")
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError(f"content too long (max {MAX_CONTENT_CHARS} chars)")
        importance = float(args.get("importance", 0.7))
        importance = max(0.0, min(1.0, importance))
        tags = _parse_tags(args)
        # scope='global' → user-level fact shared across all projects
        if str(args.get("scope", "project")).strip().lower() == "global":
            ns = GLOBAL_NAMESPACE

        c = get_client()
        if c is not None:
            try:
                m = _Obj(c.remember(content=content, importance=importance,
                                    tags=tags, namespace=ns))
            except Exception:
                m = None
        else:
            m = None
        if m is None:
            from memory_layer import MemoryType
            m = get_brain().remember(
                content=content, memory_type=MemoryType.SEMANTIC,
                importance=importance, tags=tags, namespace=ns,
            )
        meta = m.metadata
        if isinstance(meta, _Obj):
            meta = meta._d
        meta = meta or {}
        r = f'Remembered: "{content}" (importance={m.importance:.2f})'
        if meta.get("replaces"):
            r += (f"\nAuto-updated: replaced {len(meta['replaces'])} "
                  "outdated memory(ies) with this new version.")
        if meta.get("contradicts"):
            r += (f"\nNote: conflicts with {len(meta['contradicts'])} "
                  "existing memory(ies).")
        return r

    elif name == "memory_recall":
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query too long (max {MAX_QUERY_CHARS} chars)")
        top_k = int(args.get("top_k", 5))
        top_k = max(1, min(MAX_TOP_K, top_k))
        recall_tags = args.get("tags") or None
        reasoning = bool(args.get("reasoning", False))
        include_history = bool(args.get("include_history", False))
        diversity = bool(args.get("diversity", False))

        effective_top_k = top_k * 3 if diversity else top_k

        def _do_recall(namespace, use_reasoning):
            """Warm-daemon recall with in-process fallback."""
            c = get_client()
            if c is not None and not use_reasoning:
                try:
                    return _wrap_recall(c.recall(
                        query, top_k=effective_top_k, namespace=namespace,
                        min_strength=0.05, tags=recall_tags,
                        include_history=include_history,
                    ))
                except Exception:
                    pass
            return get_brain().recall(
                query, top_k=effective_top_k, min_strength=0.05,
                tags=recall_tags, namespace=namespace,
                reasoning=use_reasoning, include_history=include_history,
            )

        results = _do_recall(ns, reasoning)

        # Merge shared user-level (global) memories unless disabled or
        # we're already searching the global namespace.
        include_global = bool(args.get("include_global", True))
        if include_global and ns != GLOBAL_NAMESPACE:
            try:
                global_results = _do_recall(GLOBAL_NAMESPACE, False)
                seen_ids = {r.memory.id for r in results}
                merged = list(results) + [
                    r for r in global_results if r.memory.id not in seen_ids
                ]
                merged.sort(key=lambda r: r.composite_score, reverse=True)
                results = merged[:effective_top_k]
            except Exception:
                pass  # global merge is best-effort; project results stand

        if diversity and len(results) > 1:
            import numpy as np
            selected = [0]
            remaining = list(range(1, len(results)))
            embeddings = []
            for r in results:
                if r.memory.embedding:
                    embeddings.append(np.array(r.memory.embedding, dtype=np.float32))
                else:
                    embeddings.append(np.zeros(1))
            while len(selected) < top_k and remaining:
                best_idx, best_score = None, -float('inf')
                for idx in remaining:
                    rel = results[idx].composite_score
                    max_sim = 0.0
                    emb_i = embeddings[idx]
                    for sel_idx in selected:
                        emb_s = embeddings[sel_idx]
                        if len(emb_i) > 1 and len(emb_s) > 1:
                            norm = np.linalg.norm(emb_i) * np.linalg.norm(emb_s)
                            if norm > 0:
                                max_sim = max(max_sim, float(np.dot(emb_i, emb_s) / norm))
                    mmr = 0.7 * rel - 0.3 * max_sim
                    if mmr > best_score:
                        best_score = mmr
                        best_idx = idx
                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
            results = [results[i] for i in selected]

        if not results:
            return f'No memories found for: "{query}"'
        out = f"Found {len(results)} memory(ies):\n\n"
        for i, r in enumerate(results):
            out += f"{i+1}. [{r.memory.id[:8]}] {r.memory.content}\n   (relevance={r.relevance_score:.3f})\n\n"

        if reasoning and results:
            chain = results[0].memory.metadata.get("_reasoning_chain")
            if chain:
                synthesis = chain[-1].get("synthesis", "") if isinstance(chain[-1], dict) else str(chain[-1])
                if synthesis:
                    out += f"--- Reasoning Synthesis ---\n{synthesis}\n"

        return out

    elif name == "memory_forget":
        memory_id = str(args.get("memory_id", "")).strip()
        if not memory_id:
            raise ValueError("memory_id is required")
        hard = bool(args.get("hard_delete", False))
        ok = get_brain().forget_memory(memory_id, hard=hard)
        if not ok:
            return f"Memory not found: {memory_id}"
        action = "permanently deleted" if hard else "deactivated"
        return f"Memory {memory_id} {action}."

    elif name == "memory_record_episode":
        summary = str(args.get("summary", "")).strip()
        if not summary:
            raise ValueError("summary is required")
        if len(summary) > MAX_CONTENT_CHARS:
            raise ValueError(f"summary too long (max {MAX_CONTENT_CHARS} chars)")
        tags = _parse_tags(args) + ["episode"]
        user_msg = str(args.get("user_message", summary)).strip()
        assistant_msg = str(args.get("assistant_response", "")).strip()
        c = get_client()
        if c is not None:
            try:
                c.record_episode(
                    user_message=user_msg,
                    assistant_response=assistant_msg or summary,
                    importance=0.6, tags=tags, namespace=ns,
                )
                return "Episode recorded."
            except Exception:
                pass
        get_brain().record_episode(
            user_message=user_msg,
            assistant_response=assistant_msg or summary,
            importance=0.6,
            tags=tags,
            namespace=ns,
        )
        return "Episode recorded."

    elif name == "memory_stats":
        s = get_brain().get_stats()
        return (
            f"Total: {s.total_memories} ({s.episodic_count} episodic, "
            f"{s.semantic_count} semantic, {s.procedural_count} procedural) "
            f"| Links: {s.total_links}"
        )

    elif name == "memory_ingest_document":
        file_path = str(args.get("file_path", "")).strip()
        if not file_path:
            raise ValueError("file_path is required")
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        importance = float(args.get("importance", 0.6))
        importance = max(0.0, min(1.0, importance))
        tags = _parse_tags(args)

        from memory_layer.document_ingest import DocumentIngestor
        ingestor = DocumentIngestor()
        chunks = ingestor.extract_and_chunk(file_path, extra_tags=tags)

        if not chunks:
            return f"No text could be extracted from: {file_path}"

        memory_ids = []
        for chunk in chunks:
            m = get_brain().remember(
                content=chunk["content"],
                memory_type=MemoryType.SEMANTIC,
                importance=importance,
                tags=chunk["tags"],
                metadata=chunk["metadata"],
                namespace=ns,
            )
            memory_ids.append(m.id)

        filename = os.path.basename(file_path)
        return (
            f"Document ingested: {filename}\n"
            f"  Chunks: {len(chunks)}\n"
            f"  Memories created: {len(memory_ids)}\n"
            f"  Tags: {', '.join(tags) if tags else '(auto)'}\n"
            f"You can now recall any part of this document with memory_recall."
        )

    elif name == "memory_ingest_url":
        url = str(args.get("url", "")).strip()
        if not url:
            raise ValueError("url is required")

        importance = float(args.get("importance", 0.6))
        importance = max(0.0, min(1.0, importance))
        tags = _parse_tags(args)

        from memory_layer.document_ingest import DocumentIngestor
        ingestor = DocumentIngestor()
        chunks = ingestor.extract_and_chunk_url(url, extra_tags=tags)

        if not chunks:
            return f"No text could be extracted from: {url}"

        memory_ids = []
        for chunk in chunks:
            m = get_brain().remember(
                content=chunk["content"],
                memory_type=MemoryType.SEMANTIC,
                importance=importance,
                tags=chunk["tags"],
                metadata=chunk["metadata"],
                namespace=ns,
            )
            memory_ids.append(m.id)

        page_title = chunks[0]["metadata"].get("page_title", url) if chunks else url
        return (
            f"URL ingested: {page_title}\n"
            f"  Source: {url}\n"
            f"  Chunks: {len(chunks)}\n"
            f"  Memories created: {len(memory_ids)}\n"
            f"You can now recall any part of this page with memory_recall."
        )

    elif name == "memory_health":
        report = get_brain().health_check()
        status = report.get("status", "unknown")
        db = report.get("database", {})
        issues = report.get("issues", [])
        out = f"Health: {status}\n"
        out += f"  SQLite: {'OK' if db.get('sqlite_ok') else 'FAIL'}\n"
        out += f"  Active memories: {db.get('active_memories', '?')}\n"
        out += f"  FAISS synced: {'yes' if report.get('faiss_synced') else 'no'}\n"
        if issues:
            out += f"  Issues: {', '.join(issues)}\n"
        storage = report.get("storage", {})
        out += f"  DB size: {storage.get('db_size_mb', '?')} MB"
        return out

    elif name == "memory_maintenance":
        results = get_brain().maintenance()
        pruned = results.get("reasoning_pruned", 0)
        cleaned = results.get("queue_cleaned", 0)
        rebuilt = results.get("faiss_rebuilt", False)
        ss = results.get("storage_stats", {})
        return (
            f"Maintenance complete:\n"
            f"  Reasoning pruned: {pruned}\n"
            f"  Queue cleaned: {cleaned}\n"
            f"  FAISS rebuilt: {rebuilt}\n"
            f"  Active memories: {ss.get('active_memories', '?')}\n"
            f"  DB size: {ss.get('db_size_mb', '?')} MB"
        )

    elif name == "memory_backup":
        path = get_brain().backup()
        return f"Backup created: {path}"

    elif name == "memory_synthesize":
        topic = str(args.get("topic", "")).strip()
        if not topic:
            raise ValueError("topic is required")
        store = bool(args.get("store_result", False))
        result = get_brain().synthesize(topic=topic, store_result=store, namespace=ns)
        out = f"Synthesis for '{topic}':\n\n{result['synthesis']}\n\nSources: {result['source_count']} memories"
        if result.get("stored_memory_id"):
            out += f"\nStored as memory: {result['stored_memory_id'][:8]}"
        return out

    return f"Unknown tool: {name}"


# ══════════════════════════════════════════
#  REQUEST HANDLER
# ══════════════════════════════════════════

def handle(msg):
    method = msg.get("method", "")
    mid = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "memory-layer",
                "version": "0.5.0",
                "title": f"Memory Layer (project: {PROJECT_NAMESPACE})",
            },
            "instructions": (
                "Persistent, local, per-project memory. Memories default to "
                f"this project's namespace ('{PROJECT_NAMESPACE}'); use "
                "memory_remember scope='global' for user-level facts "
                "(preferences, identity) shared across all projects. At the "
                "start of a task, memory_recall relevant context; when the "
                "user states a lasting fact, decision, or preference, "
                "memory_remember it."
            ),
        }}
    if method.startswith("notifications/"):
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        try:
            text = handle_tool(params.get("name", ""), params.get("arguments", {}))
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": text}], "isError": False}}
        except Exception as e:
            trace(f"TOOL_ERROR: {traceback.format_exc()}")
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}}
    if mid is not None:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Unknown: {method}"}}
    return None


# ══════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════

def main():
    trace("=" * 40)
    trace(f"START pid={os.getpid()}")

    while True:
        try:
            msg = recv_msg()
            if msg is None:
                continue
            resp = handle(msg)
            if resp is not None:
                send_msg(resp)
        except EOFError as e:
            trace(f"EOF: {e}")
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            trace(f"ERROR: {traceback.format_exc()}")

    trace("EXIT")


if __name__ == "__main__":
    main()
