#!/usr/bin/env python3
"""
🧠 Memory Layer MCP Server for Cursor

Transport: newline-delimited JSON over stdio (NOT Content-Length framed).
Each message is one JSON object per line, terminated by \n.
"""

import sys
import json
import os
import traceback
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

# ── Trace for debugging ──
TRACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_trace.log")
TRACE_ENABLED = os.environ.get("MCP_TRACE", "0") == "1"
def trace(msg):
    if not TRACE_ENABLED:
        return
    with open(TRACE_PATH, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        f.flush()

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use standardized config paths; fall back gracefully if config module unavailable
try:
    from memory_layer.config import get_db_path, load_config, ensure_home_dir
    ensure_home_dir()
    _cfg = load_config()
    _default_db = get_db_path()
    _default_embed = _cfg.get("embeddings", "mode")
    _default_extract = _cfg.getboolean("llm", "extract")
except Exception:
    _default_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cursor_memory.db")
    _default_embed = "local"
    _default_extract = False

DB_PATH = os.environ.get("MEMORY_DB_PATH", _default_db)
EMBEDDING_MODE = os.environ.get("MEMORY_EMBEDDING_MODE", _default_embed)
LLM_EXTRACT = os.environ.get("MEMORY_LLM_EXTRACT", "1" if _default_extract else "0") == "1"
brain = None
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
        "description": "Store something in persistent memory about the user, their preferences, project details, or anything useful for future conversations. If similar information already exists, the old memory is automatically replaced with the new version.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember"},
                "importance": {"type": "number", "description": "0.0-1.0, default 0.7"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                "namespace": {"type": "string", "description": "Memory namespace for isolation (default: 'default')"}
            },
            "required": ["content"]
        }
    },
    {
        "name": "memory_recall",
        "description": "Search persistent memory for relevant information about the user, their preferences, past decisions, or project context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "top_k": {"type": "integer", "description": "Max results, default 5"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
                "namespace": {"type": "string", "description": "Memory namespace (default: 'default')"},
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
    from memory_layer import MemoryType
    b = get_brain()
    ns = str(args.get("namespace", "default")).strip() or "default"

    if name == "memory_remember":
        content = str(args.get("content", "")).strip()
        if not content:
            raise ValueError("content is required")
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError(f"content too long (max {MAX_CONTENT_CHARS} chars)")
        importance = float(args.get("importance", 0.7))
        importance = max(0.0, min(1.0, importance))
        tags = _parse_tags(args)

        m = b.remember(content=content, memory_type=MemoryType.SEMANTIC,
                       importance=importance, tags=tags, namespace=ns)
        r = f'Remembered: "{content}" (importance={m.importance:.2f})'
        if m.metadata.get("replaces"):
            r += f"\nAuto-updated: replaced {len(m.metadata['replaces'])} outdated memory(ies) with this new version."
        if m.metadata.get("contradicts"):
            r += f"\nNote: conflicts with {len(m.metadata['contradicts'])} existing memory(ies)."
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
        results = b.recall(query, top_k=effective_top_k, min_strength=0.05,
                           tags=recall_tags, namespace=ns,
                           reasoning=reasoning, include_history=include_history)

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
        ok = b.forget_memory(memory_id, hard=hard)
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
        b.record_episode(
            user_message=user_msg,
            assistant_response=assistant_msg or summary,
            importance=0.6,
            tags=tags,
            namespace=ns,
        )
        return "Episode recorded."

    elif name == "memory_stats":
        s = b.get_stats()
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
            m = b.remember(
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
            m = b.remember(
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
        report = b.health_check()
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
        results = b.maintenance()
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
        path = b.backup()
        return f"Backup created: {path}"

    elif name == "memory_synthesize":
        topic = str(args.get("topic", "")).strip()
        if not topic:
            raise ValueError("topic is required")
        store = bool(args.get("store_result", False))
        result = b.synthesize(topic=topic, store_result=store, namespace=ns)
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
            "serverInfo": {"name": "memory-layer", "version": "0.4.0"}
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
