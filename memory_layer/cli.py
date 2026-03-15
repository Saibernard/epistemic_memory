#!/usr/bin/env python3
"""
Unified CLI for Memory Layer.

Usage:
    memory-layer init                       Set up ~/.memory-layer/
    memory-layer serve                      Start REST API server
    memory-layer mcp                        Start MCP server (for Cursor)
    memory-layer remember "content"         Store a memory
    memory-layer recall "query"             Search memories
    memory-layer forget <memory_id>         Forget a memory
    memory-layer stats                      Show memory statistics
    memory-layer status                     Show system status
"""

import argparse
import json
import os
import sys


def _ensure_init():
    """Auto-init on first use if ~/.memory-layer/ doesn't exist."""
    from .config import get_home_dir, init_memory_layer
    if not get_home_dir().exists():
        print("First run detected — initializing Memory Layer...\n")
        init_memory_layer(verbose=True)
        print()


def _get_brain(namespace: str = "default"):
    """Create a MemoryManager using the standard config."""
    _ensure_init()
    from .config import get_db_path, load_config
    from .storage_factory import create_storage
    config = load_config()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    backend = config.get("storage", "backend", fallback="sqlite")
    storage = create_storage(backend, sqlite_path=get_db_path())

    from .core import MemoryManager
    return MemoryManager(
        db_path=get_db_path(),
        embedding_mode=config.get("embeddings", "mode"),
        embedding_model=config.get("embeddings", "model"),
        llm_extract=config.getboolean("llm", "extract"),
        llm_extract_model=config.get("llm", "model"),
        default_namespace=namespace,
        storage=storage,
    )


# ─────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────

def cmd_init(args):
    from .config import init_memory_layer
    print()
    init_memory_layer(embedding_mode=args.embedding_mode, verbose=True)
    print()
    print("  Done! You can now use:")
    print("    memory-layer serve       # Start REST API")
    print("    memory-layer mcp         # Start MCP server for Cursor")
    print('    memory-layer remember "User likes Python"')
    print('    memory-layer recall "programming preferences"')
    print()


def cmd_serve(args):
    _ensure_init()
    from .config import get_db_path, load_config
    config = load_config()

    host = args.host or config.get("server", "host")
    port = args.port or int(config.get("server", "port"))
    db_path = get_db_path()

    os.environ["MEMORY_DB_PATH"] = db_path
    os.environ["MEMORY_EMBEDDING_MODE"] = config.get("embeddings", "mode")
    os.environ["MEMORY_EMBEDDING_MODEL"] = config.get("embeddings", "model")
    if config.getboolean("llm", "extract"):
        os.environ["MEMORY_LLM_EXTRACT"] = "1"

    embed_label = f"{config.get('embeddings', 'mode')} ({config.get('embeddings', 'model')})"
    extract_label = "enabled" if config.getboolean("llm", "extract") else "disabled"

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║            MEMORY LAYER FOR AI               ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Database  : {db_path:<32}║")
    print(f"  ║  Embedding : {embed_label:<32}║")
    print(f"  ║  LLM Extract: {extract_label:<30}║")
    print(f"  ║  Server    : http://{host}:{port:<15}║")
    print(f"  ║  Docs      : http://{host}:{port}/docs{'':>8}║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    import uvicorn
    uvicorn.run(
        "memory_layer.api:app",
        host=host,
        port=port,
        reload=args.reload,
    )


def cmd_mcp(args):
    _ensure_init()
    from .config import get_db_path, load_config
    config = load_config()

    os.environ["MEMORY_DB_PATH"] = get_db_path()
    os.environ["MEMORY_EMBEDDING_MODE"] = config.get("embeddings", "mode")
    os.environ["MEMORY_EMBEDDING_MODEL"] = config.get("embeddings", "model")
    if config.getboolean("llm", "extract"):
        os.environ["MEMORY_LLM_EXTRACT"] = "1"

    # Import and run the MCP server main loop
    # mcp_server.py is at the project root; use the installed package entry point
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mcp_path = os.path.join(parent_dir, "mcp_server.py")

    if os.path.exists(mcp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("mcp_server", mcp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    else:
        print("Error: mcp_server.py not found.", file=sys.stderr)
        print("If installed via pip, use: python -m memory_layer.mcp", file=sys.stderr)
        sys.exit(1)


def cmd_remember(args):
    brain = _get_brain(args.namespace)
    from .models import MemoryType

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    memory = brain.remember(
        content=args.content,
        memory_type=MemoryType.SEMANTIC,
        importance=args.importance,
        tags=tags,
        namespace=args.namespace,
    )

    print(f'Remembered: "{args.content}"')
    print(f"  ID:         {memory.id}")
    print(f"  Importance: {memory.importance:.2f}")
    print(f"  Namespace:  {memory.namespace}")
    if memory.metadata.get("replaces"):
        print(f"  Replaced:   {len(memory.metadata['replaces'])} outdated memory(ies)")
    if tags:
        print(f"  Tags:       {', '.join(tags)}")


def cmd_recall(args):
    brain = _get_brain(args.namespace)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    results = brain.recall(
        query=args.query,
        top_k=args.top_k,
        min_strength=0.05,
        tags=tags,
        namespace=args.namespace,
    )

    if not results:
        print(f'No memories found for: "{args.query}"')
        return

    print(f'Found {len(results)} memory(ies) for: "{args.query}"\n')
    for i, r in enumerate(results):
        print(f"  {i+1}. {r.memory.content}")
        print(f"     ID: {r.memory.id}  relevance={r.relevance_score:.3f}  strength={r.effective_strength:.2f}")
        if r.memory.tags:
            print(f"     Tags: {', '.join(r.memory.tags)}")
        print()


def cmd_forget(args):
    brain = _get_brain()
    ok = brain.forget_memory(args.memory_id, hard=args.hard)
    if ok:
        action = "permanently deleted" if args.hard else "deactivated"
        print(f"Memory {args.memory_id} {action}.")
    else:
        print(f"Memory not found: {args.memory_id}")
        sys.exit(1)


def cmd_stats(args):
    brain = _get_brain(args.namespace)
    stats = brain.get_stats(namespace=args.namespace if args.namespace != "default" else None)

    print()
    print("  Memory Layer Statistics")
    print("  " + "─" * 35)
    print(f"  Total memories:   {stats.total_memories}")
    print(f"    Episodic:       {stats.episodic_count}")
    print(f"    Semantic:       {stats.semantic_count}")
    print(f"    Procedural:     {stats.procedural_count}")
    print(f"  Total links:      {stats.total_links}")
    print(f"  Working memory:   {stats.working_memory_size}")
    print(f"  Avg strength:     {stats.avg_strength:.2f}")
    print(f"  Avg importance:   {stats.avg_importance:.2f}")
    print(f"  Consolidations:   {stats.consolidation_count}")
    if stats.oldest_memory_age_hours > 0:
        days = stats.oldest_memory_age_hours / 24
        print(f"  Oldest memory:    {days:.1f} days ago")
    print()


def cmd_health(args):
    brain = _get_brain(args.namespace)
    report = brain.health_check()

    print()
    print("  Memory Layer Health Check")
    print("  " + "─" * 40)
    status = report.get("status", "unknown")
    color = "\033[92m" if status == "ok" else "\033[93m"
    print(f"  Status:        {color}{status}\033[0m")

    db = report.get("database", {})
    print(f"  SQLite:        {'OK' if db.get('sqlite_ok') else 'FAIL'}")
    print(f"  Active:        {db.get('active_memories', '?')}")
    print(f"  Null embeds:   {db.get('null_embeddings', '?')}")
    print(f"  Orphan links:  {db.get('orphaned_links', '?')}")
    print(f"  FAISS synced:  {'yes' if report.get('faiss_synced') else 'no'}")

    issues = report.get("issues", [])
    if issues:
        print(f"  Issues:")
        for issue in issues:
            print(f"    - {issue}")

    storage = report.get("storage", {})
    if storage:
        print(f"  DB size:       {storage.get('db_size_mb', '?')} MB")
    print()


def cmd_maintenance(args):
    brain = _get_brain(args.namespace)
    print("  Running maintenance...")
    results = brain.maintenance()

    print()
    print("  Maintenance Results")
    print("  " + "─" * 40)
    r = results
    consol = r.get("consolidation", {})
    if isinstance(consol, dict) and not consol.get("error"):
        l0 = consol.get("level_0_to_1", {})
        created = l0.get("semantic_memories_created", 0) if isinstance(l0, dict) else 0
        print(f"  Consolidated:  {created} new semantic memories")
    print(f"  Reasoning pruned: {r.get('reasoning_pruned', 0)}")
    print(f"  Queue cleaned:    {r.get('queue_cleaned', 0)}")
    print(f"  FAISS rebuilt:    {r.get('faiss_rebuilt', False)}")
    repair = r.get("repair", {})
    if isinstance(repair, dict):
        total = sum(v for v in repair.values() if isinstance(v, int))
        print(f"  Repaired:         {total} items")
    ss = r.get("storage_stats", {})
    print(f"  Active memories:  {ss.get('active_memories', '?')}")
    print(f"  DB size:          {ss.get('db_size_mb', '?')} MB")
    print()


def cmd_backup(args):
    brain = _get_brain()
    dest = args.output if args.output else None
    path = brain.backup(dest)
    print(f"\n  Backup created: {path}\n")


def cmd_status(args):
    from .config import get_status
    info = get_status()

    print()
    print("  Memory Layer Status")
    print("  " + "─" * 40)
    print(f"  Home dir:    {info['home_dir']}")
    print(f"  Initialized: {'yes' if info['home_exists'] else 'no'}")
    print(f"  Database:    {info['db_path']}")
    print(f"  DB exists:   {'yes' if info['db_exists'] else 'no'}")
    if info["db_exists"]:
        print(f"  DB size:     {info['db_size_mb']} MB")
        print(f"  Memories:    {info.get('total_memories', 'unknown')}")
        print(f"  Links:       {info.get('total_links', 'unknown')}")
    print(f"  Config:      {info['config_path']}")
    print(f"  Config exists: {'yes' if info['config_exists'] else 'no'}")
    print(f"  Models dir:  {info['models_dir']}")
    print()


def cmd_export(args):
    brain = _get_brain(args.namespace)
    from .passport import export_passport

    passphrase = args.encrypt or None
    summary = export_passport(
        storage=brain.storage,
        output_path=args.output,
        namespace=args.namespace if args.namespace != "default" else None,
        include_embeddings=args.include_embeddings,
        passphrase=passphrase,
    )
    print(f"\n  Exported to: {summary['output_path']}")
    print(f"  Format:      Universal Memory Passport v2")
    print(f"  Memories:    {summary['memories_exported']}")
    print(f"  Links:       {summary['links_exported']}")
    print(f"  Namespaces:  {', '.join(summary['namespaces'])}")
    print(f"  Encrypted:   {'yes' if summary['encrypted'] else 'no'}")
    print(f"  File size:   {summary['file_size_mb']} MB\n")


def cmd_import(args):
    brain = _get_brain(args.namespace)
    from .passport import import_passport

    passphrase = args.passphrase or None
    summary = import_passport(
        storage=brain.storage,
        input_path=args.input,
        passphrase=passphrase,
        target_namespace=args.namespace if args.namespace != "default" else None,
        skip_duplicates=not args.allow_duplicates,
        reembed=not args.no_reembed,
        embeddings_engine=brain.embeddings if not args.no_reembed else None,
    )
    detected = summary.get("format", "unknown")
    print(f"\n  Imported from: {args.input}")
    print(f"  Detected format: {detected}")
    print(f"  Memories imported: {summary['memories_imported']}")
    print(f"  Memories skipped:  {summary['memories_skipped']}")
    print(f"  Links imported:    {summary['links_imported']}\n")

    if summary['memories_imported'] > 0:
        print("  Rebuilding FAISS indices...")
        brain._rebuild_faiss_indices()
        brain.memory_index.save()
        brain.passage_index.save()
        print("  Done!\n")


def cmd_passport_inspect(args):
    from .passport import inspect_passport

    passphrase = args.passphrase or None
    info = inspect_passport(args.file, passphrase=passphrase)

    print(f"\n  Memory Passport Inspector")
    print(f"  {'─' * 40}")
    print(f"  File:        {info['path']}")
    print(f"  Format:      {info['format']}")
    print(f"  Size:        {info['file_size_mb']} MB")

    if info.get("passport_version"):
        print(f"  Version:     {info['passport_version']}")
    if info.get("created_at"):
        print(f"  Created:     {info['created_at']}")
    if info.get("encrypted") is not None:
        print(f"  Encrypted:   {'yes' if info['encrypted'] else 'no'}")

    mc = info.get("memory_count", "?")
    print(f"  Memories:    {mc}")
    if info.get("link_count") is not None:
        print(f"  Links:       {info['link_count']}")
    if info.get("namespaces"):
        print(f"  Namespaces:  {', '.join(info['namespaces'])}")
    if info.get("memory_types"):
        types_str = ", ".join(f"{k}={v}" for k, v in info["memory_types"].items())
        print(f"  Types:       {types_str}")
    if info.get("tag_summary"):
        top_tags = list(info["tag_summary"].items())[:10]
        tags_str = ", ".join(f"{k} ({v})" for k, v in top_tags)
        print(f"  Top tags:    {tags_str}")

    if info.get("facts") is not None:
        print(f"  Zep facts:   {info['facts']}")
        print(f"  Zep episodes:{info.get('episodes', 0)}")
        print(f"  Zep entities:{info.get('entities', 0)}")

    print()


def cmd_passport_convert(args):
    from .passport import convert_passport

    summary = convert_passport(
        input_path=args.input,
        output_path=args.output,
        input_passphrase=args.input_passphrase or None,
        output_passphrase=args.output_passphrase or None,
    )
    print(f"\n  Converted passport:")
    print(f"  Input format:  {summary['input_format']}")
    print(f"  Output format: {summary['output_format']}")
    print(f"  Memories:      {summary['memories']}")
    print(f"  Links:         {summary['links']}")
    print(f"  Encrypted:     {'yes' if summary['encrypted'] else 'no'}")
    print(f"  Output:        {summary['output_path']}\n")


def cmd_proxy(args):
    from .proxy import create_proxy_app
    import uvicorn

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║           MEMORY PROXY FOR AI                ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Endpoint : http://{args.host}:{args.port}/v1   ║")
    print(f"  ║  Docs     : http://{args.host}:{args.port}/docs ║")
    print("  ║                                              ║")
    print("  ║  Drop-in replacement for OpenAI API.         ║")
    print("  ║  All calls get automatic memory injection.   ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    app = create_proxy_app(
        db_path=args.db,
        embedding_mode=args.embedding_mode,
        namespace=args.namespace,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_chat(args):
    brain = _get_brain(args.namespace)

    if args.brain:
        from .export import import_brain
        print(f"  Loading brain from: {args.brain}")
        import_brain(
            storage=brain.storage,
            input_path=args.brain,
            reembed=True,
            embeddings_engine=brain.embeddings,
        )
        brain._rebuild_faiss_indices()
        brain.memory_index.save()
        brain.passage_index.save()

    from .chat import ChatEngine

    mode = args.mode
    try:
        engine = ChatEngine(
            brain=brain,
            mode=mode,
            namespace=args.namespace,
            top_k=args.top_k,
        )
    except (ValueError, ImportError) as e:
        print(f"  Error: {e}")
        if mode == "llm":
            print("  Falling back to local mode (no LLM).\n")
            mode = "local"
            engine = ChatEngine(brain=brain, mode="local", namespace=args.namespace, top_k=args.top_k)
        else:
            sys.exit(1)

    if args.web:
        _run_chat_web(brain, engine, args)
        return

    stats = brain.get_stats(namespace=args.namespace if args.namespace != "default" else None)
    print()
    print(f"  Memory Layer Chat ({mode} mode)")
    print(f"  {stats.total_memories} memories loaded")
    print(f"  Type 'quit' to exit.\n")

    while True:
        try:
            question = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        response = engine.ask(question)

        if not response.has_answer:
            print(f"\n  Bot: {response.answer}\n")
        else:
            print(f"\n  Bot: {response.answer}\n")


def _run_chat_web(brain, engine, args):
    """Start the web chat interface."""
    from .config import load_config
    config = load_config()
    host = args.host or config.get("server", "host")
    port = args.port or int(config.get("server", "port"))

    print(f"\n  Chat UI: http://{host}:{port}")
    print(f"  Press Ctrl+C to stop.\n")

    import uvicorn
    from .chat_web import create_chat_app
    app = create_chat_app(brain, engine)
    uvicorn.run(app, host=host, port=port)


# ─────────────────────────────────────────────
# MAIN PARSER
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="memory-layer",
        description="Memory Layer — persistent, evolving memory for AI. Local-first, no cloud required.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    p_init = subparsers.add_parser("init", help="Initialize ~/.memory-layer/ directory and config")
    p_init.add_argument("--embedding-mode", default="local", choices=["local", "openai"],
                        help="Embedding backend (default: local)")
    p_init.set_defaults(func=cmd_init)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the REST API server")
    p_serve.add_argument("--host", default=None, help="Host to bind (default: from config)")
    p_serve.add_argument("--port", type=int, default=None, help="Port (default: from config)")
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload for development")
    p_serve.set_defaults(func=cmd_serve)

    # mcp
    p_mcp = subparsers.add_parser("mcp", help="Start the MCP server (for Cursor/VS Code)")
    p_mcp.set_defaults(func=cmd_mcp)

    # remember
    p_rem = subparsers.add_parser("remember", help="Store a memory")
    p_rem.add_argument("content", help="What to remember")
    p_rem.add_argument("--importance", type=float, default=0.7, help="0.0-1.0 (default: 0.7)")
    p_rem.add_argument("--tags", default="", help="Comma-separated tags")
    p_rem.add_argument("--namespace", default="default", help="Namespace (default: default)")
    p_rem.set_defaults(func=cmd_remember)

    # recall
    p_rec = subparsers.add_parser("recall", help="Search memories")
    p_rec.add_argument("query", help="Natural language query")
    p_rec.add_argument("--top-k", type=int, default=5, help="Max results (default: 5)")
    p_rec.add_argument("--tags", default="", help="Filter by tags (comma-separated)")
    p_rec.add_argument("--namespace", default="default", help="Namespace (default: default)")
    p_rec.set_defaults(func=cmd_recall)

    # forget
    p_forget = subparsers.add_parser("forget", help="Forget/delete a memory")
    p_forget.add_argument("memory_id", help="ID of the memory to forget")
    p_forget.add_argument("--hard", action="store_true", help="Permanently delete (default: soft-delete)")
    p_forget.set_defaults(func=cmd_forget)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show memory statistics")
    p_stats.add_argument("--namespace", default="default", help="Namespace (default: default)")
    p_stats.set_defaults(func=cmd_stats)

    # health
    p_health = subparsers.add_parser("health", help="Run health check (integrity, FAISS sync, issues)")
    p_health.add_argument("--namespace", default="default", help="Namespace (default: default)")
    p_health.set_defaults(func=cmd_health)

    # maintenance
    p_maint = subparsers.add_parser("maintenance", help="Run all maintenance tasks (consolidation, decay, pruning, repair)")
    p_maint.add_argument("--namespace", default="default", help="Namespace (default: default)")
    p_maint.set_defaults(func=cmd_maintenance)

    # backup
    p_backup = subparsers.add_parser("backup", help="Create a database backup")
    p_backup.add_argument("--output", default=None, help="Output path (default: auto-timestamped)")
    p_backup.set_defaults(func=cmd_backup)

    # status
    p_status = subparsers.add_parser("status", help="Show system status (paths, DB info)")
    p_status.set_defaults(func=cmd_status)

    # export
    p_export = subparsers.add_parser("export", help="Export memories as Universal Memory Passport")
    p_export.add_argument("output", help="Output file path (e.g. brain.passport.json)")
    p_export.add_argument("--namespace", default="default", help="Namespace to export (default: all)")
    p_export.add_argument("--include-embeddings", action="store_true",
                          help="Include embedding vectors (larger file)")
    p_export.add_argument("--encrypt", default="", metavar="PASSPHRASE",
                          help="Encrypt the passport with a passphrase (AES-256)")
    p_export.set_defaults(func=cmd_export)

    # import
    p_import = subparsers.add_parser("import", help="Import memories from any supported format")
    p_import.add_argument("input", help="Input file (passport, Mem0, Zep, ChatGPT, Claude)")
    p_import.add_argument("--namespace", default="default", help="Override namespace for imported memories")
    p_import.add_argument("--passphrase", default="", help="Decryption passphrase (if encrypted)")
    p_import.add_argument("--no-reembed", action="store_true",
                          help="Skip re-computing embeddings on import")
    p_import.add_argument("--allow-duplicates", action="store_true",
                          help="Import even if memory ID already exists")
    p_import.set_defaults(func=cmd_import)

    # passport inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a passport file without importing")
    p_inspect.add_argument("file", help="Path to passport/export file")
    p_inspect.add_argument("--passphrase", default="", help="Decryption passphrase (if encrypted)")
    p_inspect.set_defaults(func=cmd_passport_inspect)

    # passport convert
    p_convert = subparsers.add_parser("convert", help="Convert between passport formats or encrypt/decrypt")
    p_convert.add_argument("input", help="Input passport file")
    p_convert.add_argument("output", help="Output passport file")
    p_convert.add_argument("--input-passphrase", default="", help="Passphrase for encrypted input")
    p_convert.add_argument("--output-passphrase", default="", help="Passphrase to encrypt output")
    p_convert.set_defaults(func=cmd_passport_convert)

    # proxy
    p_proxy = subparsers.add_parser("proxy", help="Start the Memory Proxy (OpenAI-compatible API with memory)")
    p_proxy.add_argument("--host", default="127.0.0.1", help="Host to bind")
    p_proxy.add_argument("--port", type=int, default=8585, help="Port (default: 8585)")
    p_proxy.add_argument("--db", default=None, help="Database path")
    p_proxy.add_argument("--embedding-mode", default=None, help="Embedding backend")
    p_proxy.add_argument("--namespace", default="default", help="Default namespace")
    p_proxy.set_defaults(func=cmd_proxy)

    # chat
    p_chat = subparsers.add_parser("chat", help="Chat with the memory graph")
    p_chat.add_argument("--brain", default=None, help="Load a brain.json file before chatting")
    p_chat.add_argument("--mode", default="local", choices=["local", "llm"],
                        help="Chat mode: local (no LLM) or llm (OpenAI)")
    p_chat.add_argument("--namespace", default="default", help="Namespace to search")
    p_chat.add_argument("--top-k", type=int, default=8, help="Max memories per question (default: 8)")
    p_chat.add_argument("--web", action="store_true", help="Launch web chat UI instead of terminal")
    p_chat.add_argument("--host", default=None, help="Host for web UI")
    p_chat.add_argument("--port", type=int, default=None, help="Port for web UI")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
