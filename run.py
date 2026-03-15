#!/usr/bin/env python3
"""
Memory Layer - Start the REST API server.

DEPRECATED: Use `memory-layer serve` instead.
This file is kept for backwards compatibility.

Usage:
    python run.py                          # Start with defaults
    python run.py --port 9000              # Custom port
    memory-layer serve                     # Preferred (after pip install)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Memory Layer - Biologically-inspired memory for AI"
    )
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    parser.add_argument("--model", default=None, help="Embedding model name")
    parser.add_argument(
        "--embedding-mode", default=None, choices=["local", "openai", "gemini"],
        help="Embedding backend: local, openai, or gemini",
    )
    parser.add_argument("--llm-extract", action="store_true", help="Enable LLM fact extraction")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Load standardized config as base, override with CLI args
    try:
        from memory_layer.config import get_db_path, load_config, ensure_home_dir
        ensure_home_dir()
        config = load_config()
        db_path = args.db or get_db_path()
        host = args.host or config.get("server", "host")
        port = args.port or int(config.get("server", "port"))
        embed_mode = args.embedding_mode or config.get("embeddings", "mode")
        embed_model = args.model or config.get("embeddings", "model")
        llm_extract = args.llm_extract or config.getboolean("llm", "extract")
    except Exception:
        db_path = args.db or "memory.db"
        host = args.host or "127.0.0.1"
        port = args.port or 8484
        embed_mode = args.embedding_mode or "local"
        embed_model = args.model or "all-mpnet-base-v2"
        llm_extract = args.llm_extract

    production_mode = os.environ.get("MEMORY_PRODUCTION_MODE", "0") == "1"
    if host not in {"127.0.0.1", "localhost"} and not os.environ.get("MEMORY_API_KEY"):
        msg = "Exposing Memory API without MEMORY_API_KEY set. This is unsafe for production."
        if production_mode:
            raise SystemExit(f"Refusing to start in production mode: {msg}")
        print(f"  WARNING: {msg}")

    os.environ["MEMORY_DB_PATH"] = db_path
    os.environ["MEMORY_EMBEDDING_MODEL"] = embed_model
    os.environ["MEMORY_EMBEDDING_MODE"] = embed_mode
    if llm_extract:
        os.environ["MEMORY_LLM_EXTRACT"] = "1"

    embed_label = f"{embed_mode} ({embed_model})"
    extract_label = "enabled" if llm_extract else "disabled"

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
    uvicorn.run("memory_layer.api:app", host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
