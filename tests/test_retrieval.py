#!/usr/bin/env python3
"""
Document Ingestion + Retrieval Comparison Test
================================================

Ingests 3 documents into a fresh memory layer, runs 20 targeted queries,
and compares results across different embedding modes:

  - local   : sentence-transformers all-mpnet-base-v2 (768d, free, private)
  - openai  : OpenAI text-embedding-3-small (1536d, API cost)
  - hybrid  : OpenAI embeddings + cross-encoder reranker

Usage:
    python3 test_retrieval.py                  # local only
    python3 test_retrieval.py local            # local only
    python3 test_retrieval.py openai           # openai only (needs OPENAI_API_KEY)
    python3 test_retrieval.py hybrid           # openai + reranker (needs OPENAI_API_KEY)
    python3 test_retrieval.py all              # run all modes and compare
    python3 test_retrieval.py local openai     # run specific modes
"""

import os
import sys
import time
import json
import tempfile
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DOCS_DIR = os.path.join(SCRIPT_DIR, "test_docs")

# Auto-load .env from project root so API keys persist across runs
_env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip().strip("\"'")
                if _key and _val and _key not in os.environ:
                    os.environ[_key] = _val

# Each test: (query, list_of_expected_substrings — ALL must be found in top-5 results)
RETRIEVAL_TESTS = [
    # ── Company Handbook ──
    ("Who founded Nexora Technologies?",
     ["Amara Okonkwo", "Rajesh Patel"]),
    ("What is Nexora's annual revenue?",
     ["43.7"]),
    ("What programming languages are approved for production at Nexora?",
     ["Rust", "Go"]),
    ("What frontend framework does Nexora use?",
     ["SolidJS"]),
    ("What database does Nexora use for transactional workloads?",
     ["CockroachDB"]),
    ("How many days of PTO do Nexora employees get?",
     ["28"]),
    ("Where is the 2026 company retreat?",
     ["Reykjavik"]),
    ("What is the internal code search engine called?",
     ["Lighthouse"]),
    ("What is Nexora's fine-tuned LLM model called?",
     ["Nexora-7B-Sec"]),
    ("How often do API keys rotate at Nexora?",
     ["72 hours"]),

    # ── Project Phoenix ──
    ("Who is the project lead for Project Phoenix?",
     ["Yuki Tanaka"]),
    ("When is Project Phoenix scheduled for public beta?",
     ["September 15, 2026"]),
    ("How fast is proof generation on iPhone?",
     ["1.8"]),
    ("What proof system does Phoenix use?",
     ["Plonky3"]),
    ("What is the total budget for Project Phoenix?",
     ["2.4 million"]),
    ("Which cloud provider does Phoenix run on?",
     ["GCP"]),
    ("What protocol replaced gRPC in Project Phoenix?",
     ["Phoenix Wire"]),

    # ── Team profiles (CSV — tests the chunk boundary fix) ──
    ("Who designed the Prism observability platform?",
     ["Rivera"]),
    ("Who discovered the OpenSSL CVE?",
     ["Aisha Nakamura"]),
    ("Who created the Nexora-7B-Sec ML model?",
     ["Wei Zhang"]),
]


MODE_CONFIGS = {
    "local": {
        "label": "Local (mpnet-v2, 768d)",
        "embedding_mode": "local",
        "reranker": "none",
        "enrichment": "none",
    },
    "local_rerank": {
        "label": "Local + Cross-Encoder Reranker",
        "embedding_mode": "local",
        "reranker": "auto",
        "enrichment": "none",
    },
    "openai": {
        "label": "OpenAI (embed-3-small, 1536d)",
        "embedding_mode": "openai",
        "reranker": "none",
        "enrichment": "none",
    },
    "hybrid": {
        "label": "Hybrid (OpenAI + cross-encoder reranker)",
        "embedding_mode": "openai",
        "reranker": "auto",
        "enrichment": "none",
    },
    "gemini": {
        "label": "Gemini (embed-001, 3072d)",
        "embedding_mode": "gemini",
        "reranker": "none",
        "enrichment": "none",
    },
    "gemini_rerank": {
        "label": "Gemini + Cross-Encoder Reranker",
        "embedding_mode": "gemini",
        "reranker": "auto",
        "enrichment": "none",
    },
}


def run_single_mode(mode_name: str, config: dict):
    """Ingest docs and run all queries for a single embedding mode. Returns results dict."""
    tmp_dir = tempfile.mkdtemp(prefix=f"mem_test_{mode_name}_")
    db_path = os.path.join(tmp_dir, "test.db")

    os.environ["MEMORY_ENRICHMENT_BACKEND"] = config["enrichment"]
    os.environ["MEMORY_RERANKER"] = config["reranker"]

    try:
        from memory_layer import MemoryManager, MemoryType
        from memory_layer.document_ingest import DocumentIngestor

        brain = MemoryManager(
            db_path=db_path,
            embedding_mode=config["embedding_mode"],
        )

        ingestor = DocumentIngestor()

        doc_files = [
            os.path.join(DOCS_DIR, "company_handbook.md"),
            os.path.join(DOCS_DIR, "project_phoenix.md"),
            os.path.join(DOCS_DIR, "team_profiles.csv"),
        ]

        t_ingest_start = time.time()
        total_chunks = 0
        for fpath in doc_files:
            chunks = ingestor.extract_and_chunk(fpath)
            for chunk in chunks:
                brain.remember(
                    content=chunk["content"],
                    memory_type=MemoryType.SEMANTIC,
                    importance=0.7,
                    tags=chunk["tags"],
                    metadata=chunk["metadata"],
                )
            total_chunks += len(chunks)
        ingest_time = time.time() - t_ingest_start

        query_results = []
        for query, expected in RETRIEVAL_TESTS:
            t0 = time.time()
            results = brain.recall(query, top_k=5, min_confidence=0.0, min_strength=0.0)
            latency_ms = (time.time() - t0) * 1000

            all_content = " ".join(r.memory.content for r in results).lower()

            found = [s for s in expected if s.lower() in all_content]
            missing = [s for s in expected if s.lower() not in all_content]
            passed = len(missing) == 0 and len(results) > 0
            top_score = results[0].composite_score if results else 0.0

            query_results.append({
                "query": query,
                "passed": passed,
                "found": found,
                "missing": missing,
                "top_score": top_score,
                "latency_ms": latency_ms,
                "num_results": len(results),
            })

        return {
            "mode": mode_name,
            "label": config["label"],
            "total_chunks": total_chunks,
            "ingest_time": ingest_time,
            "queries": query_results,
            "passed": sum(1 for q in query_results if q["passed"]),
            "total": len(query_results),
            "avg_latency_ms": sum(q["latency_ms"] for q in query_results) / len(query_results),
            "avg_top_score": sum(q["top_score"] for q in query_results) / len(query_results),
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def print_single_result(result: dict):
    """Print detailed results for a single mode."""
    label = result["label"]
    print(f"\n{'─' * 64}")
    print(f"  {label}")
    print(f"  Ingested {result['total_chunks']} chunks in {result['ingest_time']:.1f}s")
    print(f"{'─' * 64}")

    for i, q in enumerate(result["queries"], 1):
        status = "\033[92mPASS\033[0m" if q["passed"] else "\033[91mFAIL\033[0m"
        print(f"  [{status}] Q{i:02d} ({q['latency_ms']:5.0f}ms, score={q['top_score']:.3f}): {q['query']}")
        if q["missing"]:
            print(f"         Missing: {q['missing']}")

    pct = result["passed"] / result["total"] * 100
    color = "\033[92m" if pct >= 90 else "\033[93m" if pct >= 70 else "\033[91m"
    print(f"\n  Score: {color}{result['passed']}/{result['total']} ({pct:.0f}%)\033[0m"
          f"  |  Avg latency: {result['avg_latency_ms']:.0f}ms"
          f"  |  Avg score: {result['avg_top_score']:.3f}")


def print_comparison(all_results: list):
    """Print a side-by-side comparison table."""
    print(f"\n{'═' * 80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'═' * 80}\n")

    # Header
    modes = [r["mode"] for r in all_results]
    header = f"  {'Metric':<28}"
    for r in all_results:
        header += f"  {r['mode']:>16}"
    print(header)
    print(f"  {'─' * 28}" + "".join(f"  {'─' * 16}" for _ in all_results))

    # Accuracy
    row = f"  {'Accuracy':<28}"
    for r in all_results:
        pct = r["passed"] / r["total"] * 100
        row += f"  {r['passed']}/{r['total']} ({pct:.0f}%){' ':>4}"
    print(row)

    # Avg composite score
    row = f"  {'Avg composite score':<28}"
    for r in all_results:
        row += f"  {r['avg_top_score']:>16.3f}"
    print(row)

    # Avg latency
    row = f"  {'Avg query latency':<28}"
    for r in all_results:
        row += f"  {r['avg_latency_ms']:>13.0f}ms"
    print(row)

    # Ingest time
    row = f"  {'Ingest time':<28}"
    for r in all_results:
        row += f"  {r['ingest_time']:>14.1f}s"
    print(row)

    # Chunks
    row = f"  {'Chunks created':<28}"
    for r in all_results:
        row += f"  {r['total_chunks']:>16}"
    print(row)

    # Per-query comparison
    print(f"\n  Per-query breakdown:")
    print(f"  {'#':<4} {'Query':<45}", end="")
    for r in all_results:
        print(f" {r['mode']:>8}", end="")
    print()
    print(f"  {'─' * 4} {'─' * 45}", end="")
    for _ in all_results:
        print(f" {'─' * 8}", end="")
    print()

    for i in range(len(RETRIEVAL_TESTS)):
        query_short = RETRIEVAL_TESTS[i][0][:43]
        print(f"  Q{i+1:<3} {query_short:<45}", end="")
        for r in all_results:
            q = r["queries"][i]
            if q["passed"]:
                print(f"  \033[92m{q['top_score']:.3f}\033[0m", end="")
            else:
                print(f"  \033[91m FAIL \033[0m", end="")
        print()

    # Winner
    if len(all_results) > 1:
        best = max(all_results, key=lambda r: (r["passed"], r["avg_top_score"]))
        print(f"\n  Winner: \033[1m{best['label']}\033[0m "
              f"({best['passed']}/{best['total']}, avg score {best['avg_top_score']:.3f})")

    print()


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else ["local"]

    if "all" in args:
        modes_to_run = ["local", "local_rerank", "openai", "hybrid", "gemini", "gemini_rerank"]
    else:
        modes_to_run = [a.lower() for a in args if a.lower() in MODE_CONFIGS]
        if not modes_to_run:
            modes_to_run = ["local"]

    needs_openai = any(m in ("openai", "hybrid") for m in modes_to_run)
    if needs_openai and not os.environ.get("OPENAI_API_KEY"):
        print("\033[91mError: OPENAI_API_KEY required for openai/hybrid modes.\033[0m")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        return 1

    needs_gemini = any(m in ("gemini", "gemini_rerank") for m in modes_to_run)
    if needs_gemini and not os.environ.get("GOOGLE_API_KEY"):
        print("\033[91mError: GOOGLE_API_KEY required for gemini modes.\033[0m")
        print("Set it with: export GOOGLE_API_KEY=AIza...")
        return 1

    print("=" * 80)
    print("  DOCUMENT INGESTION & RETRIEVAL — EMBEDDING COMPARISON")
    print(f"  Modes: {', '.join(modes_to_run)}")
    print(f"  Documents: company_handbook.md, project_phoenix.md, team_profiles.csv")
    print(f"  Queries: {len(RETRIEVAL_TESTS)}")
    print("=" * 80)

    all_results = []
    for mode_name in modes_to_run:
        config = MODE_CONFIGS[mode_name]
        print(f"\n  Running: {config['label']}...")
        result = run_single_mode(mode_name, config)
        all_results.append(result)
        print_single_result(result)

    if len(all_results) > 1:
        print_comparison(all_results)

    worst_pct = min(r["passed"] / r["total"] * 100 for r in all_results)
    return 0 if worst_pct >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
