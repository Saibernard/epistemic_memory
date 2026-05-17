#!/usr/bin/env python3
"""
MCP vs CLI Latency Benchmark
=============================
Measures end-to-end latency for every memory operation through both
the MCP (stdio JSON-RPC) and CLI (subprocess) interfaces.

MCP: Spawns mcp_server.py as a subprocess, sends JSON-RPC over stdin,
     reads responses from stdout — same as Cursor does.

CLI: Runs `python3 -m memory_layer.cli <command>` as a subprocess —
     same as a user typing in a terminal.

Results are printed as a comparison table and saved to a JSON file.

Usage:
    python3 tests/benchmarks/mcp_vs_cli_latency.py
"""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MCP_SERVER = str(PROJECT_ROOT / "mcp_server.py")
PYTHON = sys.executable

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────
# MCP Client — talks to mcp_server.py via stdio
# ─────────────────────────────────────────

class MCPClient:
    def __init__(self):
        env = os.environ.copy()
        env["MEMORY_EMBEDDING_MODE"] = "local"
        env["MEMORY_LLM_EXTRACT"] = "0"
        self.proc = subprocess.Popen(
            [PYTHON, MCP_SERVER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._msg_id = 0
        self._initialize()

    def _send(self, msg):
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        self.proc.stdin.write(line.encode())
        self.proc.stdin.flush()

    def _recv(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise EOFError("MCP server closed stdout")
            line = line.strip()
            if line:
                return json.loads(line)

    def _initialize(self):
        self._msg_id += 1
        self._send({
            "jsonrpc": "2.0", "id": self._msg_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "latency-bench", "version": "1.0"}
            }
        })
        self._recv()  # initialize response
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name, arguments=None):
        self._msg_id += 1
        self._send({
            "jsonrpc": "2.0", "id": self._msg_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}}
        })
        return self._recv()

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ─────────────────────────────────────────
# CLI Runner — runs memory_layer.cli as subprocess
# ─────────────────────────────────────────

def cli_run(args: list) -> tuple:
    """Run a CLI command and return (stdout, stderr, elapsed_ms)."""
    env = os.environ.copy()
    env["MEMORY_EMBEDDING_MODE"] = "local"
    env["MEMORY_LLM_EXTRACT"] = "0"

    t0 = time.perf_counter()
    result = subprocess.run(
        [PYTHON, "-m", "memory_layer.cli"] + args,
        capture_output=True, text=True, env=env,
        cwd=str(PROJECT_ROOT),
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return result.stdout, result.stderr, elapsed


# ─────────────────────────────────────────
# Benchmark Operations
# ─────────────────────────────────────────

def benchmark_mcp(client: MCPClient, operations: list) -> dict:
    """Run operations through MCP and record latencies."""
    results = {}
    for op_name, tool_name, args in operations:
        t0 = time.perf_counter()
        resp = client.call_tool(tool_name, args)
        elapsed = (time.perf_counter() - t0) * 1000
        success = "error" not in resp
        results[op_name] = {
            "latency_ms": round(elapsed, 1),
            "success": success,
        }
        print(f"  MCP  {op_name:25s}  {elapsed:8.1f} ms  {'OK' if success else 'FAIL'}")
    return results


def benchmark_cli(operations: list) -> dict:
    """Run operations through CLI and record latencies."""
    results = {}
    for op_name, cli_args in operations:
        stdout, stderr, elapsed = cli_run(cli_args)
        success = "error" not in stdout.lower() and "Traceback" not in stderr
        results[op_name] = {
            "latency_ms": round(elapsed, 1),
            "success": success,
        }
        print(f"  CLI  {op_name:25s}  {elapsed:8.1f} ms  {'OK' if success else 'FAIL'}")
    return results


def main():
    print("=" * 65)
    print("  MCP vs CLI Latency Benchmark")
    print("=" * 65)

    test_content = f"Benchmark test memory {uuid.uuid4().hex[:8]}"
    test_query = "benchmark test memory"
    test_episode = "User asked about latency benchmarks. Assistant provided comparison."

    # ── Phase 1: MCP benchmark ──
    print("\n[1/3] Starting MCP server...")
    mcp = MCPClient()
    time.sleep(1)

    mcp_ops = [
        ("remember",        "memory_remember",       {"content": test_content, "tags": ["bench"]}),
        ("recall",          "memory_recall",          {"query": test_query, "top_k": 5}),
        ("record_episode",  "memory_record_episode",  {"summary": test_episode, "tags": ["bench"]}),
        ("stats",           "memory_stats",           {}),
        ("health",          "memory_health",          {}),
        ("maintenance",     "memory_maintenance",     {}),
    ]

    print("\n── MCP Latencies ──")
    mcp_results = benchmark_mcp(mcp, mcp_ops)

    # Get memory ID for forget test
    recall_resp = mcp.call_tool("memory_recall", {"query": test_content, "top_k": 1})
    mem_id = None
    try:
        text = recall_resp.get("result", {}).get("content", [{}])[0].get("text", "")
        for line in text.split("\n"):
            if line.strip().startswith("1. ["):
                mem_id = line.split("[")[1].split("]")[0]
                break
    except Exception:
        pass

    if mem_id:
        t0 = time.perf_counter()
        mcp.call_tool("memory_forget", {"memory_id": mem_id, "hard_delete": True})
        elapsed = (time.perf_counter() - t0) * 1000
        mcp_results["forget"] = {"latency_ms": round(elapsed, 1), "success": True}
        print(f"  MCP  {'forget':25s}  {elapsed:8.1f} ms  OK")
    else:
        mcp_results["forget"] = {"latency_ms": -1, "success": False}
        print(f"  MCP  {'forget':25s}  {'N/A':>8s}     SKIP (no ID found)")

    mcp.close()
    print("  MCP server stopped.\n")

    # ── Phase 2: CLI benchmark ──
    # CLI cold start — each command spawns a new process + loads model
    print("── CLI Latencies (cold start per command) ──")

    cli_ops = [
        ("remember",        ["remember", test_content + " cli", "--tags", "bench"]),
        ("recall",          ["recall", test_query]),
        ("stats",           ["stats"]),
        ("health",          ["health"]),
        ("maintenance",     ["maintenance"]),
        ("status",          ["status"]),
    ]

    cli_results = benchmark_cli(cli_ops)

    # ── Phase 3: CLI with warm Python (import-only overhead) ──
    print("\n── CLI Latencies (warm — Python API direct) ──")
    warm_results = benchmark_cli_warm()

    # ── Results ──
    print("\n" + "=" * 65)
    print(f"  {'Operation':<20s} {'MCP (ms)':>10s} {'CLI cold (ms)':>14s} {'CLI warm (ms)':>14s} {'MCP vs warm':>12s}")
    print("-" * 65)

    all_ops = sorted(set(list(mcp_results.keys()) + list(cli_results.keys()) + list(warm_results.keys())))
    for op in all_ops:
        mcp_ms = mcp_results.get(op, {}).get("latency_ms", -1)
        cli_ms = cli_results.get(op, {}).get("latency_ms", -1)
        warm_ms = warm_results.get(op, {}).get("latency_ms", -1)

        mcp_str = f"{mcp_ms:.1f}" if mcp_ms >= 0 else "N/A"
        cli_str = f"{cli_ms:.1f}" if cli_ms >= 0 else "N/A"
        warm_str = f"{warm_ms:.1f}" if warm_ms >= 0 else "N/A"

        if mcp_ms > 0 and warm_ms > 0:
            ratio = f"{mcp_ms / warm_ms:.2f}x"
        else:
            ratio = "-"

        print(f"  {op:<20s} {mcp_str:>10s} {cli_str:>14s} {warm_str:>14s} {ratio:>12s}")

    print("=" * 65)
    print("  MCP = persistent server (no cold start after first call)")
    print("  CLI cold = new process per command (model load each time)")
    print("  CLI warm = Python API calls (no subprocess overhead)")
    print()

    # ── Save results ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "mcp": mcp_results,
        "cli_cold": cli_results,
        "cli_warm": warm_results,
    }

    out_path = RESULTS_DIR / f"mcp_vs_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}")


def benchmark_cli_warm() -> dict:
    """Benchmark the Python API directly — no subprocess, no cold start."""
    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("MEMORY_EMBEDDING_MODE", "local")
    os.environ.setdefault("MEMORY_LLM_EXTRACT", "0")

    from memory_layer import MemoryManager, MemoryType
    from memory_layer.config import get_db_path

    t0 = time.perf_counter()
    brain = MemoryManager(db_path=get_db_path(), embedding_mode="local", llm_extract=False)
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"  WARM {'init (one-time)':25s}  {init_ms:8.1f} ms  OK")

    results = {}
    test_content = f"Warm benchmark test {uuid.uuid4().hex[:8]}"

    ops = [
        ("remember",  lambda: brain.remember(content=test_content, memory_type=MemoryType.SEMANTIC, importance=0.7, tags=["bench"])),
        ("recall",    lambda: brain.recall("benchmark test", top_k=5)),
        ("stats",     lambda: brain.get_stats()),
        ("health",    lambda: brain.health_check()),
        ("maintenance", lambda: brain.maintenance()),
    ]

    for op_name, fn in ops:
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        results[op_name] = {"latency_ms": round(elapsed, 1), "success": True}
        print(f"  WARM {op_name:25s}  {elapsed:8.1f} ms  OK")

    return results


if __name__ == "__main__":
    main()
