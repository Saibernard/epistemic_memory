"""Tests for the warm-daemon architecture (v2).

Covers:
  - enrichment backend policy: cloud requires explicit opt-in
  - recall hot path is LLM-free
  - daemon_matches_db routing guard
  - live daemon integration: spawn -> health -> remember -> recall ->
    episode -> shutdown (one daemon reused across the class)
  - hook capture filters (framework text never captured)
  - episode distillation with a fake local LLM
"""

import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── enrichment policy ─────────────────────────────────────────────────

def test_auto_enrichment_never_uses_cloud(monkeypatch):
    """A stray GOOGLE_API_KEY must NOT silently enable a paid backend."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-should-not-be-used")
    from memory_layer.enrichment import EnrichmentPipeline
    # Dead base_url so a locally-running Ollama can't satisfy "auto"
    p = EnrichmentPipeline(backend="auto", base_url="http://127.0.0.1:1")
    assert not p.backend_name.startswith("gemini")
    assert p.backend_name in ("regex",) or p.backend_name.startswith(
        "openai_compat"
    )


def test_recall_query_expansion_is_llm_free(brain):
    """_expand_query must never call the enrichment LLM (hot path)."""
    calls = []

    class _SpyLLM:
        def generate(self, *a, **k):
            calls.append(1)
            return "should never be called"

    brain.enrichment._llm = _SpyLLM()
    alts = brain._expand_query("what database does this project use")
    assert isinstance(alts, list) and alts
    assert calls == []


# ── routing guard ─────────────────────────────────────────────────────

def test_daemon_matches_db_guard(tmp_path, monkeypatch):
    from memory_layer import daemon as d
    monkeypatch.delenv("MEMORY_DAEMON_URL", raising=False)
    # A random other DB must not match the configured default
    assert d.daemon_matches_db(str(tmp_path / "other.db")) is False
    # Explicit MEMORY_DAEMON_URL means the operator chose the daemon
    monkeypatch.setenv("MEMORY_DAEMON_URL", "http://127.0.0.1:9")
    assert d.daemon_matches_db(str(tmp_path / "other.db")) is True


# ── live daemon integration ───────────────────────────────────────────

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def live_daemon(tmp_path_factory):
    """Spawn one real daemon on a temp DB + free port for these tests."""
    workdir = tmp_path_factory.mktemp("daemon")
    db = str(workdir / "daemon_test.db")
    port = _free_port()
    env = {
        **os.environ,
        "MEMORY_DB_PATH": db,
        "PYTHONPATH": REPO,
        "MEMORY_ENRICHMENT_BACKEND": "none",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "memory_layer", "serve",
         "--host", "127.0.0.1", "--port", str(port)],
        env=env, cwd=str(workdir),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    up = False
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url + "/health", timeout=1).read()
            up = True
            break
        except Exception:
            if proc.poll() is not None:
                break
            time.sleep(0.5)
    if not up:
        proc.terminate()
        pytest.skip("could not start daemon (no model/uvicorn here)")
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_daemon_roundtrip(live_daemon, monkeypatch):
    monkeypatch.setenv("MEMORY_DAEMON_URL", live_daemon)
    from memory_layer.daemon import MemoryClient
    c = MemoryClient()
    assert c.is_up()

    m = c.remember(
        "The staging cluster runs Kubernetes 1.31 on GKE",
        importance=0.8, namespace="proj-x",
    )
    assert m.get("id")

    rows = c.recall("what kubernetes version is staging on",
                    top_k=3, namespace="proj-x")
    assert rows and "1.31" in rows[0]["memory"]["content"]

    ep = c.record_episode(
        "please bump the staging cluster version",
        "Bumped staging to Kubernetes 1.31 and verified all pods healthy",
        namespace="proj-x",
    )
    assert ep.get("id")

    # namespace isolation holds over HTTP too
    other = c.recall("kubernetes staging version", top_k=3, namespace="proj-y")
    assert all("1.31" not in r["memory"]["content"] for r in other)


def test_daemon_latency_budget(live_daemon, monkeypatch):
    """Warm recall must be interactive: well under 500ms."""
    monkeypatch.setenv("MEMORY_DAEMON_URL", live_daemon)
    from memory_layer.daemon import MemoryClient
    c = MemoryClient()
    c.recall("warmup", top_k=1, namespace="proj-x")
    t0 = time.time()
    for _ in range(3):
        c.recall("kubernetes staging cluster", top_k=3, namespace="proj-x")
    avg_ms = (time.time() - t0) / 3 * 1000
    assert avg_ms < 500, f"warm recall too slow: {avg_ms:.0f}ms"


# ── hook capture filters ──────────────────────────────────────────────

def _run_record_turn_on(user_text, transcript_dir):
    from memory_layer import hooks
    path = os.path.join(str(transcript_dir), "t.jsonl")
    with open(path, "w") as fh:
        fh.write(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": user_text},
        }) + "\n")
        fh.write(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": "Did the thing successfully with details."}]},
        }) + "\n")
    stdin = io.StringIO(json.dumps({"transcript_path": path}))
    old = sys.stdin
    sys.stdin = stdin
    try:
        return hooks.record_turn()
    finally:
        sys.stdin = old


@pytest.mark.parametrize("junk", [
    "# Update Config Skill\n\nModify Claude Code configuration...",
    "<command-name>/goal</command-name> something",
    "<system-reminder>injected</system-reminder> plus more text here",
    "Caveat: The messages below were generated while running local commands",
])
def test_record_turn_skips_framework_text(junk, tmp_path, monkeypatch):
    # Point at a temp DB: if the filter failed, capture would need the
    # model — instead the filter must return before any heavy work.
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "cap.db"))
    monkeypatch.delenv("MEMORY_DAEMON_URL", raising=False)
    rc = _run_record_turn_on(junk, tmp_path)
    assert rc == 0
    assert not os.path.exists(str(tmp_path / "cap.db"))


def test_record_turn_skips_trivial(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "cap.db"))
    rc = _run_record_turn_on("ok thanks", tmp_path)
    assert rc == 0
    assert not os.path.exists(str(tmp_path / "cap.db"))


# ── distillation ──────────────────────────────────────────────────────

def test_record_episode_distills_with_local_llm(brain):
    class _FakeLLM:
        def generate(self, prompt, max_tokens=200):
            return ("The project database is CockroachDB 24.1\n"
                    "User prefers squash-merge for all PRs\n")

    brain.enrichment._llm = _FakeLLM()
    brain.record_episode(
        user_message="set up the database layer and merge strategy please",
        assistant_response="Configured CockroachDB 24.1 and squash-merge.",
        namespace="distill-test",
    )
    results = brain.recall(
        "which database does the project use",
        namespace="distill-test", top_k=5,
    )
    distilled = [
        r for r in results
        if "distilled" in (r.memory.tags or [])
    ]
    assert distilled, "expected distilled semantic facts to be stored"


def test_record_episode_no_llm_no_distill(brain):
    brain.enrichment._llm = None
    ep = brain.record_episode(
        user_message="just a plain turn without any llm available here",
        assistant_response="plain response, nothing distilled",
        namespace="distill-none",
    )
    assert ep is not None
    results = brain.recall("plain turn", namespace="distill-none", top_k=5)
    assert all("distilled" not in (r.memory.tags or []) for r in results)
