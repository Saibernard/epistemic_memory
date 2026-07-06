#!/usr/bin/env python3
"""
Reproducible experiments for the paper:
"A Local, LLM-Free, Self-Auditing Memory Layer for Personal AI."

Everything here runs fully on-device: local embeddings (all-mpnet-base-v2),
local cross-encoder reranker, NO LLM enrichment, NO network. Produces the
numbers and figures cited in the paper.

    python3 paper/experiments.py

Outputs:
    paper/results.json          all measured numbers
    paper/figures/*.png         figures
"""

import os
# Force a fully-local, LLM-free configuration BEFORE importing the package.
os.environ["MEMORY_ENRICHMENT_BACKEND"] = "none"
os.environ["MEMORY_LLM_EXTRACT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import random
import statistics
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memory_layer import MemoryManager
from memory_layer.models import Memory, MemoryType, MemoryLink, LinkType
from memory_layer.core import (
    W_CONFIDENCE_FLOOR, _EPISTEMIC_STATUS_PENALTY,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
RESULTS = {}


def _fresh_brain():
    tmp = tempfile.mkdtemp(prefix="paper_exp_")
    return MemoryManager(
        db_path=os.path.join(tmp, "mem.db"),
        embedding_mode="local",
        enrichment_backend="none",
        llm_extract=False,
    ), tmp


# ──────────────────────────────────────────────────────────────────────────
# Experiment 1 — Forgetting curve (biologically-inspired decay), LLM-free
# ──────────────────────────────────────────────────────────────────────────
def exp_forgetting_curve(brain):
    print("\n[1] Forgetting curve ...")
    engine = brain.decay_engine
    now = time.time()
    days = np.linspace(0, 30, 60)

    def curve(memory_type, importance, access_count):
        ys = []
        for d in days:
            m = Memory(
                content="x",
                memory_type=memory_type,
                importance=importance,
                strength=1.0,
                access_count=access_count,
                last_accessed=now - d * 86400.0,
            )
            ys.append(engine.compute_current_strength(m))
        return ys

    series = {
        "episodic (imp=0.5, recalls=0)": curve(MemoryType.EPISODIC, 0.5, 0),
        "semantic (imp=0.5, recalls=0)": curve(MemoryType.SEMANTIC, 0.5, 0),
        "procedural (imp=0.5, recalls=0)": curve(MemoryType.PROCEDURAL, 0.5, 0),
        "episodic (imp=0.5, recalls=5)": curve(MemoryType.EPISODIC, 0.5, 5),
    }

    plt.figure(figsize=(7, 4.5))
    for label, ys in series.items():
        plt.plot(days, ys, label=label, linewidth=2)
    plt.xlabel("Days since last recall")
    plt.ylabel("Retained strength")
    plt.title("Ebbinghaus forgetting curve — type-aware, recall-reinforced (LLM-free)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_forgetting.png", dpi=150)
    plt.close()

    RESULTS["forgetting_curve"] = {
        "days": days.tolist(),
        "series": {k: [round(v, 4) for v in ys] for k, ys in series.items()},
        "note": "type stability multipliers: episodic 1x, semantic 3x, procedural 5x; recall raises stability via ln(2+access_count)",
    }
    print("    saved fig1_forgetting.png")


# ──────────────────────────────────────────────────────────────────────────
# Experiment 2 — Confidence/epistemic status changes retrieval ranking
# ──────────────────────────────────────────────────────────────────────────
def exp_confidence_ranking(brain):
    print("\n[2] Confidence-aware retrieval ...")
    content = "The user's production database is PostgreSQL 16 on AWS RDS."
    m = brain.remember(content, tags=["paper-exp2"])

    confs = np.linspace(0.0, 1.0, 11)
    statuses = ["verified", "inferred", "uncertain", "contradicted"]
    measured = {s: [] for s in statuses}

    base = None
    for s in statuses:
        for c in confs:
            mm = brain.storage.get_memory(m.id)
            mm.confidence = float(c)
            mm.epistemic_status = s
            brain.storage.update_memory(mm)
            res = brain.recall(content, top_k=5, use_epistemic=True)
            score = next((r.composite_score for r in res if r.memory.id == m.id), 0.0)
            measured[s].append(score)
        if base is None:
            # baseline: same memory, modifier OFF (relevance-only composite)
            res0 = brain.recall(content, top_k=5, use_epistemic=False)
            base = next((r.composite_score for r in res0 if r.memory.id == m.id), 0.0)

    plt.figure(figsize=(7, 4.5))
    for s in statuses:
        plt.plot(confs, measured[s], marker="o", label=f"status={s}")
    plt.axhline(base, ls="--", color="gray", label="modifier OFF (relevance only)")
    plt.xlabel("Stored confidence")
    plt.ylabel("Composite retrieval score")
    plt.title("Stored reliability changes ranking (read-path), not just metadata")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_confidence_ranking.png", dpi=150)
    plt.close()

    # theoretical modifier table
    mod_table = {}
    for s in statuses:
        mod_table[s] = {
            f"conf={c:.1f}": round(
                (W_CONFIDENCE_FLOOR + (1 - W_CONFIDENCE_FLOOR) * c)
                * _EPISTEMIC_STATUS_PENALTY[s], 3)
            for c in [0.0, 0.5, 1.0]
        }
    RESULTS["confidence_ranking"] = {
        "baseline_modifier_off": round(base, 4),
        "measured_composite": {s: [round(v, 4) for v in measured[s]] for s in statuses},
        "modifier_formula": "(W_FLOOR + (1-W_FLOOR)*conf) * status_penalty",
        "W_CONFIDENCE_FLOOR": W_CONFIDENCE_FLOOR,
        "status_penalty": _EPISTEMIC_STATUS_PENALTY,
        "modifier_table": mod_table,
    }
    print(f"    baseline(off)={base:.4f}; saved fig2_confidence_ranking.png")


# ──────────────────────────────────────────────────────────────────────────
# Experiment 3 — End-to-end contradiction -> contradicted status + demotion
# ──────────────────────────────────────────────────────────────────────────
def exp_contradiction_flow():
    print("\n[3] End-to-end contradiction flow ...")
    brain, _ = _fresh_brain()
    v1 = brain.remember("The project deadline is March 15, 2026.", tags=["paper-exp3"])
    v2 = brain.remember("The project deadline is April 20, 2026.", tags=["paper-exp3"])

    old = brain.storage.get_memory(v1.id)
    new = brain.storage.get_memory(v2.id)
    res = brain.recall("When is the project deadline?", top_k=5)
    returned_ids = [r.memory.id for r in res]

    out = {
        "old_epistemic_status": old.epistemic_status,
        "old_confidence": round(old.confidence, 3),
        "old_is_current": old.is_current,
        "new_epistemic_status": new.epistemic_status,
        "new_is_current": new.is_current,
        "recall_returns_new": v2.id in returned_ids,
        "recall_returns_old": v1.id in returned_ids,
        "superseded_by": old.metadata.get("superseded_by") == v2.id,
    }
    RESULTS["contradiction_flow"] = out
    print(f"    old -> {out['old_epistemic_status']} (conf {out['old_confidence']}), "
          f"recall_returns_new={out['recall_returns_new']}, returns_old={out['recall_returns_old']}")


# ──────────────────────────────────────────────────────────────────────────
# Experiment 4 — Self-audit (lint) precision on planted issues, LLM-free
# ──────────────────────────────────────────────────────────────────────────
def exp_lint_precision():
    print("\n[4] Self-audit (lint) on planted issues ...")
    brain, _ = _fresh_brain()
    planted = {"contradiction": 0, "stale": 0, "duplicate": 0}

    # healthy filler — deliberately DIVERSE so it creates no incidental
    # near-duplicate pairs (each is a distinct topic).
    fillers = [
        "The user's name is Dana and she works in Berlin.",
        "The team ships on a four-day release cadence.",
        "Invoices are generated on the first of each month.",
        "The staging environment mirrors production hardware.",
        "Customer support hours are 9am to 6pm CET.",
        "The mobile app targets iOS 17 and Android 14.",
        "Quarterly planning happens in the second week of the quarter.",
        "The design system lives in Figma.",
    ]
    for f in fillers:
        brain.remember(f, tags=["filler"])

    # plant a contradiction: two CURRENT memories explicitly linked CONTRADICTS.
    # Phrased to be semantically distinct (not a near-duplicate pair) so the
    # duplicate detector does not also flag them.
    a = brain.remember("The cache TTL is set to 60 seconds.", tags=["plant"])
    b = brain.remember("Responses are never cached; everything is fetched live.",
                       tags=["plant"])
    a = brain.storage.get_memory(a.id); b = brain.storage.get_memory(b.id)
    a.is_current = True; b.is_current = True
    brain.storage.update_memory(a); brain.storage.update_memory(b)
    brain.storage.store_link(MemoryLink(source_id=a.id, target_id=b.id,
                                        link_type=LinkType.CONTRADICTS, weight=0.9))
    planted["contradiction"] += 1

    # plant stale memories: old + never accessed, and mutually distinct.
    now = time.time()
    stales = [
        "An experimental dark-mode-v1 flag was tried in 2023 and abandoned.",
        "The old payment provider PayGlobe was deprecated years ago.",
        "A one-off migration script for the 2022 schema ran exactly once.",
    ]
    for text in stales:
        s = brain.remember(text, tags=["plant"])
        sm = brain.storage.get_memory(s.id)
        sm.created_at = now - 20 * 86400  # 20 days old
        sm.last_accessed = sm.created_at
        sm.access_count = 0
        brain.storage.update_memory(sm)
        planted["stale"] += 1

    # plant a near-duplicate: identical embedding, distinct id
    orig = brain.remember("The on-call rotation handoff happens every Monday at 10am.",
                          tags=["plant"])
    orig = brain.storage.get_memory(orig.id)
    dup = Memory(
        content="On-call rotation handoff is every Monday at 10am.",
        memory_type=orig.memory_type,
        embedding=orig.embedding,  # force cosine ~1.0
        namespace=orig.namespace,
        tags=["plant"],
    )
    brain.storage.store_memory(dup)
    brain.memory_index.add(dup.id, dup.embedding)
    planted["duplicate"] += 1

    report = brain.lint()
    caught = {
        "contradiction": len(report.get("unresolved_contradictions", [])),
        "stale": len(report.get("stale_memories", [])),
        "duplicate": len(report.get("duplicates", [])),
    }
    RESULTS["lint_precision"] = {
        "planted": planted,
        "caught": caught,
        "total_issues": report.get("total_issues"),
        "orphans": len(report.get("orphan_memories", [])),
    }
    print(f"    planted={planted}  caught={caught}")


# ──────────────────────────────────────────────────────────────────────────
# Experiment 5 — On-device latency & footprint
# ──────────────────────────────────────────────────────────────────────────
def exp_latency_footprint(n_store=400, n_query=80):
    print(f"\n[5] On-device latency/footprint (store {n_store}, query {n_query}) ...")
    brain, tmp = _fresh_brain()
    topics = ["database", "deployment", "frontend", "testing", "auth", "billing",
              "caching", "logging", "monitoring", "api design"]
    verbs = ["prefers", "configured", "migrated to", "debugged", "documented", "reviewed"]
    techs = ["PostgreSQL", "Redis", "React", "FastAPI", "Docker", "Kubernetes",
             "pytest", "Grafana", "OAuth2", "Stripe"]

    t0 = time.time()
    for i in range(n_store):
        c = (f"On {topics[i % len(topics)]}, the user {verbs[i % len(verbs)]} "
             f"{techs[i % len(techs)]} for project P{i % 17} (note {i}).")
        brain.remember(c, tags=["bench"])
    store_elapsed = time.time() - t0

    lat = []
    for i in range(n_query):
        q = f"What did the user do about {topics[i % len(topics)]}?"
        t = time.time()
        brain.recall(q, top_k=5)
        lat.append((time.time() - t) * 1000.0)

    db_path = Path(tmp) / "mem.db"
    db_bytes = db_path.stat().st_size if db_path.exists() else 0
    faiss_bytes = sum(p.stat().st_size for p in Path(tmp).glob("*.faiss"))

    RESULTS["latency_footprint"] = {
        "n_store": n_store,
        "n_query": n_query,
        "store_total_s": round(store_elapsed, 2),
        "store_per_mem_ms": round(store_elapsed / max(n_store, 1) * 1000, 2),
        "recall_ms_mean": round(statistics.mean(lat), 2),
        "recall_ms_p50": round(statistics.median(lat), 2),
        "recall_ms_p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 2),
        "db_size_kb": round(db_bytes / 1024, 1),
        "faiss_size_kb": round(faiss_bytes / 1024, 1),
    }
    print(f"    recall p50={RESULTS['latency_footprint']['recall_ms_p50']}ms "
          f"p95={RESULTS['latency_footprint']['recall_ms_p95']}ms; "
          f"db={RESULTS['latency_footprint']['db_size_kb']}KB")


def main():
    shared, _ = _fresh_brain()
    exp_forgetting_curve(shared)
    exp_confidence_ranking(shared)
    exp_contradiction_flow()
    exp_lint_precision()
    exp_latency_footprint()

    RESULTS["_meta"] = {
        "seed": SEED,
        "embedding_mode": "local (all-mpnet-base-v2)",
        "llm": "none (enrichment disabled)",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2 if available",
    }
    out = HERE / "results.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nWrote {out}")
    print("Figures in", FIG_DIR)


if __name__ == "__main__":
    main()
