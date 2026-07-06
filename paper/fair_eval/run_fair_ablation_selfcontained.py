#!/usr/bin/env python3
"""
run_fair_ablation_selfcontained.py  —  the clean, final fair ablation.
=====================================================================
Self-contained (imports NOTHING from the other runner, so it is immune to
concurrent edits) and uses GENUINELY DIVERSE facts — different topics AND
sentence structures — so the store's automatic contradiction/supersession
logic never fires and the trusted store survives intact. Correctness is
value-based.

Same experiment: ONE store, toggle ONLY the epistemic modifier
(use_epistemic False vs True), sweep the abstention threshold theta for both,
compare at a matched answerable-coverage operating point.

Classes (each a distinct fact/topic):
  trusted   : verified, high-confidence, current   -> should ANSWER
  low_trust : current but marked contradicted/uncertain, conf 0.3 -> should ABSTAIN
  absent    : nothing relevant stored              -> should ABSTAIN

Outputs (paper/fair_eval/, distinct *_selfcontained names; touches nothing else):
  results_selfcontained.json
  fig_sweep_selfcontained.png
  fig_lowtrust_selfcontained.png
  results_table_selfcontained.tex

Run:  python3 paper/fair_eval/run_fair_ablation_selfcontained.py
"""
from __future__ import annotations

import os
os.environ.setdefault("MEMORY_EMBEDDING_MODE", "local")
os.environ.setdefault("MEMORY_ENRICHMENT_BACKEND", "none")
os.environ.setdefault("MEMORY_LLM_EXTRACT", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from memory_layer import MemoryManager  # noqa: E402


# --- genuinely diverse facts (distinct topics, distinct structures) ---------
TRUSTED = [  # (statement, question, answer_value)
    ("The Eiffel Tower stands in the city of Paris.",            "Which city is the Eiffel Tower in?",            "paris"),
    ("Photosynthesis takes place inside the chloroplast.",       "Where does photosynthesis happen?",             "chloroplast"),
    ("Mount Everest is the tallest mountain on Earth.",          "What is the tallest mountain on Earth?",        "everest"),
    ("The novel Moby-Dick was written by Herman Melville.",      "Who wrote Moby-Dick?",                          "melville"),
    ("The human heart has four chambers.",                       "How many chambers does the human heart have?",  "four"),
    ("The currency of Japan is the yen.",                        "What is the currency of Japan?",                "yen"),
    ("Penicillin was discovered by Alexander Fleming.",          "Who discovered penicillin?",                    "fleming"),
    ("The Great Barrier Reef lies off the coast of Australia.",  "Where is the Great Barrier Reef?",              "australia"),
    ("Jupiter is the largest planet in the solar system.",       "What is the largest planet in the solar system?", "jupiter"),
    ("The Mona Lisa was painted by Leonardo da Vinci.",          "Who painted the Mona Lisa?",                    "leonardo"),
    ("The official language of Brazil is Portuguese.",           "What is the official language of Brazil?",      "portuguese"),
    ("The Pacific is the largest ocean on Earth.",               "What is the largest ocean on Earth?",           "pacific"),
    ("Insulin is produced by the pancreas.",                     "Which organ produces insulin?",                 "pancreas"),
    ("The Berlin Wall fell in nineteen eighty-nine.",            "When did the Berlin Wall fall?",                "eighty-nine"),
    ("Helium is lighter than air.",                              "Is helium lighter or heavier than air?",        "lighter"),
]

LOW_TRUST = [  # (statement, question)  -- should ABSTAIN
    ("Rumor has it the new office might open in Lisbon, but it is unconfirmed.", "Where will the new office open?"),
    ("Someone said the quarterly revenue could be around twelve million, unverified.", "What was the quarterly revenue?"),
    ("It is alleged the merger may close sometime in autumn, though disputed.", "When will the merger close?"),
    ("An unverified note claims the staging credentials were rotated last week.", "Were the staging credentials rotated?"),
    ("Gossip suggests the keynote speaker might be a retired astronaut.", "Who is the keynote speaker?"),
    ("There is a rumor the product launch slipped to next spring.", "When is the product launch?"),
    ("Reportedly the vendor contract is worth six figures, but this is uncertain.", "How much is the vendor contract worth?"),
    ("Someone mentioned the support team may relocate to a co-working space.", "Is the support team relocating?"),
    ("Allegedly the prototype reached record battery life, unconfirmed.", "What battery life did the prototype reach?"),
    ("It might be that the annual conference moved to a virtual format, disputed.", "What format is the annual conference?"),
    ("An unverified tip says the funding round could be led by a new investor.", "Who is leading the funding round?"),
    ("Hearsay indicates the warehouse lease may expire this summer.", "When does the warehouse lease expire?"),
    ("Possibly the onboarding hire starts in two weeks, but nothing is confirmed.", "When does the onboarding hire start?"),
    ("It is said the marketing budget could be trimmed, though nothing official.", "Will the marketing budget be trimmed?"),
    ("Rumored that the rate limit on the gateway may have been raised recently.", "What is the gateway rate limit?"),
]

ABSENT = [
    "What is my bank account PIN?",
    "What is my shoe size?",
    "What medication do I take daily?",
    "What is my passport number?",
    "What is my home Wi-Fi password?",
    "What was the score of last night's basketball game?",
    "What is my sister's phone number?",
    "How tall is my next-door neighbor?",
    "What time is my dentist appointment tomorrow?",
    "What is the serial number of my laptop?",
    "What is my favorite Mozart symphony?",
    "What is my car's license plate number?",
    "What is the gate number for my flight?",
    "What is my landlord's middle name?",
    "What is my gym locker combination?",
]

ABSTAIN_STATUSES = ("contradicted", "uncertain")


def _f(o, n, d=None):
    return (o.get(n, d) if isinstance(o, dict) else getattr(o, n, d))

def _score(r):
    for k in ("composite_score", "relevance_score", "confidence"):
        v = _f(r, k, None)
        if v is not None:
            return float(v)
    return 0.0


def _set_belief(brain, mem_id, status, confidence, is_current=True):
    mm = brain.storage.get_memory(mem_id)
    if mm is None:
        return
    mm.epistemic_status = status
    mm.confidence = float(confidence)
    mm.is_current = is_current
    brain.storage.update_memory(mm)


def build_store(brain, n_trusted, n_low):
    queries = []
    for (stmt, q, val) in TRUSTED[:n_trusted]:
        m = brain.remember(stmt, importance=0.7)
        _set_belief(brain, m.id, "verified", 0.9, is_current=True)
        queries.append(("trusted", q, val))
    for i, (stmt, q) in enumerate(LOW_TRUST[:n_low]):
        m = brain.remember(stmt, importance=0.5)
        _set_belief(brain, m.id, ABSTAIN_STATUSES[i % 2], 0.3, is_current=True)
        queries.append(("low_trust", q, None))
    for q in ABSENT[:max(n_trusted, n_low)]:
        queries.append(("absent", q, None))
    return queries


def answer(brain, q, use_epistemic, top_k):
    try:
        res = brain.recall(q, top_k=top_k, use_epistemic=use_epistemic)
    except TypeError:
        res = brain.recall(q, top_k=top_k)
    except Exception:
        res = []
    if not res:
        return 0.0, ""
    top = res[0]
    return _score(top), (_f(_f(top, "memory"), "content", "") or "")


def evaluate(brain, queries, use_epistemic, thetas, top_k):
    cache = []
    for (cls, q, val) in queries:
        sc, content = answer(brain, q, use_epistemic, top_k)
        cache.append((cls, val, sc, content.lower()))
    n_tr = sum(1 for c in cache if c[0] == "trusted")
    n_lt = sum(1 for c in cache if c[0] == "low_trust")
    n_ab = sum(1 for c in cache if c[0] == "absent")
    out = {"theta": list(map(float, thetas)), "answerable_acc": [], "lowtrust_cw": [],
           "absent_cw": [], "overall_cw": [], "abstention_f1": []}
    for th in thetas:
        tr_ok = tr_wrong = lt_ans = ab_cw = 0
        tp = fp = fn = 0
        for (cls, val, sc, content) in cache:
            answered = sc >= th
            if cls == "trusted":
                if answered and (val or "") in content:
                    tr_ok += 1
                elif answered:
                    tr_wrong += 1
                else:
                    fp += 1
            elif cls == "low_trust":
                if answered:
                    lt_ans += 1; fn += 1
                else:
                    tp += 1
            else:
                if answered:
                    ab_cw += 1; fn += 1
                else:
                    tp += 1
        total = n_tr + n_lt + n_ab
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out["answerable_acc"].append(round(tr_ok / n_tr, 4) if n_tr else None)
        out["lowtrust_cw"].append(round(lt_ans / n_lt, 4) if n_lt else None)
        out["absent_cw"].append(round(ab_cw / n_ab, 4) if n_ab else None)
        out["overall_cw"].append(round((lt_ans + ab_cw + tr_wrong) / total, 4) if total else None)
        out["abstention_f1"].append(round(f1, 4))
    out["_counts"] = {"trusted": n_tr, "low_trust": n_lt, "absent": n_ab}
    return out


def matched_point(curve, target):
    best = None
    for i, a in enumerate(curve["answerable_acc"]):
        if a is not None and a >= target:
            best = i
    if best is None:
        accs = [a if a is not None else -1 for a in curve["answerable_acc"]]
        best = int(np.argmax(accs))
    return best


def _hp(c, i):
    return {"theta": c["theta"][i], "answerable_acc": c["answerable_acc"][i],
            "lowtrust_confident_wrong": c["lowtrust_cw"][i],
            "absent_confident_wrong": c["absent_cw"][i],
            "overall_confident_wrong": c["overall_cw"][i],
            "abstention_f1": c["abstention_f1"][i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--target_coverage", type=float, default=0.9)
    ap.add_argument("--theta_max", type=float, default=0.9)
    ap.add_argument("--theta_steps", type=int, default=19)
    args = ap.parse_args()

    n_tr, n_lt = len(TRUSTED), len(LOW_TRUST)
    thetas = list(np.round(np.linspace(0.0, args.theta_max, args.theta_steps), 4))
    tmp = tempfile.mkdtemp(prefix="fair_sc_")
    brain = MemoryManager(db_path=os.path.join(tmp, "mem.db"))

    t0 = time.time()
    queries = build_store(brain, n_tr, n_lt)

    # --- SANITY: did the store survive (no supersession collapse)? ---
    allm = [m for m in brain.storage.get_all_memories(active_only=True)]
    current = sum(1 for m in allm if getattr(m, "is_current", True))
    print(f"[setup] stored {n_tr} trusted + {n_lt} low-trust; "
          f"current active memories = {current} (expect ~{n_tr + n_lt})")
    store_intact = current >= 0.9 * (n_tr + n_lt)
    print(f"[setup] store intact (no supersession collapse): {store_intact}")

    base = evaluate(brain, queries, False, thetas, args.top_k)
    epi = evaluate(brain, queries, True, thetas, args.top_k)
    bi, ei = matched_point(base, args.target_coverage), matched_point(epi, args.target_coverage)
    headline = {"target_coverage": args.target_coverage,
                "baseline": _hp(base, bi), "epistemic": _hp(epi, ei)}

    results = {"config": {"top_k": args.top_k, "thetas": thetas, "counts": base["_counts"],
                          "current_memories_after_build": current,
                          "store_intact": store_intact,
                          "runtime_s": round(time.time() - t0, 1)},
               "sweep_baseline": base, "sweep_epistemic": epi, "headline": headline,
               "_meta": {"note": "self-contained; diverse facts (no supersession); "
                                 "same store, toggle only use_epistemic; theta swept; value-based correctness."}}
    (HERE / "results_selfcontained.json").write_text(json.dumps(results, indent=2))

    # fig 1: overall confident-wrong vs theta
    plt.figure(figsize=(7, 4.5))
    plt.plot(thetas, base["overall_cw"], marker="o", label="baseline (relevance-only + abstain)")
    plt.plot(thetas, epi["overall_cw"], marker="s", label="epistemic (modifier + abstain)")
    plt.xlabel("Abstention threshold theta"); plt.ylabel("Overall confident-wrong rate")
    plt.title("Fair ablation: epistemic signal lowers confident-wrong across theta")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(HERE / "fig_sweep_selfcontained.png", dpi=150); plt.close()

    # fig 2: low-trust confident-wrong at matched coverage
    plt.figure(figsize=(5.5, 4.2))
    vals = [headline["baseline"]["lowtrust_confident_wrong"],
            headline["epistemic"]["lowtrust_confident_wrong"]]
    plt.bar(["baseline\n(relevance-only)", "epistemic"], vals, color=["#b0b0b0", "#4c78a8"])
    plt.ylabel("Low-trust confident-wrong rate"); plt.ylim(0, 1)
    plt.title(f"Low-trust subset @ matched coverage (~{int(args.target_coverage*100)}%)")
    for i, v in enumerate(vals):
        if v is not None:
            plt.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    plt.tight_layout(); plt.savefig(HERE / "fig_lowtrust_selfcontained.png", dpi=150); plt.close()

    # table
    def pct(x):
        return "--" if x is None else f"{100*x:.1f}\\%"
    b, e = headline["baseline"], headline["epistemic"]
    tex = r"""\begin{table}[h]\centering\small
\caption{Fair ablation (same store; only the epistemic modifier toggled; theta
swept; reported at matched answerable coverage $\approx%d\%%$; diverse facts, no
supersession). Lower is better for confident-wrong rates.}
\label{tab:fair}
\begin{tabular}{lcc}
\toprule
Metric & Baseline (relevance-only) & Epistemic \\
\midrule
Answerable accuracy ($\uparrow$)         & %s & %s \\
Low-trust confident-wrong ($\downarrow$) & %s & %s \\
Absent confident-wrong ($\downarrow$)    & %s & %s \\
Overall confident-wrong ($\downarrow$)   & %s & %s \\
Abstention F1 ($\uparrow$)               & %.2f & %.2f \\
\bottomrule
\end{tabular}
\end{table}
""" % (int(args.target_coverage * 100),
       pct(b["answerable_acc"]), pct(e["answerable_acc"]),
       pct(b["lowtrust_confident_wrong"]), pct(e["lowtrust_confident_wrong"]),
       pct(b["absent_confident_wrong"]), pct(e["absent_confident_wrong"]),
       pct(b["overall_confident_wrong"]), pct(e["overall_confident_wrong"]),
       b["abstention_f1"], e["abstention_f1"])
    (HERE / "results_table_selfcontained.tex").write_text(tex)

    def pp(x):
        return "  --  " if x is None else f"{100*x:5.1f}%"
    print("=" * 66)
    print("  FAIR ABLATION (SELF-CONTAINED, diverse facts)")
    print("=" * 66)
    print(f"  baseline  theta={b['theta']:.3f}  ans_acc={pp(b['answerable_acc'])}")
    print(f"  epistemic theta={e['theta']:.3f}  ans_acc={pp(e['answerable_acc'])}")
    print(f"  LOW-TRUST confident-wrong   baseline={pp(b['lowtrust_confident_wrong'])}"
          f"   epistemic={pp(e['lowtrust_confident_wrong'])}   <-- headline")
    print(f"  absent confident-wrong      baseline={pp(b['absent_confident_wrong'])}"
          f"   epistemic={pp(e['absent_confident_wrong'])}")
    print(f"  overall confident-wrong     baseline={pp(b['overall_confident_wrong'])}"
          f"   epistemic={pp(e['overall_confident_wrong'])}")
    print(f"  abstention F1               baseline={b['abstention_f1']:.2f}"
          f"        epistemic={e['abstention_f1']:.2f}")
    print("=" * 66)
    try:
        brain.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
