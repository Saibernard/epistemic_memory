#!/usr/bin/env python3
"""
longmemeval_runner.py
=====================================================================
Run the Memory Layer (with the epistemic / grounded-abstention policy)
on the LongMemEval-S benchmark and produce an external, comparable
accuracy number for the paper.

What it does, per benchmark question:
  1. spins up an isolated memory namespace for that question,
  2. ingests the question's chat "haystack" (all sessions/turns) as
     memories (on-device embeddings, no API key needed for this step),
  3. retrieves the top-k memories for the question,
  4. applies the EPISTEMIC ABSTENTION GATE (abstain if the evidence is
     weak / superseded / contradicted), otherwise asks an LLM to answer
     using ONLY the retrieved memories,
  5. writes the answer to a predictions file in LongMemEval's
     hypothesis format.

Then EITHER:
  * (recommended, official) score with LongMemEval's own evaluate_qa.py
    on the predictions file, OR
  * (quick, approximate) pass --quick-judge to get an in-script
    LLM-judged accuracy + per-question-type breakdown.

Cost / time: ingesting a full haystack is hundreds of embeddings per
question, and answering+judging calls an LLM. START SMALL with --limit
(default 20). Scale up once it works.

Usage:
  export OPENAI_API_KEY=sk-...
  python paper/longmemeval_runner.py \\
      --data /path/to/longmemeval_s.json \\
      --limit 20 --quick-judge

Get the dataset from: https://github.com/xiaowu0162/LongMemEval
=====================================================================
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

# --- embeddings stay local/offline; only answering+judging use the LLM ---
os.environ.setdefault("MEMORY_EMBEDDING_MODE", "local")
os.environ.setdefault("MEMORY_LLM_EXTRACT", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------
#  Lazy imports with friendly errors
# ---------------------------------------------------------------------
def _import_memory():
    try:
        from memory_layer import MemoryManager
        return MemoryManager
    except Exception as e:
        sys.exit("ERROR: could not import memory_layer. From the repo root run:\n"
                 "  pip install -e \".[dev]\"\n"
                 f"Import error: {e}")

def _make_llm(base_url=None):
    """Return a function answer(messages)->str using the OpenAI SDK."""
    try:
        from openai import OpenAI
    except Exception:
        sys.exit("ERROR: the OpenAI SDK is required for answering/judging.\n"
                 "  pip install openai\n"
                 "and set OPENAI_API_KEY (and optionally OPENAI_BASE_URL).")
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
    )

    def call(messages, model, temperature=0.0):
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # simple backoff on rate limits / transient errors
                if attempt == 3:
                    return f"[LLM_ERROR: {e}]"
                time.sleep(2 * (attempt + 1))
    return call


# ---------------------------------------------------------------------
#  Result-field helpers (robust to objects or dicts)
# ---------------------------------------------------------------------
def _f(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _score(r):
    for k in ("composite_score", "relevance_score", "confidence"):
        v = _f(r, k, None)
        if v is not None:
            return float(v)
    return 0.0


ABSTAIN_TEXT = "I don't know."
_ABSTAIN_MARKERS = ("i don't know", "i do not know", "not sure", "no information",
                    "cannot determine", "can't determine", "don't have", "do not have")

def looks_like_abstention(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _ABSTAIN_MARKERS)


# ---------------------------------------------------------------------
#  Ingest one question's haystack into an isolated namespace
# ---------------------------------------------------------------------
def ingest_haystack(brain, inst, namespace, max_turns_per_session=None):
    sessions = inst.get("haystack_sessions") or inst.get("sessions") or []
    dates = inst.get("haystack_dates") or []
    n = 0
    for si, session in enumerate(sessions):
        turns = session if isinstance(session, list) else (session.get("turns") or [])
        date = dates[si] if si < len(dates) else ""
        for ti, turn in enumerate(turns):
            if max_turns_per_session and ti >= max_turns_per_session:
                break
            if not isinstance(turn, dict):
                continue
            role = turn.get("role", turn.get("speaker", ""))
            content = turn.get("content", turn.get("text", ""))
            if not content:
                continue
            prefix = f"[{date}] " if date else ""
            try:
                brain.remember(f"{prefix}{role}: {content}",
                               importance=0.5, namespace=namespace)
                n += 1
            except Exception:
                pass
    return n


# ---------------------------------------------------------------------
#  Answer one question (epistemic abstention gate + LLM)
# ---------------------------------------------------------------------
def answer_question(brain, llm, inst, namespace, args):
    question = inst.get("question", "")
    try:
        results = brain.recall(question, top_k=args.top_k, namespace=namespace)
    except TypeError:
        results = brain.recall(question, top_k=args.top_k)  # namespace kwarg fallback
    except Exception:
        results = []

    # --- epistemic abstention gate ---
    if not results:
        return ABSTAIN_TEXT, 0.0, "no_results"
    top = results[0]
    top_score = _score(top)
    mem = _f(top, "memory")
    status = (_f(mem, "epistemic_status", "inferred") or "inferred")
    meta = _f(mem, "metadata", {}) or {}
    contradicted = (status in ("contradicted", "uncertain")) or bool(meta.get("contradicts"))
    if top_score < args.theta:
        return ABSTAIN_TEXT, top_score, "low_score"
    if contradicted:
        return ABSTAIN_TEXT, top_score, "contradicted"

    # --- grounded answer from retrieved memories only ---
    context = "\n".join(f"- {_f(_f(r, 'memory'), 'content', '')}" for r in results[:args.top_k])
    qdate = inst.get("question_date", "")
    sys_prompt = (
        "You answer questions about a user using ONLY the provided memory context. "
        "Be concise and direct. If the context does not contain enough information "
        f"to answer, reply exactly: {ABSTAIN_TEXT}"
    )
    user_prompt = (f"Today's date: {qdate}\n\n" if qdate else "") + \
        f"Memory context:\n{context}\n\nQuestion: {inst.get('question','')}"
    ans = llm([{"role": "system", "content": sys_prompt},
               {"role": "user", "content": user_prompt}], args.model)
    return ans, top_score, "answered"


# ---------------------------------------------------------------------
#  Quick (approximate) LLM judge
# ---------------------------------------------------------------------
def quick_judge(llm, inst, hypothesis, model):
    qid = str(inst.get("question_id", ""))
    # Abstention questions: id convention ends in "_abs"; correct == abstained.
    if qid.endswith("_abs") or inst.get("question_type", "").endswith("abstention"):
        return looks_like_abstention(hypothesis)
    ref = inst.get("answer", inst.get("reference", ""))
    sys_p = ("You are grading a candidate answer against a reference answer for a "
             "question. Reply with a single word: 'yes' if the candidate is correct "
             "(captures the reference's key information), otherwise 'no'.")
    user_p = (f"Question: {inst.get('question','')}\n"
              f"Reference answer: {ref}\n"
              f"Candidate answer: {hypothesis}\n\nCorrect? (yes/no)")
    verdict = llm([{"role": "system", "content": sys_p},
                   {"role": "user", "content": user_p}], model).lower()
    return verdict.strip().startswith("yes")


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LongMemEval-S runner for Memory Layer")
    ap.add_argument("--data", required=True, help="path to longmemeval_s.json")
    ap.add_argument("--limit", type=int, default=20, help="number of questions (START SMALL)")
    ap.add_argument("--model", default="gpt-4o-mini", help="LLM for answering/judging")
    ap.add_argument("--base_url", default=None, help="optional OpenAI-compatible base URL")
    ap.add_argument("--theta", type=float, default=0.35, help="abstention score threshold")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--max_turns_per_session", type=int, default=None,
                    help="cap turns ingested per session (speed)")
    ap.add_argument("--out", default=os.path.join(_HERE, "longmemeval_predictions.jsonl"))
    ap.add_argument("--quick-judge", action="store_true",
                    help="also compute an approximate accuracy (calls the LLM)")
    args = ap.parse_args()

    MemoryManager = _import_memory()
    llm = _make_llm(args.base_url)  # needed for answering (and judging)

    with open(args.data) as fh:
        data = json.load(fh)
    if isinstance(data, dict):  # some releases wrap in {"data": [...]}
        data = data.get("data") or data.get("questions") or list(data.values())
    if args.limit:
        data = data[:args.limit]

    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="lme_")
    brain = MemoryManager(db_path=os.path.join(tmpdir, "lme.db"))

    preds = []
    judged = []  # (question_type, correct)
    t0 = time.time()
    print(f"[run] {len(data)} questions | model={args.model} | theta={args.theta}")
    for i, inst in enumerate(data):
        qid = inst.get("question_id", f"q{i}")
        ns = f"lme_{qid}"
        n_ing = ingest_haystack(brain, inst, ns, args.max_turns_per_session)
        hyp, score, route = answer_question(brain, llm, inst, ns, args)
        preds.append({"question_id": qid, "hypothesis": hyp})
        line = f"  [{i+1}/{len(data)}] {qid:>24}  ingested={n_ing:<4} route={route:<11}"
        if args.quick_judge:
            ok = quick_judge(llm, inst, hyp, args.model)
            judged.append((inst.get("question_type", "unknown"), ok))
            line += f"  judged={'OK ' if ok else 'x  '}"
        print(line)

    with open(args.out, "w") as fh:
        for p in preds:
            fh.write(json.dumps(p) + "\n")

    try:
        brain.shutdown()
    except Exception:
        pass

    dt = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  wrote {len(preds)} predictions -> {args.out}  ({dt:.0f}s)")
    if args.quick_judge and judged:
        overall = sum(1 for _, ok in judged if ok) / len(judged)
        print(f"  APPROX overall accuracy: {100*overall:.1f}%  (n={len(judged)})")
        # per type
        by = {}
        for qt, ok in judged:
            by.setdefault(qt, [0, 0])
            by[qt][0] += int(ok); by[qt][1] += 1
        print("  by question type:")
        for qt in sorted(by):
            c, t = by[qt]
            print(f"    {qt:28s} {100*c/t:5.1f}%  ({c}/{t})")
        print("\n  NOTE: --quick-judge is an APPROXIMATION. For the official number,")
    else:
        print("\n  For the official number,")
    print("  run LongMemEval's own evaluator on the predictions file, e.g.:")
    print(f"    python evaluate_qa.py gpt-4o {args.out} <longmemeval_s.json>")
    print("  (check the LongMemEval repo README for the exact evaluator signature)")
    print("=" * 60)


if __name__ == "__main__":
    main()
