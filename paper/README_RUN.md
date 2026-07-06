# Epistemic Memory — paper + validation harness

This folder contains a self-contained arXiv-style paper and a reproducible,
API-free validation harness for the "Epistemic Memory" idea built on top of
`memory_layer`.

## Files
- `epistemic_memory.tex` — the paper source (single file, no bibtex needed).
- `epistemic_memory.pdf` — compiled preview (10 pages).
- `validate_epistemic.py` — runnable validation harness (3 experiments).
- `README_RUN.md` — this file.

## 1. Get a real results table (local, no API key)
From the **repo root**:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # installs faiss, sentence-transformers, etc.
python paper/validate_epistemic.py --trials 20 --theta 0.35
```

This prints metrics for Experiments 1–3 and writes:
- `paper/results.json`        — raw numbers
- `paper/results_table.tex`   — a drop-in LaTeX table

The first run downloads the local embedding model (~400 MB) once.

**Tuning `--theta`:** the script prints the top-1 score distribution per class
(answerable / updated / absent). If it answers questions it should refuse,
raise `--theta`; if it refuses answerable questions, lower it. Pick a value
between the absent and answerable means.

## 2. Put the real numbers in the paper
Open `epistemic_memory.tex`, find the placeholder table (search for
`\label{tab:results}` in the Results section), delete that `table` environment,
and replace it with:

```latex
\input{results_table.tex}
```

Then recompile (see below). The generated table carries the same label, so the
in-text reference still resolves.

## 3. Compile the PDF
```bash
cd paper
latexmk -pdf epistemic_memory.tex      # or: pdflatex epistemic_memory.tex  (twice)
```

## 4. External benchmark — LongMemEval-S (the number that makes it credible)
This needs an API key (the answering + judging steps call an LLM). Start small.

1. Get the dataset from https://github.com/xiaowu0162/LongMemEval — you want
   `longmemeval_s.json`.
2. Run the runner (begin with a small `--limit`; it's the cheap way to sanity-check):

```bash
pip install openai
export OPENAI_API_KEY=sk-...
python paper/longmemeval_runner.py --data /path/to/longmemeval_s.json --limit 20 --quick-judge
```

It ingests each question's chat history into Memory Layer (local embeddings),
answers via the epistemic abstention policy, writes
`paper/longmemeval_predictions.jsonl`, and with `--quick-judge` prints an
**approximate** accuracy + per-question-type breakdown.

3. For the **official** number to cite in the paper, score the predictions file
   with LongMemEval's own evaluator (see their repo README for the exact
   command), then report overall accuracy plus the **knowledge-update** and
   **abstention** slices — where the epistemic policy should help most.

Useful flags: `--theta` (abstention threshold), `--top_k` (retrieved context
size), `--max_turns_per_session` (cap ingestion for speed/cost), `--model`,
`--base_url` (any OpenAI-compatible endpoint).

## Before you submit to arXiv
- **Check the references.** Author/year/arXiv-id details were collected from
  public sources; sanity-check each `\bibitem` (one entry is flagged
  `% VERIFY`).
- **Keep it honest.** The paper already states that belief propagation through
  consolidation and a weighted epistemic ranking term are *proposed but only
  partially implemented* (see the Limitations section). Don't quietly drop that.
- arXiv needs an endorsement for first-time cs.AI / cs.CL / cs.IR submitters.
