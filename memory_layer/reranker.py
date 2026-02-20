"""
Cross-Encoder Neural Reranker for the Memory Layer.

After the composite heuristic scoring narrows candidates to a shortlist,
a cross-encoder model jointly encodes (query, memory_content) and produces
a single relevance score.  This is significantly more accurate than
dot-product similarity for ambiguous or paraphrased queries.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80 MB, runs on CPU)

Falls back gracefully to a no-op when:
  - sentence-transformers is not installed
  - the model fails to load
  - reranking is disabled via MEMORY_RERANKER=none

This keeps the system fully functional without the reranker —
it's a quality boost, not a hard dependency.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional, Protocol


class RerankerBackend(Protocol):
    def score_pairs(self, query: str, texts: List[str]) -> List[float]: ...


_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for reranking.

    Scores (query, document) pairs jointly — the gold standard for
    retrieval precision. ~5-15 ms per pair on modern CPU.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._available = False
        self._tried = False

    def _ensure_model(self):
        if self._tried:
            return
        self._tried = True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        self._ensure_model()
        return self._available

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """
        Score each (query, text) pair. Returns a list of float scores
        aligned with the input texts. Higher = more relevant.
        """
        self._ensure_model()
        if not self._available or not texts:
            return [0.0] * len(texts)

        pairs = [[query, t] for t in texts]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


class NoOpReranker:
    """Passthrough reranker that returns zeros (no reranking applied)."""

    @property
    def available(self) -> bool:
        return False

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        return [0.0] * len(texts)


def create_reranker(
    mode: Optional[str] = None,
    model_name: str = _DEFAULT_MODEL,
) -> CrossEncoderReranker | NoOpReranker:
    """
    Factory for the reranker.

    mode:
      "auto"  — try to load cross-encoder, fall back to no-op
      "cross_encoder" — require cross-encoder
      "none"  — no reranking
    """
    mode = mode or os.environ.get("MEMORY_RERANKER", "auto")

    if mode == "none":
        return NoOpReranker()

    reranker = CrossEncoderReranker(model_name=model_name)
    if mode == "cross_encoder":
        if not reranker.available:
            raise RuntimeError(
                f"Cross-encoder model {model_name} could not be loaded. "
                "Install sentence-transformers: pip install sentence-transformers"
            )
    return reranker
