"""
OpenAI Embedding Engine for the Memory Layer.

Uses OpenAI's text-embedding-3 models via API for higher-quality embeddings.
Requires an API key and internet connection.

Models:
  - text-embedding-3-small  (1536d, good quality, cheap — default)
  - text-embedding-3-large  (3072d, best quality, more expensive)

Cost (as of 2025):
  - text-embedding-3-small: ~$0.02 per 1M tokens
  - text-embedding-3-large: ~$0.13 per 1M tokens
"""

import os
import time
import numpy as np
from typing import List, Optional, Tuple


# Default retry/batch settings
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0
_BATCH_SIZE = 100  # OpenAI supports up to 2048 inputs per request


class OpenAIEmbeddingEngine:
    """
    Generates text embeddings via the OpenAI API.

    Same interface as LocalEmbeddingEngine so they can be swapped freely.

    Usage:
        engine = OpenAIEmbeddingEngine(api_key="sk-...")
        vector = engine.embed("Hello world")
        similarity = engine.cosine_similarity(vec_a, vec_b)
    """

    # Dimension lookup for supported models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.dimension = self.MODEL_DIMENSIONS.get(model_name, 1536)
        self._cache: dict = {}
        self._client = None
        self._using_fallback = False

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key="
            )

        self._init_client()

    def _init_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            print(f"  + Loaded OpenAI embedding model: {self.model_name} ({self.dimension}d)")
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI embeddings. "
                "Install with: pip install openai"
            )

    def _api_embed(self, texts: List[str]) -> List[List[float]]:
        """Call the OpenAI embeddings API with retry logic."""
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                )
                # Sort by index to maintain order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAY * (2 ** attempt)
                    print(f"  ! OpenAI embed retry {attempt + 1}/{_MAX_RETRIES}: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"OpenAI embedding failed after {_MAX_RETRIES} retries: {e}")

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        if text in self._cache:
            return self._cache[text]

        result = self._api_embed([text])
        embedding = self._normalize(result[0])
        self._cache[text] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        results: list = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        # Fill from cache
        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        # Batch API calls for uncached texts
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), _BATCH_SIZE):
                batch = uncached_texts[batch_start:batch_start + _BATCH_SIZE]
                batch_indices = uncached_indices[batch_start:batch_start + _BATCH_SIZE]
                embeddings = self._api_embed(batch)

                for idx, emb in zip(batch_indices, embeddings):
                    normalized = self._normalize(emb)
                    self._cache[texts[idx]] = normalized
                    results[idx] = normalized

        return results

    @staticmethod
    def _normalize(embedding: List[float]) -> List[float]:
        """L2 normalize an embedding vector."""
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    # ─────────────────────────────────────────────
    # PASSAGE CHUNKING (same interface as LocalEmbeddingEngine)
    # ─────────────────────────────────────────────

    @staticmethod
    def chunk_text(
        text: str,
        max_chars: int = 500,
        overlap_chars: int = 150,
    ) -> List[str]:
        """Split text into overlapping character-level chunks."""
        if len(text) <= max_chars:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            if end >= len(text):
                break
            start += max_chars - overlap_chars
        return chunks if chunks else [text]

    def embed_passages(
        self, text: str, max_chars: int = 500, overlap_chars: int = 150
    ) -> List[Tuple[str, List[float]]]:
        """Split text into overlapping chunks and embed each one."""
        chunks = self.chunk_text(text, max_chars, overlap_chars)
        if len(chunks) <= 1:
            return []
        embeddings = self.embed_batch(chunks)
        return list(zip(chunks, embeddings))

    # ─────────────────────────────────────────────
    # VECTORIZED SIMILARITY (same interface as LocalEmbeddingEngine)
    # ─────────────────────────────────────────────

    @staticmethod
    def batch_cosine_similarities(
        query: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        """Vectorized cosine similarity between one query vector and a matrix."""
        if matrix.size == 0:
            return np.array([], dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(matrix.shape[0], dtype=np.float32)
        row_norms = np.linalg.norm(matrix, axis=1)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        dots = matrix @ query
        return (dots / (row_norms * query_norm)).astype(np.float32)

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)
        dot = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
