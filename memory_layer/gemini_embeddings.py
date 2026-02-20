"""
Gemini Embedding Engine for the Memory Layer.

Uses Google's Gemini embedding models via the Gemini API.
Requires a Google API key and internet connection.

Text-only models:
  - gemini-embedding-001  (3072d, MTEB leader, cheap — default)

Multimodal models (text + image + audio + video + PDF):
  - gemini-embedding-2-preview  (3072d, multimodal, public preview)

Cost (as of March 2026):
  - ~$0.004 per 1K characters

Set MEMORY_EMBEDDING_MODE=gemini and GOOGLE_API_KEY=... to use.
"""

import os
import time
import numpy as np
from typing import List, Optional, Tuple


_MAX_RETRIES = 3
_RETRY_DELAY = 1.0
_BATCH_SIZE = 50


class GeminiEmbeddingEngine:
    """
    Generates text embeddings via the Google Gemini API.

    Same interface as LocalEmbeddingEngine / OpenAIEmbeddingEngine.

    Usage:
        engine = GeminiEmbeddingEngine(api_key="AIza...")
        vector = engine.embed("Hello world")
    """

    MODEL_DIMENSIONS = {
        "gemini-embedding-001": 3072,
        "gemini-embedding-2-preview": 3072,
    }

    # MRL-supported output dimensions for Gemini models
    SUPPORTED_DIMS = {768, 1536, 3072}

    def __init__(
        self,
        model_name: str = "gemini-embedding-2-preview",
        api_key: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._cache: dict = {}
        self._client = None
        self._using_fallback = False

        self._output_dimensionality = output_dimensionality
        if output_dimensionality:
            self.dimension = output_dimensionality
        else:
            self.dimension = self.MODEL_DIMENSIONS.get(model_name, 3072)

        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY env var or pass api_key="
            )

        self._init_client()

    def _init_client(self):
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            print(f"  + Loaded Gemini embedding model: {self.model_name} ({self.dimension}d)")
        except ImportError:
            raise ImportError(
                "google-genai package required for Gemini embeddings. "
                "Install with: pip install google-genai"
            )

    def _api_embed(self, texts: List[str]) -> List[List[float]]:
        """Call the Gemini embed_content API with retry logic."""
        for attempt in range(_MAX_RETRIES):
            try:
                kwargs = {
                    "model": self.model_name,
                    "contents": texts,
                }
                if self._output_dimensionality:
                    kwargs["config"] = {
                        "output_dimensionality": self._output_dimensionality,
                    }

                result = self._client.models.embed_content(**kwargs)
                return [emb.values for emb in result.embeddings]
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAY * (2 ** attempt)
                    print(f"  ! Gemini embed retry {attempt + 1}/{_MAX_RETRIES}: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Gemini embedding failed after {_MAX_RETRIES} retries: {e}"
                    )

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        if text in self._cache:
            return self._cache[text]

        result = self._api_embed([text])
        embedding = self._normalize(result[0])
        self._cache[text] = embedding
        return embedding

    def embed_media(self, data: bytes, mime_type: str, caption: str = None) -> List[float]:
        """
        Generate embedding for an image, video, audio, or PDF file
        using gemini-embedding-2-preview's multimodal capability.

        Returns a single embedding vector in the same space as text embeddings,
        enabling cross-modal search (query text, find matching image, etc.).
        """
        if self.model_name != "gemini-embedding-2-preview":
            raise ValueError(
                f"Multimodal embedding requires gemini-embedding-2-preview, "
                f"but current model is {self.model_name}"
            )

        from google.genai import types

        parts = []
        if caption:
            parts.append(types.Part(text=caption))
        parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))

        for attempt in range(_MAX_RETRIES):
            try:
                kwargs = {
                    "model": self.model_name,
                    "contents": [types.Content(parts=parts)],
                }
                if self._output_dimensionality:
                    kwargs["config"] = {
                        "output_dimensionality": self._output_dimensionality,
                    }
                result = self._client.models.embed_content(**kwargs)
                return self._normalize(result.embeddings[0].values)
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY * (2 ** attempt))
                else:
                    raise RuntimeError(f"Gemini multimodal embed failed: {e}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        results: list = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

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
    # PASSAGE CHUNKING (same interface as other engines)
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
    # VECTORIZED SIMILARITY (same interface as other engines)
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
