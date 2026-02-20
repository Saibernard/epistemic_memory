"""
Embedding Engines for the Memory Layer.

Supports multiple backends:
  - local  : sentence-transformers running on your machine (default, free, private)
  - openai : OpenAI text-embedding-3 models via API (higher quality, costs money)
  - gemini : Google Gemini embedding models via API (best quality, multimodal-ready)

Local models (set via MEMORY_EMBEDDING_MODEL env var):
  - all-mpnet-base-v2   (768d, best local quality — default)
  - all-MiniLM-L12-v2  (384d, fast, good quality)
  - all-MiniLM-L6-v2    (384d, fastest, adequate quality)

OpenAI models:
  - text-embedding-3-small  (1536d, good quality, cheap)
  - text-embedding-3-large  (3072d, best quality)

Gemini models:
  - gemini-embedding-001        (3072d, MTEB leader, text-only)
  - gemini-embedding-2-preview  (3072d, multimodal: text+image+audio+video+PDF)

Set MEMORY_EMBEDDING_MODE=local|openai|gemini to choose the backend.
"""

import hashlib
import os
import numpy as np
from typing import List, Optional, Protocol, Tuple, Union, runtime_checkable


@runtime_checkable
class EmbeddingEngine(Protocol):
    """Protocol that all embedding backends must satisfy."""
    model_name: str
    dimension: int

    def embed(self, text: str) -> List[float]: ...
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...
    def embed_passages(
        self, text: str, max_chars: int = 500, overlap_chars: int = 150
    ) -> List[Tuple[str, List[float]]]: ...


def create_embedding_engine(
    mode: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> "Union[LocalEmbeddingEngine, 'OpenAIEmbeddingEngine', 'GeminiEmbeddingEngine']":
    """
    Factory function to create the appropriate embedding engine.

    Args:
        mode: "local", "openai", or "gemini"
              (default: from MEMORY_EMBEDDING_MODE env or "local")
        model_name: Model name override (default: depends on mode)
        api_key: API key for the chosen provider

    Returns:
        An embedding engine instance with a consistent interface.
    """
    mode = mode or os.environ.get("MEMORY_EMBEDDING_MODE", "local")

    if mode == "openai":
        from .openai_embeddings import OpenAIEmbeddingEngine
        openai_model = model_name or os.environ.get(
            "MEMORY_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        return OpenAIEmbeddingEngine(
            model_name=openai_model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        )
    elif mode == "gemini":
        from .gemini_embeddings import GeminiEmbeddingEngine
        gemini_model = model_name or os.environ.get(
            "MEMORY_EMBEDDING_MODEL", "gemini-embedding-2-preview"
        )
        return GeminiEmbeddingEngine(
            model_name=gemini_model,
            api_key=api_key or os.environ.get("GOOGLE_API_KEY", ""),
        )
    else:
        local_model = model_name or os.environ.get(
            "MEMORY_EMBEDDING_MODEL", "all-mpnet-base-v2"
        )
        return LocalEmbeddingEngine(model_name=local_model)


class LocalEmbeddingEngine:
    """
    Generates text embeddings locally using sentence-transformers.
    
    Primary model: all-mpnet-base-v2 (768 dimensions, best quality)
    Fallback: Deterministic hash-based pseudo-embeddings
    
    Usage:
        engine = LocalEmbeddingEngine()
        vector = engine.embed("Hello world")
        similarity = engine.cosine_similarity(vec_a, vec_b)
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2",
                 cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.model = None
        self.dimension = 768
        self._cache: dict = {}
        self._using_fallback = False

        if cache_dir is None:
            try:
                from .config import get_models_dir
                cache_dir = get_models_dir()
            except Exception:
                pass
        self.cache_dir = cache_dir

        self._load_model()

    def _load_model(self):
        """Load the sentence-transformers model with user-friendly messaging."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("  ⚠ sentence-transformers not installed.")
            print("    Install it:  pip install sentence-transformers")
            print("    Using fallback hash embeddings (reduced quality).")
            self.model = None
            self._using_fallback = True
            return

        try:
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

            model_path = os.path.join(self.cache_dir, self.model_name.replace("/", "_")) if self.cache_dir else None
            is_first_download = model_path and not os.path.exists(model_path)

            if is_first_download:
                print(f"  ↓ Downloading embedding model: {self.model_name} (~90 MB, one-time)...")
                print(f"    Cache: {self.cache_dir}")

            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"  ✓ Loaded embedding model: {self.model_name} ({self.dimension}d)")

        except Exception as e:
            error_str = str(e)
            print(f"  ⚠ Could not load embedding model: {self.model_name}")

            if "ConnectionError" in error_str or "ProxyError" in error_str or "Max retries" in error_str:
                print("    Cause: No internet connection or blocked by proxy.")
                print("    Fix:   Connect to internet for first-time model download,")
                print("           or use --embedding-mode openai with an API key.")
            elif "disk" in error_str.lower() or "space" in error_str.lower():
                print("    Cause: Insufficient disk space.")
                print(f"    Fix:   Free up ~500 MB in {self.cache_dir or '~/.cache'}")
            else:
                print(f"    Error: {error_str[:200]}")

            print("    Using fallback hash embeddings (reduced quality).")
            self.model = None
            self._using_fallback = True

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        if text in self._cache:
            return self._cache[text]

        if self.model is not None:
            embedding = self.model.encode(text, normalize_embeddings=True).tolist()
        else:
            embedding = self._fallback_embed(text)

        self._cache[text] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if self.model is not None:
            # Separate cached and uncached
            uncached = [(i, t) for i, t in enumerate(texts) if t not in self._cache]
            results: list = [None] * len(texts)

            # Fill from cache
            for i, t in enumerate(texts):
                if t in self._cache:
                    results[i] = self._cache[t]

            # Batch encode uncached
            if uncached:
                indices, uncached_texts = zip(*uncached)
                embeddings = self.model.encode(
                    list(uncached_texts), normalize_embeddings=True
                )
                for i, emb in zip(indices, embeddings):
                    emb_list = emb.tolist()
                    self._cache[texts[i]] = emb_list
                    results[i] = emb_list

            return results
        else:
            return [self._fallback_embed(t) for t in texts]

    def _fallback_embed(self, text: str) -> List[float]:
        """
        Fallback embedding using deterministic hashing.
        
        Creates a pseudo-embedding by combining word-level hashes.
        Not as semantically rich as a real model, but deterministic
        and functional for basic similarity matching.
        """
        # Create base vector from full text hash
        text_hash = int(hashlib.sha256(text.lower().encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(text_hash)
        base = rng.randn(self.dimension).astype(np.float32)

        # Add word-level features for semantic sensitivity
        words = text.lower().split()
        for i, word in enumerate(words):
            word_clean = word.strip('.,!?;:()[]{}"\'-')
            if len(word_clean) > 1:
                word_hash = int(hashlib.sha256(word_clean.encode()).hexdigest()[:8], 16)
                word_rng = np.random.RandomState(word_hash)
                word_vec = word_rng.randn(self.dimension).astype(np.float32)
                # Position-weighted contribution (earlier words matter more)
                position_weight = 1.0 / (1.0 + i * 0.1)
                base += word_vec * 0.3 * position_weight

        # L2 normalize
        norm = np.linalg.norm(base)
        if norm > 0:
            base = base / norm

        return base.tolist()

    # ─────────────────────────────────────────────
    # PASSAGE CHUNKING (for long content)
    # ─────────────────────────────────────────────

    @staticmethod
    def chunk_text(
        text: str,
        max_chars: int = 500,
        overlap_chars: int = 150,
    ) -> List[str]:
        """
        Split text into overlapping character-level chunks.

        For memories longer than ~500 characters the embedding model only
        captures the first ~256 tokens.  Chunking ensures every section of a
        long memory gets its own embedding so retrieval can match on *any*
        part of the content.
        """
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
        """
        Split *text* into overlapping chunks and embed each one.

        Returns a list of (chunk_text, embedding) tuples.
        Only generates passages when the text is long enough to benefit.
        """
        chunks = self.chunk_text(text, max_chars, overlap_chars)
        if len(chunks) <= 1:
            return []                       # no need for passage-level indexing
        embeddings = self.embed_batch(chunks)
        return list(zip(chunks, embeddings))

    # ─────────────────────────────────────────────
    # VECTORIZED SIMILARITY
    # ─────────────────────────────────────────────

    @staticmethod
    def batch_cosine_similarities(
        query: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized cosine similarity between one query vector and a matrix
        of candidate vectors.

        Args:
            query:  shape (d,)
            matrix: shape (n, d)

        Returns:
            shape (n,) of cosine similarities
        """
        if matrix.size == 0:
            return np.array([], dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(matrix.shape[0], dtype=np.float32)
        row_norms = np.linalg.norm(matrix, axis=1)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)  # avoid /0
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
