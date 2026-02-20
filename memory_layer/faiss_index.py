"""
FAISS Vector Index for the Memory Layer.

Provides fast approximate nearest-neighbor search using Facebook's FAISS
library.  Automatically selects the best index type for the collection size:

  - Small  (≤ 10 000 vectors):  IndexFlatIP  — exact, SIMD-optimised
  - Large  (> 10 000 vectors):  IndexIVFFlat — approximate, very fast

Gracefully falls back to brute-force numpy when FAISS is not installed so
the system still works (just slower at scale).

Persistence:
  Index files are saved alongside the SQLite DB as ``<db>.faiss`` and
  ``<db>.faiss.meta.json``.  They are rebuilt automatically from SQLite on
  first load or after an embedding model migration.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

# Prevent OpenMP crash on macOS when both FAISS and sentence-transformers
# link to libomp.  This must be set *before* importing faiss.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import faiss                      # type: ignore[import-untyped]
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class MemoryIndex:
    """
    Manages one FAISS vector index with string-ID ↔ integer mapping.

    Two instances are used by :class:`MemoryManager`:
      • ``memory_index``  – one vector per stored memory
      • ``passage_index`` – one vector per passage chunk (maps back to parent
        memory ID via *passage_to_memory*)
    """

    # When the collection exceeds this size, switch to an IVF index.
    IVF_THRESHOLD = 10_000
    IVF_NLIST     = 100
    IVF_NPROBE    = 10

    def __init__(self, dimension: int, index_path: Optional[str] = None):
        self.dimension   = dimension
        self.index_path  = index_path

        # ID mapping  (string ID → sequential int, and back)
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx   = 0
        self._removed: set = set()      # lazy-deletion set of FAISS row indices

        self._using_faiss = FAISS_AVAILABLE
        self.index        = None
        self._index_type  = "none"

        # Numpy fallback storage
        self._np_vectors: Dict[str, np.ndarray] = {}

        if self._using_faiss:
            self._init_flat()

    # ──────────────────────────────────────────
    #  Index lifecycle
    # ──────────────────────────────────────────

    def _init_flat(self):
        """Create a new Flat (exact) inner-product index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self._index_type = "flat"

    def _init_ivf(self, training_vectors: np.ndarray):
        """Create an IVF index and train it on *training_vectors*."""
        nlist = min(self.IVF_NLIST, max(1, training_vectors.shape[0] // 40))
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(
            quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT,
        )
        self.index.nprobe = self.IVF_NPROBE
        self.index.train(training_vectors)
        self._index_type = "ivf"

    # ──────────────────────────────────────────
    #  Add / Remove / Search
    # ──────────────────────────────────────────

    def add(self, item_id: str, embedding: List[float]):
        """Add (or replace) a vector in the index."""
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)

        if not self._using_faiss:
            # Numpy fallback — just normalise and store
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._np_vectors[item_id] = vec.flatten()
            return

        faiss.normalize_L2(vec)

        # If this ID is already indexed, mark the old row as deleted
        if item_id in self._id_to_idx:
            self._removed.add(self._id_to_idx[item_id])

        idx = self._next_idx
        self._next_idx += 1
        self._id_to_idx[item_id] = idx
        self._idx_to_id[idx] = item_id

        self.index.add(vec)

    def remove(self, item_id: str):
        """Lazy-remove a vector from the index."""
        if not self._using_faiss:
            self._np_vectors.pop(item_id, None)
            return

        if item_id in self._id_to_idx:
            old_idx = self._id_to_idx.pop(item_id)
            self._removed.add(old_idx)
            self._idx_to_id.pop(old_idx, None)

    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Return the *k* nearest neighbours as ``[(item_id, score), …]``.

        Scores are cosine similarities (inner product of L2-normalised
        vectors), ranging from –1 to +1.
        """
        vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        if not self._using_faiss:
            return self._numpy_search(vec, k)

        faiss.normalize_L2(vec)

        # Ask for extra results to compensate for lazy-deleted rows
        search_k = min(k + len(self._removed) + 10, max(1, self.index.ntotal))
        if search_k == 0 or self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(vec, search_k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if idx in self._removed:
                continue
            item_id = self._idx_to_id.get(int(idx))
            if item_id is None:
                continue
            results.append((item_id, float(score)))
            if len(results) >= k:
                break
        return results

    # ──────────────────────────────────────────
    #  Bulk operations
    # ──────────────────────────────────────────

    def build_from_dict(self, embeddings: Dict[str, List[float]]):
        """
        (Re-)build the entire index from a ``{id: embedding}`` dict.

        Automatically picks Flat vs IVF depending on collection size.
        """
        self.clear()

        if not embeddings:
            return

        ids   = list(embeddings.keys())
        vecs  = np.array(
            [embeddings[i] for i in ids], dtype=np.float32,
        )
        faiss.normalize_L2(vecs) if self._using_faiss else None

        if self._using_faiss:
            if len(ids) > self.IVF_THRESHOLD:
                self._init_ivf(vecs)
            else:
                self._init_flat()
            self.index.add(vecs)
            for seq, item_id in enumerate(ids):
                self._id_to_idx[item_id] = seq
                self._idx_to_id[seq] = item_id
            self._next_idx = len(ids)
        else:
            for item_id, vec_row in zip(ids, vecs):
                norm = np.linalg.norm(vec_row)
                self._np_vectors[item_id] = (
                    vec_row / norm if norm > 0 else vec_row
                )

    def clear(self):
        """Reset the index to empty."""
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._removed.clear()
        self._next_idx = 0
        self._np_vectors.clear()
        if self._using_faiss:
            self._init_flat()

    # ──────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────

    def save(self):
        """Write the FAISS index + metadata to disk."""
        if not self.index_path or not self._using_faiss or self.index is None:
            return
        try:
            faiss.write_index(self.index, self.index_path + ".faiss")
            meta = {
                "id_to_idx":  self._id_to_idx,
                "idx_to_id":  {str(k): v for k, v in self._idx_to_id.items()},
                "next_idx":   self._next_idx,
                "removed":    list(self._removed),
                "dimension":  self.dimension,
                "index_type": self._index_type,
            }
            with open(self.index_path + ".meta.json", "w") as fh:
                json.dump(meta, fh)
        except Exception:
            pass   # non-critical — index is rebuilt from SQLite on next load

    def load(self) -> bool:
        """
        Try to load a previously saved index.  Returns ``True`` on success.
        """
        if (
            not self._using_faiss
            or not self.index_path
            or not os.path.exists(self.index_path + ".faiss")
            or not os.path.exists(self.index_path + ".meta.json")
        ):
            return False

        try:
            self.index = faiss.read_index(self.index_path + ".faiss")
            with open(self.index_path + ".meta.json") as fh:
                meta = json.load(fh)
            self._id_to_idx = meta["id_to_idx"]
            self._idx_to_id = {int(k): v for k, v in meta["idx_to_id"].items()}
            self._next_idx  = meta["next_idx"]
            self._removed   = set(meta.get("removed", []))
            self._index_type = meta.get("index_type", "flat")

            # Verify dimension matches
            if meta.get("dimension") != self.dimension:
                self.clear()
                return False

            return True
        except Exception:
            self.clear()
            return False

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────

    @property
    def size(self) -> int:
        if self._using_faiss:
            return (self.index.ntotal if self.index else 0) - len(self._removed)
        return len(self._np_vectors)

    def needs_compaction(self) -> bool:
        """True when >30 % of rows are tombstoned — time to rebuild."""
        if not self._using_faiss or self.index is None:
            return False
        total = self.index.ntotal
        return total > 0 and len(self._removed) > max(500, int(total * 0.3))

    def _numpy_search(
        self, vec: np.ndarray, k: int,
    ) -> List[Tuple[str, float]]:
        """Brute-force numpy fallback."""
        if not self._np_vectors:
            return []
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vec = vec.flatten()

        scored: List[Tuple[str, float]] = []
        for mid, mvec in self._np_vectors.items():
            scored.append((mid, float(np.dot(vec, mvec))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
