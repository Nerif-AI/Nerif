import json
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from .base import SearchResult, VectorStoreBase


class NumpyVectorStore(VectorStoreBase):
    """In-memory vector store using numpy cosine similarity. No extra dependencies beyond numpy."""

    def __init__(self, dimension: Optional[int] = None):
        self._dimension = dimension
        self._vectors: Optional[np.ndarray] = None  # shape: (n, dimension)
        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._ids: List[str] = []

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not texts:
            return []

        vectors = np.array(embeddings, dtype=np.float32)
        if self._dimension is None:
            self._dimension = vectors.shape[1]
        elif vectors.shape[1] != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}, got {vectors.shape[1]}")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadata is None:
            metadata = [{} for _ in texts]

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

        self._texts.extend(texts)
        self._metadata.extend(metadata)
        self._ids.extend(ids)
        return ids

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        if self._vectors is None or len(self._ids) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        # Cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1)
        query_norm = np.linalg.norm(query)
        # Avoid division by zero
        valid = (norms > 0) & (query_norm > 0)
        similarities = np.zeros(len(self._ids))
        if valid.any():
            similarities[valid] = (self._vectors[valid] @ query) / (norms[valid] * query_norm)

        top_k = min(top_k, len(self._ids))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            SearchResult(
                id=self._ids[i],
                text=self._texts[i],
                score=float(similarities[i]),
                metadata=self._metadata[i],
            )
            for i in top_indices
        ]

    def delete(self, ids: List[str]) -> None:
        ids_set = set(ids)
        indices_to_keep = [i for i, doc_id in enumerate(self._ids) if doc_id not in ids_set]
        if len(indices_to_keep) == len(self._ids):
            return  # Nothing to delete

        if not indices_to_keep:
            self._vectors = None
            self._texts = []
            self._metadata = []
            self._ids = []
            return

        self._vectors = self._vectors[indices_to_keep]
        self._texts = [self._texts[i] for i in indices_to_keep]
        self._metadata = [self._metadata[i] for i in indices_to_keep]
        self._ids = [self._ids[i] for i in indices_to_keep]

    def count(self) -> int:
        return len(self._ids)

    def save(self, path: str) -> None:
        """Save to .npz file with JSON sidecar for metadata."""
        if self._vectors is not None:
            np.savez(path, vectors=self._vectors)
        else:
            np.savez(path, vectors=np.array([]))

        sidecar = path + ".meta.json"
        with open(sidecar, "w") as f:
            json.dump(
                {
                    "dimension": self._dimension,
                    "texts": self._texts,
                    "metadata": self._metadata,
                    "ids": self._ids,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "NumpyVectorStore":
        """Load from .npz file with JSON sidecar."""
        data = np.load(path)
        vectors = data["vectors"]

        sidecar = path + ".meta.json"
        with open(sidecar) as f:
            meta = json.load(f)

        store = cls(dimension=meta["dimension"])
        if vectors.size > 0:
            store._vectors = vectors.astype(np.float32)
        store._texts = meta["texts"]
        store._metadata = meta["metadata"]
        store._ids = meta["ids"]
        return store
