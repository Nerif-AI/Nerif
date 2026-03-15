from .base import SearchResult, SimpleRAG, VectorStoreBase

try:
    from .numpy_store import NumpyVectorStore
except ImportError:
    NumpyVectorStore = None  # numpy not installed; use pip install nerif[rag]

__all__ = ["VectorStoreBase", "NumpyVectorStore", "SimpleRAG", "SearchResult"]
