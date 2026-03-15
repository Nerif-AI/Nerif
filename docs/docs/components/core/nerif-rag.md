---
sidebar_position: 5
---

# Lightweight RAG

Nerif provides a lightweight RAG (Retrieval Augmented Generation) interface with a built-in NumPy vector store.

## Quick Start

```python
from nerif.model import SimpleChatModel, SimpleEmbeddingModel
from nerif.rag import NumpyVectorStore, SimpleRAG

# Set up components
embed_model = SimpleEmbeddingModel()
store = NumpyVectorStore()
rag = SimpleRAG(embed_model=embed_model, store=store)

# Add documents
rag.add_texts(["Python is great", "Rust is fast", "Go is simple"])

# Query
results = rag.query("Which language is fastest?", top_k=2)
for r in results:
    print(f"[{r.score:.3f}] {r.text}")

# Query with LLM context
model = SimpleChatModel()
answer = rag.query_with_context("Which language is fastest?", model=model)
```

## Custom Vector Store

Implement `VectorStoreBase` for ChromaDB, FAISS, etc.:

```python
from nerif.rag import VectorStoreBase, SearchResult

class ChromaVectorStore(VectorStoreBase):
    def add(self, texts, embeddings, metadata=None, ids=None):
        ...
    def search(self, query_embedding, top_k=5):
        ...
    def delete(self, ids):
        ...
    def count(self):
        ...
```

## Persistence

```python
store.save("knowledge.npz")
loaded = NumpyVectorStore.load("knowledge.npz")
```
