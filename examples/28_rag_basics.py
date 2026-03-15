"""Example 28: Lightweight RAG with NumpyVectorStore.

Demonstrates the built-in vector store for simple retrieval-augmented generation.
For production use, implement VectorStoreBase with ChromaDB, FAISS, etc.
"""

from nerif.rag import NumpyVectorStore, SimpleRAG

# For a real use case, you'd use SimpleEmbeddingModel:
# from nerif.model import SimpleEmbeddingModel
# embed_model = SimpleEmbeddingModel()

# Demo with mock embeddings (no API needed)
store = NumpyVectorStore()

# Add documents with pre-computed embeddings
texts = [
    "Python is a programming language.",
    "Machine learning uses statistical models.",
    "Neural networks are inspired by the brain.",
]
# Mock 3D embeddings for demonstration
embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]

ids = store.add(texts=texts, embeddings=embeddings)
print(f"Added {store.count()} documents")

# Search
results = store.search(query_embedding=[0.9, 0.1, 0.0], top_k=2)
for r in results:
    print(f"  [{r.score:.3f}] {r.text}")

# Save and load
store.save("/tmp/nerif_rag_demo.npz")
loaded = NumpyVectorStore.load("/tmp/nerif_rag_demo.npz")
print(f"Loaded store with {loaded.count()} documents")

# SimpleRAG usage example (with mock embed model)
# In practice, replace with a real embedding model:
#   rag = SimpleRAG(embed_model=SimpleEmbeddingModel(), store=NumpyVectorStore())
#   rag.add_texts(["doc one", "doc two"])
#   results = rag.query("search query", top_k=3)
#   answer = rag.query_with_context("my question", model=SimpleChatModel())
print(f"\nSimpleRAG: {SimpleRAG.__doc__}")
