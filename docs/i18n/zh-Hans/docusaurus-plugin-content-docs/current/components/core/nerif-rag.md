---
sidebar_position: 5
---

# 轻量级 RAG

Nerif 提供轻量级的 RAG（检索增强生成）接口，内置基于 NumPy 的向量存储。

## 快速开始

```python
from nerif.model import SimpleChatModel, SimpleEmbeddingModel
from nerif.rag import NumpyVectorStore, SimpleRAG

# 设置组件
embed_model = SimpleEmbeddingModel()
store = NumpyVectorStore()
rag = SimpleRAG(embed_model=embed_model, store=store)

# 添加文档
rag.add_texts(["Python is great", "Rust is fast", "Go is simple"])

# 查询
results = rag.query("Which language is fastest?", top_k=2)
for r in results:
    print(f"[{r.score:.3f}] {r.text}")

# 结合 LLM 上下文查询
model = SimpleChatModel()
answer = rag.query_with_context("Which language is fastest?", model=model)
```

## 自定义向量存储

实现 `VectorStoreBase` 以支持 ChromaDB、FAISS 等：

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

## 持久化

```python
store.save("knowledge.npz")
loaded = NumpyVectorStore.load("knowledge.npz")
```
