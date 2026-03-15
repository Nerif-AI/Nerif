from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStoreBase(ABC):
    @abstractmethod
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]: ...

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]: ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None: ...

    @abstractmethod
    def count(self) -> int: ...


class SimpleRAG:
    """Convenience class combining an embedding model with a vector store."""

    def __init__(self, embed_model, store: VectorStoreBase):
        # embed_model is nerif.model.SimpleEmbeddingModel but we use duck typing
        # to avoid circular imports - just needs .embed(text) -> List[float]
        self.embed_model = embed_model
        self.store = store

    def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        embeddings = [self.embed_model.embed(t) for t in texts]
        return self.store.add(texts=texts, embeddings=embeddings, metadata=metadata)

    def query(self, question: str, top_k: int = 5) -> List[SearchResult]:
        query_embedding = self.embed_model.embed(question)
        return self.store.search(query_embedding, top_k=top_k)

    def query_with_context(
        self,
        question: str,
        model,
        top_k: int = 5,
        prompt_template: Optional[str] = None,
    ) -> str:
        """Query the store and use results as context for an LLM response.

        Args:
            question: The user's question.
            model: A SimpleChatModel instance (or anything with .chat(str) -> str).
            top_k: Number of results to retrieve.
            prompt_template: Custom template with {context} and {question} placeholders.
        """
        results = self.query(question, top_k=top_k)
        context = "\n\n".join(f"[{i + 1}] {r.text}" for i, r in enumerate(results))

        if prompt_template is None:
            prompt_template = (
                "Answer the question based on the following context.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )

        prompt = prompt_template.format(context=context, question=question)
        return model.chat(prompt)
