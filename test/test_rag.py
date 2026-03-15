"""Tests for the nerif.rag module. No API calls required - uses mock embeddings."""

import re
from unittest.mock import MagicMock

import numpy as np
import pytest

from nerif.rag import NumpyVectorStore, SearchResult, SimpleRAG, VectorStoreBase

# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


def test_search_result_dataclass():
    result = SearchResult(id="abc", text="hello", score=0.9, metadata={"key": "value"})
    assert result.id == "abc"
    assert result.text == "hello"
    assert result.score == 0.9
    assert result.metadata == {"key": "value"}


def test_search_result_default_metadata():
    result = SearchResult(id="abc", text="hello", score=0.5)
    assert result.metadata == {}


# ---------------------------------------------------------------------------
# VectorStoreBase ABC
# ---------------------------------------------------------------------------


def test_vector_store_base_not_instantiable():
    with pytest.raises(TypeError):
        VectorStoreBase()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# NumpyVectorStore - basic add and count
# ---------------------------------------------------------------------------


def test_numpy_store_add_and_count():
    store = NumpyVectorStore()
    ids = store.add(
        texts=["doc one", "doc two", "doc three"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert store.count() == 3
    assert len(ids) == 3


def test_numpy_store_add_empty():
    store = NumpyVectorStore()
    ids = store.add(texts=[], embeddings=[])
    assert ids == []
    assert store.count() == 0


# ---------------------------------------------------------------------------
# NumpyVectorStore - cosine similarity search
# ---------------------------------------------------------------------------


def test_numpy_store_search_cosine():
    store = NumpyVectorStore()
    store.add(
        texts=["doc A", "doc B", "doc C"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    # Query closest to doc A (along x-axis)
    results = store.search(query_embedding=[1.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3
    assert results[0].text == "doc A"
    assert results[0].score == pytest.approx(1.0, abs=1e-6)
    # doc B and doc C should have score 0
    assert results[1].score == pytest.approx(0.0, abs=1e-6)
    assert results[2].score == pytest.approx(0.0, abs=1e-6)


def test_numpy_store_search_partial_similarity():
    store = NumpyVectorStore()
    store.add(
        texts=["AB", "A only"],
        embeddings=[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    )
    # Query along x-axis: "A only" should be closest (score=1), "AB" at ~0.707
    results = store.search(query_embedding=[1.0, 0.0, 0.0], top_k=2)
    assert results[0].text == "A only"
    assert results[0].score == pytest.approx(1.0, abs=1e-6)
    assert results[1].score == pytest.approx(1.0 / np.sqrt(2), abs=1e-5)


def test_numpy_store_search_top_k_clipped():
    store = NumpyVectorStore()
    store.add(
        texts=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    results = store.search(query_embedding=[1.0, 0.0], top_k=10)
    assert len(results) == 2  # Only 2 docs available


# ---------------------------------------------------------------------------
# NumpyVectorStore - empty store search
# ---------------------------------------------------------------------------


def test_numpy_store_search_empty_store():
    store = NumpyVectorStore()
    results = store.search(query_embedding=[1.0, 0.0, 0.0], top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# NumpyVectorStore - delete
# ---------------------------------------------------------------------------


def test_numpy_store_delete():
    store = NumpyVectorStore()
    ids = store.add(
        texts=["keep", "remove", "keep2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
    )
    store.delete([ids[1]])
    assert store.count() == 2
    remaining_texts = store._texts
    assert "remove" not in remaining_texts
    assert "keep" in remaining_texts
    assert "keep2" in remaining_texts


def test_numpy_store_delete_all():
    store = NumpyVectorStore()
    ids = store.add(
        texts=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    store.delete(ids)
    assert store.count() == 0
    assert store._vectors is None


def test_numpy_store_delete_nonexistent():
    store = NumpyVectorStore()
    store.add(
        texts=["doc"],
        embeddings=[[1.0, 0.0]],
    )
    # Deleting a non-existent ID should be a no-op
    store.delete(["nonexistent-id-xyz"])
    assert store.count() == 1


# ---------------------------------------------------------------------------
# NumpyVectorStore - auto IDs
# ---------------------------------------------------------------------------


def test_numpy_store_auto_ids():
    store = NumpyVectorStore()
    ids = store.add(
        texts=["x", "y"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    for doc_id in ids:
        assert uuid_pattern.match(doc_id), f"ID {doc_id!r} is not a valid UUID"
    assert ids[0] != ids[1]


# ---------------------------------------------------------------------------
# NumpyVectorStore - dimension mismatch
# ---------------------------------------------------------------------------


def test_numpy_store_dimension_mismatch():
    store = NumpyVectorStore(dimension=3)
    with pytest.raises(ValueError, match="Expected dimension 3"):
        store.add(
            texts=["wrong dim"],
            embeddings=[[1.0, 0.0]],  # 2D, not 3D
        )


def test_numpy_store_dimension_mismatch_after_first_add():
    store = NumpyVectorStore()
    store.add(texts=["first"], embeddings=[[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Expected dimension 3"):
        store.add(texts=["second"], embeddings=[[1.0, 0.0]])  # 2D


# ---------------------------------------------------------------------------
# NumpyVectorStore - save/load roundtrip
# ---------------------------------------------------------------------------


def test_numpy_store_save_load_roundtrip(tmp_path):
    store = NumpyVectorStore()
    store.add(
        texts=["alpha", "beta"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadata=[{"tag": "a"}, {"tag": "b"}],
        ids=["id-alpha", "id-beta"],
    )

    npz_path = str(tmp_path / "store.npz")
    store.save(npz_path)

    loaded = NumpyVectorStore.load(npz_path)
    assert loaded.count() == 2
    assert loaded._dimension == 3
    assert loaded._texts == ["alpha", "beta"]
    assert loaded._ids == ["id-alpha", "id-beta"]
    assert loaded._metadata == [{"tag": "a"}, {"tag": "b"}]
    assert loaded._vectors is not None
    np.testing.assert_array_almost_equal(loaded._vectors, store._vectors)


def test_numpy_store_save_load_empty(tmp_path):
    store = NumpyVectorStore()
    npz_path = str(tmp_path / "empty.npz")
    store.save(npz_path)

    loaded = NumpyVectorStore.load(npz_path)
    assert loaded.count() == 0
    assert loaded._vectors is None


# ---------------------------------------------------------------------------
# SimpleRAG
# ---------------------------------------------------------------------------


def _make_embed_model(vectors: dict):
    """Create a mock embed model that maps text -> vector."""
    model = MagicMock()
    model.embed.side_effect = lambda text: vectors[text]
    return model


def test_simple_rag_add_texts():
    vectors = {
        "hello world": [1.0, 0.0, 0.0],
        "goodbye world": [0.0, 1.0, 0.0],
    }
    embed_model = _make_embed_model(vectors)
    store = NumpyVectorStore()
    rag = SimpleRAG(embed_model=embed_model, store=store)

    ids = rag.add_texts(texts=["hello world", "goodbye world"])
    assert store.count() == 2
    assert len(ids) == 2
    assert embed_model.embed.call_count == 2


def test_simple_rag_add_texts_with_metadata():
    vectors = {"doc": [1.0, 0.0]}
    embed_model = _make_embed_model(vectors)
    store = NumpyVectorStore()
    rag = SimpleRAG(embed_model=embed_model, store=store)

    rag.add_texts(texts=["doc"], metadata=[{"source": "test"}])
    assert store._metadata[0] == {"source": "test"}


def test_simple_rag_query():
    vectors = {
        "doc A": [1.0, 0.0, 0.0],
        "doc B": [0.0, 1.0, 0.0],
        "doc C": [0.0, 0.0, 1.0],
        "what is A?": [0.9, 0.1, 0.0],
    }
    embed_model = _make_embed_model(vectors)
    store = NumpyVectorStore()
    rag = SimpleRAG(embed_model=embed_model, store=store)

    rag.add_texts(texts=["doc A", "doc B", "doc C"])
    results = rag.query("what is A?", top_k=2)
    assert len(results) == 2
    assert results[0].text == "doc A"


def test_simple_rag_query_with_context():
    vectors = {
        "doc A": [1.0, 0.0],
        "doc B": [0.0, 1.0],
        "my question": [1.0, 0.0],
    }
    embed_model = _make_embed_model(vectors)
    store = NumpyVectorStore()
    rag = SimpleRAG(embed_model=embed_model, store=store)
    rag.add_texts(texts=["doc A", "doc B"])

    chat_model = MagicMock()
    chat_model.chat.return_value = "mocked answer"

    answer = rag.query_with_context("my question", model=chat_model, top_k=1)
    assert answer == "mocked answer"

    # Verify the prompt was built and passed to the model
    call_args = chat_model.chat.call_args[0][0]
    assert "my question" in call_args
    assert "doc A" in call_args  # top result should be doc A
    assert "Answer:" in call_args


def test_simple_rag_query_with_custom_template():
    vectors = {
        "doc": [1.0, 0.0],
        "q": [1.0, 0.0],
    }
    embed_model = _make_embed_model(vectors)
    store = NumpyVectorStore()
    rag = SimpleRAG(embed_model=embed_model, store=store)
    rag.add_texts(texts=["doc"])

    chat_model = MagicMock()
    chat_model.chat.return_value = "custom answer"

    template = "CTX: {context} | Q: {question}"
    answer = rag.query_with_context("q", model=chat_model, top_k=1, prompt_template=template)
    assert answer == "custom answer"
    call_args = chat_model.chat.call_args[0][0]
    assert call_args.startswith("CTX:")
    assert "| Q: q" in call_args
