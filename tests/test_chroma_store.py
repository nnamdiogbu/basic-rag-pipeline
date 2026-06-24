"""Tests for ChromaStore.

Requires chromadb; tests skip when the optional dependency is missing.
Chroma embeds text internally (default all-MiniLM-L6-v2), so these are
real semantic-similarity checks rather than exact-vector arithmetic.
"""

import uuid

import pytest

from rag_pipeline.models import Chunk

pytest.importorskip("chromadb")

from rag_pipeline.datastore import ChromaStore  # noqa: E402


def make_chunk(chunk_id, content, doc_id="d1", metadata=None):
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        metadata=metadata or {},
    )


@pytest.fixture
def store():
    # chromadb's EphemeralClient is a shared in-process singleton, so a
    # unique collection name per test keeps them isolated.
    return ChromaStore(collection_name=f"test_{uuid.uuid4().hex}")


def test_empty_store_counts_zero_and_searches_nothing(store):
    assert store.count() == 0
    assert store.search("anything") == []


def test_add_and_count(store):
    store.add([make_chunk("a", "alpha"), make_chunk("b", "beta")])
    assert store.count() == 2


def test_search_ranks_by_semantic_similarity(store):
    store.add(
        [
            make_chunk("banana", "Bananas are a yellow fruit."),
            make_chunk("ocean", "The ocean is full of salt water."),
        ]
    )
    results = store.search("What colour is a banana?", top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["banana", "ocean"]
    assert results[0].score > results[1].score


def test_search_respects_top_k_and_clamps_to_count(store):
    store.add([make_chunk("a", "alpha"), make_chunk("b", "beta")])
    assert len(store.search("alpha", top_k=1)) == 1
    # top_k larger than the collection returns everything, no error
    assert len(store.search("alpha", top_k=10)) == 2


def test_upsert_replaces_same_chunk_id(store):
    store.add([make_chunk("a", "the first version")])
    store.add([make_chunk("a", "the second version")])
    assert store.count() == 1
    assert store.search("version", top_k=1)[0].chunk.content == "the second version"


def test_metadata_and_doc_id_round_trip(store):
    store.add(
        [make_chunk("a", "hello", doc_id="doc-7", metadata={"source": "a.md", "chunk_index": 3})]
    )
    chunk = store.search("hello", top_k=1)[0].chunk
    assert chunk.doc_id == "doc-7"
    assert chunk.metadata == {"source": "a.md", "chunk_index": 3}
    # embeddings live inside Chroma, not surfaced back to the pipeline
    assert chunk.embedding is None


def test_add_empty_is_noop(store):
    store.add([])
    assert store.count() == 0
