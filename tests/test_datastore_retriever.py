"""Tests for DataStoreRetriever.

Uses a fake DataStore so the retriever is tested in isolation, without
depending on a concrete (heavy) store implementation.
"""

from collections.abc import Sequence

from rag_pipeline.datastore import DataStore
from rag_pipeline.models import Chunk, ScoredChunk
from rag_pipeline.retriever import DataStoreRetriever


class FakeDataStore(DataStore):
    """Returns canned ScoredChunks and records how search was called."""

    def __init__(self, results: Sequence[ScoredChunk]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, int]] = []

    def add(self, chunks):  # pragma: no cover - unused by the retriever
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        self.calls.append((query, top_k))
        return self._results[:top_k]

    def count(self) -> int:
        return len(self._results)


def scored(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=chunk_id, doc_id="d1", content=chunk_id), score=score)


def test_forwards_query_and_top_k_to_store():
    store = FakeDataStore([scored("a", 0.9), scored("b", 0.5)])
    retriever = DataStoreRetriever(store)

    results = retriever.retrieve("hello", top_k=2)

    assert store.calls == [("hello", 2)]
    assert [r.chunk.chunk_id for r in results] == ["a", "b"]


def test_preserves_store_ordering():
    store = FakeDataStore([scored("a", 0.9), scored("b", 0.8), scored("c", 0.7)])
    results = DataStoreRetriever(store).retrieve("q", top_k=3)
    assert [r.chunk.chunk_id for r in results] == ["a", "b", "c"]


def test_score_threshold_drops_weak_results():
    store = FakeDataStore([scored("a", 0.9), scored("b", 0.4), scored("c", 0.2)])
    retriever = DataStoreRetriever(store, score_threshold=0.5)

    results = retriever.retrieve("q", top_k=3)

    assert [r.chunk.chunk_id for r in results] == ["a"]


def test_threshold_is_inclusive():
    store = FakeDataStore([scored("a", 0.5), scored("b", 0.49)])
    results = DataStoreRetriever(store, score_threshold=0.5).retrieve("q", top_k=2)
    assert [r.chunk.chunk_id for r in results] == ["a"]


def test_no_threshold_returns_everything():
    store = FakeDataStore([scored("a", 0.9), scored("b", -0.3)])
    results = DataStoreRetriever(store).retrieve("q", top_k=2)
    assert [r.chunk.chunk_id for r in results] == ["a", "b"]
