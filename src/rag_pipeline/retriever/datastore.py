"""Retriever that delegates the query to a DataStore."""

from __future__ import annotations

from rag_pipeline.datastore.base import DataStore
from rag_pipeline.models import ScoredChunk
from rag_pipeline.retriever.base import Retriever


class DataStoreRetriever(Retriever):
    """Retrieves chunks by forwarding the query to a DataStore.

    The datastore owns embedding and similarity search; this retriever
    passes the query through and optionally drops results scoring below a
    threshold to suppress weakly-relevant chunks.
    """

    def __init__(self, datastore: DataStore, score_threshold: float | None = None) -> None:
        """
        Args:
            datastore: Backing store that embeds the query and searches.
            score_threshold: If set, results with a score below it are
                dropped.
        """
        self._datastore = datastore
        self._score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        results = self._datastore.search(query, top_k=top_k)
        if self._score_threshold is not None:
            results = [r for r in results if r.score >= self._score_threshold]
        return results
