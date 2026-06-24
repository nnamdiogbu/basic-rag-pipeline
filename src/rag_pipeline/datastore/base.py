"""DataStore interface: persistence and similarity search over chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag_pipeline.models import Chunk, ScoredChunk


class DataStore(ABC):
    """Stores chunks and finds the ones most similar to a query.

    The store owns embedding: ``add`` embeds chunk content and ``search``
    embeds the query, so both sides use the same model.
    """

    @abstractmethod
    def add(self, chunks: Sequence[Chunk]) -> None:
        """Store chunks, replacing any with the same chunk_id."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return the top_k chunks most similar to the query text, best first."""

    @abstractmethod
    def count(self) -> int:
        """Number of chunks currently stored."""
