"""Retriever interface: natural-language query in, relevant chunks out."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_pipeline.models import ScoredChunk


class Retriever(ABC):
    """Finds the chunks most relevant to a user query."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return the top_k chunks most relevant to the query, best first."""
