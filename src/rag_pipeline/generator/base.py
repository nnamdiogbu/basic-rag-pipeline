"""ResponseGenerator interface: query + retrieved context in, answer out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag_pipeline.models import GeneratedResponse, ScoredChunk


class ResponseGenerator(ABC):
    """Produces an answer to a query, grounded in retrieved context."""

    @abstractmethod
    def generate(self, query: str, context: Sequence[ScoredChunk]) -> GeneratedResponse:
        """Answer the query using the given context (most relevant first)."""
