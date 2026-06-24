"""Indexer interface: documents in, chunks out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag_pipeline.models import Chunk, Document


class Indexer(ABC):
    """Transforms raw documents into chunks ready for storage."""

    @abstractmethod
    def index(self, documents: Sequence[Document]) -> list[Chunk]:
        """Split (and optionally embed) documents into chunks, in document order."""
