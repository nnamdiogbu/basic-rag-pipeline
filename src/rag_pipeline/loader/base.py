"""DocumentLoader interface: external source in, documents out."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_pipeline.models import Document


class DocumentLoader(ABC):
    """Extracts documents from an external source."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Extract all documents from the configured source."""
