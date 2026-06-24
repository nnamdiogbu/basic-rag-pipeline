"""Loading: extracting documents from external sources."""

from rag_pipeline.loader.audio import AudioLoader
from rag_pipeline.loader.base import DocumentLoader
from rag_pipeline.loader.docling import DoclingLoader

__all__ = ["AudioLoader", "DoclingLoader", "DocumentLoader"]
