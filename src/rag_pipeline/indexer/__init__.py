"""Indexing: splitting documents into chunks ready for storage."""

from rag_pipeline.indexer.base import Indexer
from rag_pipeline.indexer.fixed_size import FixedSizeIndexer

__all__ = ["FixedSizeIndexer", "Indexer"]
