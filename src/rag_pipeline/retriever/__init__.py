"""Retrieval: mapping a natural-language query to relevant chunks."""

from rag_pipeline.retriever.base import Retriever
from rag_pipeline.retriever.datastore import DataStoreRetriever

__all__ = ["DataStoreRetriever", "Retriever"]
