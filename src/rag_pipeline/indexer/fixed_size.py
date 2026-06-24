"""Indexer that splits documents into fixed-size, overlapping character windows."""

from __future__ import annotations

from collections.abc import Sequence

from rag_pipeline.indexer.base import Indexer
from rag_pipeline.models import Chunk, Document


class FixedSizeIndexer(Indexer):
    """Splits document text into fixed-size character windows with overlap.

    Overlap carries context across chunk boundaries so a passage split
    between two chunks is still retrievable. Embeddings are left unset;
    the datastore embeds chunk content.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Characters shared between consecutive chunks;
                must be less than chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap} "
                f"with chunk_size {chunk_size}"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def index(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._chunk_document(document))
        return chunks

    def _chunk_document(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for start, end in self._windows(len(document.content)):
            content = document.content[start:end].strip()
            if not content:
                continue
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{index}",
                    doc_id=document.doc_id,
                    content=content,
                    embedding=None,
                    metadata={
                        **document.metadata,
                        "chunk_index": index,
                        "char_start": start,
                        "char_end": end,
                    },
                )
            )
        return chunks

    def _windows(self, length: int) -> list[tuple[int, int]]:
        """Return (start, end) character offsets for each window.

        Stops once a window reaches the end so the final window is not a
        redundant slice already covered by the previous one's overlap.
        """
        step = self._chunk_size - self._chunk_overlap
        windows: list[tuple[int, int]] = []
        start = 0
        while start < length:
            end = min(start + self._chunk_size, length)
            windows.append((start, end))
            if end == length:
                break
            start += step
        return windows
