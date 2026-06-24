"""DataStore backed by Chroma.

Stores chunk text and runs cosine similarity search. Chroma embeds
content on ``add`` and the query on ``search`` using its built-in
embedding function (the default all-MiniLM-L6-v2 model), so the same
model is used on both sides. Chroma is an optional dependency: install
with ``pip install "rag-pipeline[chroma]"``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from rag_pipeline.datastore.base import DataStore
from rag_pipeline.models import Chunk, ScoredChunk

# Reserved metadata key holding the JSON-serialized original chunk
# metadata, since Chroma metadata values must be flat scalars.
_METADATA_KEY = "_chunk_metadata"


class ChromaStore(DataStore):
    """Stores chunks in a Chroma collection and searches by cosine similarity."""

    def __init__(
        self,
        collection_name: str = "rag_pipeline",
        persist_directory: str | None = None,
    ) -> None:
        """
        Args:
            collection_name: Name of the Chroma collection to use.
            persist_directory: Directory for on-disk persistence; if None,
                an ephemeral in-memory client is used.
        """
        chromadb = _import_chromadb()
        client = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory is not None
            else chromadb.EphemeralClient()
        )
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append(
                {"doc_id": chunk.doc_id, _METADATA_KEY: json.dumps(chunk.metadata)}
            )
        # Pass documents (not vectors): Chroma embeds them internally.
        # upsert so re-adding a chunk_id replaces it, per the interface.
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        count = self.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        scored: list[ScoredChunk] = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=metadata["doc_id"],
                content=document,
                # Embeddings live inside Chroma; not surfaced to the pipeline.
                embedding=None,
                metadata=json.loads(metadata[_METADATA_KEY]),
            )
            # Chroma returns cosine distance; convert to similarity so
            # higher means more relevant, per ScoredChunk.
            scored.append(ScoredChunk(chunk=chunk, score=1.0 - distance))
        return scored

    def count(self) -> int:
        return self._collection.count()


def _import_chromadb() -> Any:
    # Imported lazily so the core package works without chromadb installed.
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "ChromaStore requires chromadb. Install it with: pip install 'rag-pipeline[chroma]'"
        ) from exc
    return chromadb
