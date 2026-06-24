"""Sanity checks for the pipeline interfaces.

Uses deliberately naive fake implementations to verify that the five
interfaces compose into a working end-to-end pipeline. Real
implementations will replace these one piece at a time.
"""

from collections.abc import Sequence

import pytest

from rag_pipeline import (
    Chunk,
    DataStore,
    Document,
    DocumentLoader,
    EvaluationResult,
    Evaluator,
    GeneratedResponse,
    Indexer,
    ResponseGenerator,
    Retriever,
    ScoredChunk,
)


def embed(text: str) -> list[float]:
    """Toy embedding: counts of a few letters. Just enough for tests."""
    return [float(text.lower().count(letter)) for letter in "etaoin"]


class StaticLoader(DocumentLoader):
    """Serves a fixed list of documents, standing in for a real source."""

    def __init__(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)

    def load(self) -> list[Document]:
        return list(self._documents)


class WholeDocumentIndexer(Indexer):
    """One chunk per document; embedding is left to the datastore."""

    def index(self, documents: Sequence[Document]) -> list[Chunk]:
        return [
            Chunk(chunk_id=f"{doc.doc_id}:0", doc_id=doc.doc_id, content=doc.content)
            for doc in documents
        ]


class InMemoryDataStore(DataStore):
    """Brute-force dot-product search; embeds chunk text and the query itself."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        query_embedding = embed(query)
        scored = [
            ScoredChunk(
                chunk=c,
                score=sum(q * v for q, v in zip(query_embedding, embed(c.content))),
            )
            for c in self._chunks.values()
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)


class PassthroughRetriever(Retriever):
    """Forwards the query to the datastore, which owns embedding."""

    def __init__(self, datastore: DataStore) -> None:
        self._datastore = datastore

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        return self._datastore.search(query, top_k=top_k)


class EchoGenerator(ResponseGenerator):
    """Answers with the single most relevant chunk's content."""

    def generate(self, query: str, context: Sequence[ScoredChunk]) -> GeneratedResponse:
        answer = context[0].chunk.content if context else "I don't know."
        return GeneratedResponse(query=query, answer=answer, context=list(context))


class ExactMatchEvaluator(Evaluator):
    """1.0 if the answer equals the reference, else 0.0."""

    def evaluate(
        self, response: GeneratedResponse, reference_answer: str | None = None
    ) -> EvaluationResult:
        if reference_answer is None:
            return EvaluationResult(metrics={})
        match = float(response.answer.strip() == reference_answer.strip())
        return EvaluationResult(metrics={"exact_match": match})


def test_interfaces_are_abstract() -> None:
    for interface in (DocumentLoader, Indexer, DataStore, Retriever, ResponseGenerator, Evaluator):
        with pytest.raises(TypeError):
            interface()  # type: ignore[abstract]


def test_end_to_end_with_fakes() -> None:
    loader = StaticLoader(
        [
            Document(doc_id="d1", content="Bananas are yellow."),
            Document(doc_id="d2", content="The sky is blue."),
        ]
    )
    documents = loader.load()

    indexer = WholeDocumentIndexer()
    datastore = InMemoryDataStore()
    datastore.add(indexer.index(documents))
    assert datastore.count() == 2

    retriever = PassthroughRetriever(datastore)
    context = retriever.retrieve("Bananas are what color?", top_k=1)
    assert len(context) == 1
    assert context[0].chunk.doc_id == "d1"

    response = EchoGenerator().generate("Bananas are what color?", context)
    assert response.answer == "Bananas are yellow."
    assert response.context == context

    result = ExactMatchEvaluator().evaluate(response, reference_answer="Bananas are yellow.")
    assert result.metrics["exact_match"] == 1.0
