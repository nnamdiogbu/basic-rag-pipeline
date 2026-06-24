"""Tests for FixedSizeIndexer."""

import pytest

from rag_pipeline.indexer import FixedSizeIndexer
from rag_pipeline.models import Document


def test_short_document_yields_single_chunk():
    doc = Document(doc_id="d1", content="Bananas are yellow.")
    chunks = FixedSizeIndexer(chunk_size=1000, chunk_overlap=200).index([doc])

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1:0"
    assert chunks[0].doc_id == "d1"
    assert chunks[0].content == "Bananas are yellow."
    assert chunks[0].embedding is None


def test_long_document_splits_with_overlap():
    content = "".join(str(i % 10) for i in range(100))  # 100 chars
    doc = Document(doc_id="d1", content=content)
    chunks = FixedSizeIndexer(chunk_size=40, chunk_overlap=10).index([doc])

    # step = 30: windows (0,40), (30,70), (60,100); the third reaches the
    # end, so iteration stops at 3 chunks rather than emitting a (90,100) slice.
    assert [c.chunk_id for c in chunks] == ["d1:0", "d1:1", "d1:2"]
    # consecutive chunks share `chunk_overlap` characters
    assert chunks[0].content[-10:] == chunks[1].content[:10]
    # offsets recorded in metadata reconstruct the original text
    for chunk in chunks:
        start, end = chunk.metadata["char_start"], chunk.metadata["char_end"]
        assert chunk.content == content[start:end]


def test_no_redundant_trailing_chunk():
    content = "x" * 1000
    chunks = FixedSizeIndexer(chunk_size=1000, chunk_overlap=200).index(
        [Document(doc_id="d1", content=content)]
    )
    assert len(chunks) == 1


def test_empty_and_whitespace_documents_yield_no_chunks():
    docs = [Document(doc_id="empty", content=""), Document(doc_id="blank", content="   \n\t  ")]
    assert FixedSizeIndexer().index(docs) == []


def test_chunks_preserve_document_order_and_metadata():
    docs = [
        Document(doc_id="a", content="alpha", metadata={"source": "a.md"}),
        Document(doc_id="b", content="beta", metadata={"source": "b.md"}),
    ]
    chunks = FixedSizeIndexer().index(docs)

    assert [c.doc_id for c in chunks] == ["a", "b"]
    assert chunks[0].metadata["source"] == "a.md"
    assert chunks[0].metadata["chunk_index"] == 0


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap",
    [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_parameters_raise(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        FixedSizeIndexer(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
