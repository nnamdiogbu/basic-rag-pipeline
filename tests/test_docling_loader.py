"""Tests for DoclingLoader.

File-discovery behaviour needs no docling install; conversion tests
skip when the optional dependency is missing.
"""

import pytest

from rag_pipeline.loader import DoclingLoader


def test_missing_path_raises(tmp_path):
    loader = DoclingLoader([tmp_path / "nope.pdf"])
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_unsupported_explicit_file_raises(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("plain text")
    with pytest.raises(ValueError, match="Unsupported file type"):
        DoclingLoader([txt]).load()


def test_empty_directory_loads_nothing(tmp_path):
    assert DoclingLoader([tmp_path]).load() == []


def test_load_markdown_file(tmp_path):
    pytest.importorskip("docling")
    md = tmp_path / "notes.md"
    md.write_text("# Fruit facts\n\nBananas are yellow.\n")

    documents = DoclingLoader([md]).load()

    assert len(documents) == 1
    doc = documents[0]
    assert doc.doc_id == str(md)
    assert "Bananas are yellow." in doc.content
    assert doc.metadata["source"] == str(md)
    assert doc.metadata["format"] == "md"


def test_directory_scan_skips_unsupported(tmp_path):
    pytest.importorskip("docling")
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# B\n\nBeta.\n")
    (tmp_path / "ignored.txt").write_text("not supported")

    documents = DoclingLoader([tmp_path]).load()

    assert sorted(d.metadata["format"] for d in documents) == ["md", "md"]
    assert {d.doc_id for d in documents} == {
        str(tmp_path / "a.md"),
        str(tmp_path / "sub" / "b.md"),
    }
