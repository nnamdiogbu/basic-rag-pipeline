"""Document loader backed by docling (optional: ``pip install "rag-pipeline[docs]"``)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_pipeline.loader.base import DocumentLoader
from rag_pipeline.loader.discovery import discover_files
from rag_pipeline.models import Document

# Formats docling can convert. Directory scans silently skip anything
# else; explicitly listed files with other suffixes are an error.
SUPPORTED_SUFFIXES = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".md", ".html", ".htm", ".csv", ".adoc", ".asciidoc"}
)


class DoclingLoader(DocumentLoader):
    """Loads documents from files or directories using docling, exported as Markdown.

    Directories are scanned recursively for supported file types.
    """

    def __init__(self, paths: Sequence[str | Path]) -> None:
        """
        Args:
            paths: Files (suffix in SUPPORTED_SUFFIXES) and/or directories
                to scan recursively.
        """
        self._paths = [Path(p) for p in paths]

    def load(self) -> list[Document]:
        files = discover_files(self._paths, SUPPORTED_SUFFIXES)
        if not files:
            return []
        converter = _make_converter()
        documents: list[Document] = []
        for path in files:
            result = converter.convert(path)
            documents.append(
                Document(
                    doc_id=str(path),
                    content=result.document.export_to_markdown(),
                    metadata={"source": str(path), "format": path.suffix.lstrip(".")},
                )
            )
        return documents


def _make_converter() -> Any:
    # Imported lazily so the core package works without the heavy
    # docling/torch dependency chain installed.
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError(
            "DoclingLoader requires docling. Install it with: pip install 'rag-pipeline[docs]'"
        ) from exc
    return DocumentConverter()
