"""Document loader that transcribes audio files with Whisper.

Turns spoken audio into plain-text ``Document`` objects via
faster-whisper, which runs locally. Optional dependency: install
with ``pip install "rag-pipeline[audio]"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_pipeline.loader.base import DocumentLoader
from rag_pipeline.loader.discovery import discover_files
from rag_pipeline.models import Document

# Common audio containers faster-whisper can decode. Directory scans
# silently skip anything else; explicit files with other suffixes error.
SUPPORTED_SUFFIXES = frozenset(
    {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".aac", ".wma"}
)


class AudioLoader(DocumentLoader):
    """Loads documents by transcribing audio files or directories.

    Directories are scanned recursively for supported audio types. Each
    file's transcript becomes one ``Document``.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        model_size: str = "base",
        model: Any | None = None,
    ) -> None:
        """
        Args:
            paths: Audio files (suffix in SUPPORTED_SUFFIXES) and/or
                directories to scan recursively.
            model_size: Whisper model to load (e.g. tiny, base, small,
                medium, large-v3); larger is more accurate but slower.
            model: A preloaded faster-whisper ``WhisperModel``; built from
                model_size on first use if not supplied.
        """
        self._paths = [Path(p) for p in paths]
        self._model_size = model_size
        self._model = model

    def load(self) -> list[Document]:
        files = discover_files(self._paths, SUPPORTED_SUFFIXES)
        if not files:
            return []
        model = self._ensure_model()
        documents: list[Document] = []
        for path in files:
            segments, info = model.transcribe(str(path))
            text = " ".join(segment.text.strip() for segment in segments).strip()
            documents.append(
                Document(
                    doc_id=str(path),
                    content=text,
                    metadata={
                        "source": str(path),
                        "format": path.suffix.lstrip("."),
                        "language": getattr(info, "language", None),
                        "duration": getattr(info, "duration", None),
                    },
                )
            )
        return documents

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _load_model(self._model_size)
        return self._model


def _load_model(model_size: str) -> Any:
    # Imported lazily so the core package works without faster-whisper.
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "AudioLoader requires faster-whisper. "
            "Install it with: pip install 'rag-pipeline[audio]'"
        ) from exc
    # CPU + int8 is the portable default (e.g. Apple Silicon, where
    # CTranslate2 has no GPU backend); override by passing model=.
    return WhisperModel(model_size, device="cpu", compute_type="int8")
