"""Tests for AudioLoader.

A fake transcription model stands in for faster-whisper, so the loader's
discovery and Document-building behaviour is tested without the optional
dependency or real audio files.
"""

from types import SimpleNamespace

import pytest

from rag_pipeline.loader import AudioLoader


class FakeModel:
    """Returns canned segments; records the paths it was asked to transcribe."""

    def __init__(self, segments, language="en", duration=1.5):
        self._segments = segments
        self._info = SimpleNamespace(language=language, duration=duration)
        self.transcribed = []

    def transcribe(self, path):
        self.transcribed.append(path)
        return ([SimpleNamespace(text=t) for t in self._segments], self._info)


def write_audio(path):
    # Discovery only checks the suffix and that the file exists, not content.
    path.write_bytes(b"\x00\x00")
    return path


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AudioLoader([tmp_path / "nope.mp3"], model=FakeModel(["x"])).load()


def test_unsupported_explicit_file_raises(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("plain text")
    with pytest.raises(ValueError, match="Unsupported file type"):
        AudioLoader([txt], model=FakeModel(["x"])).load()


def test_empty_directory_loads_nothing(tmp_path):
    assert AudioLoader([tmp_path], model=FakeModel(["x"])).load() == []


def test_transcribes_audio_file(tmp_path):
    clip = write_audio(tmp_path / "talk.mp3")
    model = FakeModel(["Hello ", " world"], language="en", duration=2.0)

    documents = AudioLoader([clip], model=model).load()

    assert len(documents) == 1
    doc = documents[0]
    assert doc.doc_id == str(clip)
    assert doc.content == "Hello world"
    assert doc.metadata == {
        "source": str(clip),
        "format": "mp3",
        "language": "en",
        "duration": 2.0,
    }
    assert model.transcribed == [str(clip)]


def test_directory_scan_skips_unsupported(tmp_path):
    write_audio(tmp_path / "a.mp3")
    (tmp_path / "sub").mkdir()
    write_audio(tmp_path / "sub" / "b.wav")
    (tmp_path / "notes.txt").write_text("ignored")

    documents = AudioLoader([tmp_path], model=FakeModel(["hi"])).load()

    assert {d.doc_id for d in documents} == {
        str(tmp_path / "a.mp3"),
        str(tmp_path / "sub" / "b.wav"),
    }
