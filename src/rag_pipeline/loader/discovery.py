"""Shared file discovery for path-based loaders."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path


def discover_files(paths: Sequence[str | Path], suffixes: Collection[str]) -> list[Path]:
    """Resolve paths to a sorted list of files with a supported suffix.

    Directories are scanned recursively and non-matching files silently
    skipped; an explicitly listed file with an unsupported suffix is an
    error, as is a path that does not exist. Suffix matching is
    case-insensitive.
    """
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: {path}")
        if path.is_dir():
            files.extend(
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in suffixes
            )
        elif path.suffix.lower() in suffixes:
            files.append(path)
        else:
            raise ValueError(
                f"Unsupported file type {path.suffix!r}: {path}. "
                f"Supported: {', '.join(sorted(suffixes))}"
            )
    return files
