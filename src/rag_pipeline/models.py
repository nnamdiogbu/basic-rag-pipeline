"""Data models passed between pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A raw source document."""

    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a document, sized for retrieval. Embedding is optional."""

    chunk_id: str
    doc_id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A chunk returned from search, with its relevance score (higher is better)."""

    chunk: Chunk
    score: float


@dataclass
class GeneratedResponse:
    """An answer plus the context chunks used to produce it."""

    query: str
    answer: str
    context: list[ScoredChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Scores for a generated response: metric name -> value, plus optional details."""

    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
