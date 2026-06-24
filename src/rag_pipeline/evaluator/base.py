"""Evaluator interface: scores a generated response."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_pipeline.models import EvaluationResult, GeneratedResponse


class Evaluator(ABC):
    """Scores the quality of a generated response."""

    @abstractmethod
    def evaluate(
        self,
        response: GeneratedResponse,
        reference_answer: str | None = None,
    ) -> EvaluationResult:
        """Score one response, using reference_answer for ground-truth metrics when given."""
