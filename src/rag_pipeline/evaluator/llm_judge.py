"""Evaluator that scores a response with an LLM judge.

Asks a language model to rate the answer's faithfulness to its context
(always) and its correctness against a reference (when one is given).
Provider-agnostic: it runs against any LLMClient.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rag_pipeline.evaluator.base import Evaluator
from rag_pipeline.llm.base import LLMClient
from rag_pipeline.models import EvaluationResult, GeneratedResponse

DEFAULT_SYSTEM = (
    "You are a strict evaluator of retrieval-augmented answers. Score the "
    "answer on the requested criteria and respond with ONLY a JSON object — "
    "no prose, no code fences. Each score is a float from 0.0 (worst) to "
    "1.0 (best)."
)


class LLMJudgeEvaluator(Evaluator):
    """Scores faithfulness (always) and correctness (with a reference)."""

    def __init__(self, llm: LLMClient, system: str = DEFAULT_SYSTEM) -> None:
        """
        Args:
            llm: The language model backend used as the judge.
            system: System prompt setting the judging rubric and output format.
        """
        self._llm = llm
        self._system = system

    def evaluate(
        self,
        response: GeneratedResponse,
        reference_answer: str | None = None,
    ) -> EvaluationResult:
        result = self._llm.complete(
            _build_prompt(response, reference_answer), system=self._system
        )
        scores = _extract_json(result.text)
        try:
            metrics = {"faithfulness": float(scores["faithfulness"])}
            if reference_answer is not None:
                metrics["correctness"] = float(scores["correctness"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Judge output missing an expected score: {result.text!r}"
            ) from exc
        return EvaluationResult(
            metrics=metrics,
            details={"reasoning": scores.get("reasoning", ""), "raw": result.text},
        )


def _build_prompt(response: GeneratedResponse, reference_answer: str | None) -> str:
    context = "\n\n".join(
        f"[{i}] {scored.chunk.content}" for i, scored in enumerate(response.context, start=1)
    ) or "(no context)"
    lines = [
        f"Question:\n{response.query}",
        f"\nContext:\n{context}",
        f"\nAnswer:\n{response.answer}",
    ]
    criteria = ["- faithfulness: is every claim in the answer supported by the context?"]
    fields = ['"faithfulness": <0.0-1.0>']
    if reference_answer is not None:
        lines.append(f"\nReference answer:\n{reference_answer}")
        criteria.append("- correctness: does the answer agree with the reference answer?")
        fields.append('"correctness": <0.0-1.0>')
    fields.append('"reasoning": "<one sentence>"')
    lines.append("\nScore (0.0 = worst, 1.0 = best):\n" + "\n".join(criteria))
    lines.append("\nRespond with ONLY this JSON object:\n{" + ", ".join(fields) + "}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first {...} block if the model added stray prose.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse JSON scores from judge output: {text!r}")
