"""Tests for LLMJudgeEvaluator using a fake LLMClient (no provider needed)."""

import pytest

from rag_pipeline.evaluator import LLMJudgeEvaluator
from rag_pipeline.llm import LLMClient, LLMResponse
from rag_pipeline.models import Chunk, GeneratedResponse, ScoredChunk


class FakeLLM(LLMClient):
    """Returns canned judge text and records the prompt it received."""

    def __init__(self, text):
        self._text = text
        self.calls = []

    def complete(self, prompt, system=None):
        self.calls.append({"prompt": prompt, "system": system})
        return LLMResponse(text=self._text)


def make_response(answer="Bananas are yellow.", query="colour?", context_text="Bananas are yellow."):
    context = [
        ScoredChunk(chunk=Chunk(chunk_id="d1:0", doc_id="d1", content=context_text), score=0.9)
    ]
    return GeneratedResponse(query=query, answer=answer, context=context)


def test_faithfulness_only_without_reference():
    llm = FakeLLM('{"faithfulness": 0.9, "reasoning": "supported by context"}')
    result = LLMJudgeEvaluator(llm).evaluate(make_response())

    assert result.metrics == {"faithfulness": 0.9}
    assert result.details["reasoning"] == "supported by context"


def test_correctness_added_with_reference():
    llm = FakeLLM('{"faithfulness": 0.9, "correctness": 1.0, "reasoning": "matches"}')
    result = LLMJudgeEvaluator(llm).evaluate(make_response(), reference_answer="Bananas are yellow.")

    assert result.metrics == {"faithfulness": 0.9, "correctness": 1.0}
    # the reference is included in the judge prompt only when supplied
    assert "Reference answer:" in llm.calls[0]["prompt"]


def test_prompt_contains_question_context_and_answer():
    llm = FakeLLM('{"faithfulness": 1.0}')
    LLMJudgeEvaluator(llm).evaluate(make_response(answer="A", query="Q", context_text="CTX"))

    prompt = llm.calls[0]["prompt"]
    assert "Q" in prompt and "CTX" in prompt and "A" in prompt
    assert "Reference answer:" not in prompt


def test_json_embedded_in_prose_is_extracted():
    llm = FakeLLM('Here is my verdict: {"faithfulness": 0.4} — hope that helps!')
    result = LLMJudgeEvaluator(llm).evaluate(make_response())
    assert result.metrics["faithfulness"] == 0.4


def test_unparseable_output_raises():
    with pytest.raises(ValueError, match="Could not parse JSON"):
        LLMJudgeEvaluator(FakeLLM("no json here")).evaluate(make_response())


def test_missing_score_raises():
    llm = FakeLLM('{"reasoning": "forgot the number"}')
    with pytest.raises(ValueError, match="missing an expected score"):
        LLMJudgeEvaluator(llm).evaluate(make_response())
