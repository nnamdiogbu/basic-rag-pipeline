"""Tests for GroundedGenerator using a fake LLMClient (no provider needed)."""

from rag_pipeline.generator import GroundedGenerator
from rag_pipeline.llm import LLMClient, LLMResponse
from rag_pipeline.models import Chunk, ScoredChunk


class FakeLLM(LLMClient):
    """Echoes a canned answer and records the prompt/system it received."""

    def __init__(self, text="canned answer", metadata=None):
        self._text = text
        self._metadata = metadata or {"model": "fake-1"}
        self.calls = []

    def complete(self, prompt, system=None):
        self.calls.append({"prompt": prompt, "system": system})
        return LLMResponse(text=self._text, metadata=self._metadata)


def scored(content, doc_id="d1", score=0.9, metadata=None):
    return ScoredChunk(
        chunk=Chunk(chunk_id=f"{doc_id}:0", doc_id=doc_id, content=content, metadata=metadata or {}),
        score=score,
    )


def test_generate_maps_llm_response_into_generated_response():
    llm = FakeLLM(text="Bananas are yellow.", metadata={"model": "fake-1"})
    context = [scored("Bananas are a yellow fruit.")]

    response = GroundedGenerator(llm).generate("What colour are bananas?", context)

    assert response.query == "What colour are bananas?"
    assert response.answer == "Bananas are yellow."
    assert response.context == context
    assert response.metadata == {"model": "fake-1"}


def test_prompt_contains_context_source_and_query():
    llm = FakeLLM()
    GroundedGenerator(llm).generate("colour?", [scored("Bananas are yellow.", metadata={"source": "fruit.md"})])

    prompt = llm.calls[0]["prompt"]
    assert "Bananas are yellow." in prompt
    assert "fruit.md" in prompt
    assert "colour?" in prompt


def test_system_prompt_is_passed_through():
    llm = FakeLLM()
    GroundedGenerator(llm, system="Be terse.").generate("q", [scored("c")])
    assert llm.calls[0]["system"] == "Be terse."


def test_empty_context_still_generates():
    llm = FakeLLM(text="I don't know.")
    response = GroundedGenerator(llm).generate("anything?", [])

    assert response.answer == "I don't know."
    assert "(no context retrieved)" in llm.calls[0]["prompt"]
