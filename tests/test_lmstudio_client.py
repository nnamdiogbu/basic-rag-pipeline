"""Tests for LMStudioClient using a fake OpenAI-compatible client (no server)."""

from types import SimpleNamespace

from rag_pipeline.llm import LMStudioClient


def fake_completion(content, model="qwen2.5", finish_reason="stop", usage=True):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7) if usage else None,
    )


class FakeCompletions:
    def __init__(self, completion):
        self._completion = completion
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._completion


class FakeClient:
    def __init__(self, completion):
        self.chat = SimpleNamespace(completions=FakeCompletions(completion))


def test_complete_returns_text_and_metadata():
    client = FakeClient(fake_completion("Bananas are yellow."))
    response = LMStudioClient(client=client).complete("question?", system="be grounded")

    assert response.text == "Bananas are yellow."
    assert response.metadata["model"] == "qwen2.5"
    assert response.metadata["finish_reason"] == "stop"
    assert response.metadata["usage"] == {"input_tokens": 42, "output_tokens": 7}


def test_messages_model_and_max_tokens_are_forwarded():
    client = FakeClient(fake_completion("ok"))
    LMStudioClient(model="qwen2.5-coder", max_tokens=512, client=client).complete("hi", system="terse")

    call = client.chat.completions.calls[0]
    assert call["model"] == "qwen2.5-coder"
    assert call["max_tokens"] == 512
    assert call["messages"] == [
        {"role": "system", "content": "terse"},
        {"role": "user", "content": "hi"},
    ]


def test_system_message_omitted_when_none():
    client = FakeClient(fake_completion("ok"))
    LMStudioClient(client=client).complete("hi")
    assert client.chat.completions.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_missing_usage_is_tolerated():
    client = FakeClient(fake_completion("ok", usage=False))
    response = LMStudioClient(client=client).complete("hi")
    assert response.metadata["usage"] == {}
