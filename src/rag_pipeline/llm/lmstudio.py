"""LLMClient backed by a local LM Studio server.

LM Studio exposes an OpenAI-compatible API, so this talks to it through
the ``openai`` SDK pointed at the local endpoint. Optional dependency:
install with ``pip install "rag-pipeline[lmstudio]"`` and start LM Studio's
local server with a model (e.g. qwen2.5) loaded.
"""

from __future__ import annotations

from typing import Any

from rag_pipeline.llm.base import LLMClient, LLMResponse

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "qwen/qwen2.5-coder-14b"
DEFAULT_MAX_TOKENS = 2048


class LMStudioClient(LLMClient):
    """Calls a model served by LM Studio's local OpenAI-compatible API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str = "lm-studio",
        client: Any | None = None,
    ) -> None:
        """
        Args:
            model: Model identifier as loaded in LM Studio.
            base_url: LM Studio server URL (its OpenAI-compatible endpoint).
            max_tokens: Maximum tokens to generate.
            api_key: Placeholder key; LM Studio ignores it but the SDK
                requires one.
            client: An ``openai.OpenAI`` client; created from base_url and
                api_key if not supplied.
        """
        self._model = model
        self._max_tokens = max_tokens
        self._client = client if client is not None else _make_client(base_url, api_key)

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        completion = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )
        choice = completion.choices[0]
        usage = {}
        if completion.usage is not None:
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            metadata={
                "model": completion.model,
                "finish_reason": choice.finish_reason,
                "usage": usage,
            },
        )


def _make_client(base_url: str, api_key: str) -> Any:
    # Imported lazily so the core package works without the openai SDK.
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "LMStudioClient requires the openai SDK. "
            "Install it with: pip install 'rag-pipeline[lmstudio]'"
        ) from exc
    return OpenAI(base_url=base_url, api_key=api_key)
