"""LLMClient interface: a swappable text-completion backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Text returned by a model, plus provider metadata (model, usage, ...)."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """A large language model backend.

    Implementations wrap one backend (a local LM Studio server, a hosted
    API, ...). The
    generator depends on this interface, not on any provider, so the model
    is hot-swappable.
    """

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Complete the prompt, optionally steered by a system instruction."""
