"""Language models: swappable text-completion backends for generation."""

from rag_pipeline.llm.base import LLMClient, LLMResponse
from rag_pipeline.llm.lmstudio import LMStudioClient

__all__ = ["LLMClient", "LLMResponse", "LMStudioClient"]
