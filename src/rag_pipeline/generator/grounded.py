"""ResponseGenerator that answers from context using any LLMClient."""

from __future__ import annotations

from collections.abc import Sequence

from rag_pipeline.generator.base import ResponseGenerator
from rag_pipeline.llm.base import LLMClient
from rag_pipeline.models import GeneratedResponse, ScoredChunk

DEFAULT_SYSTEM = (
    "You answer questions using only the provided context. If the context "
    "does not contain the answer, say you don't know rather than guessing. "
    "Be concise and ground every claim in the context."
)


class GroundedGenerator(ResponseGenerator):
    """Builds a grounded prompt from context and delegates to an LLMClient.

    The LLM is injected, so the provider is hot-swappable without changing
    the prompt-assembly or grounding logic.
    """

    def __init__(self, llm: LLMClient, system: str = DEFAULT_SYSTEM) -> None:
        """
        Args:
            llm: The language model backend to generate with.
            system: System prompt steering the model to stay grounded.
        """
        self._llm = llm
        self._system = system

    def generate(self, query: str, context: Sequence[ScoredChunk]) -> GeneratedResponse:
        response = self._llm.complete(_build_prompt(query, context), system=self._system)
        return GeneratedResponse(
            query=query,
            answer=response.text,
            context=list(context),
            metadata=response.metadata,
        )


def _build_prompt(query: str, context: Sequence[ScoredChunk]) -> str:
    if not context:
        return f"Context:\n(no context retrieved)\n\nQuestion: {query}"
    blocks = []
    for i, scored in enumerate(context, start=1):
        source = scored.chunk.metadata.get("source", scored.chunk.doc_id)
        blocks.append(f"[{i}] (source: {source})\n{scored.chunk.content}")
    return "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}"
