"""Generation: producing an answer grounded in retrieved context."""

from rag_pipeline.generator.base import ResponseGenerator
from rag_pipeline.generator.grounded import GroundedGenerator

__all__ = ["GroundedGenerator", "ResponseGenerator"]
