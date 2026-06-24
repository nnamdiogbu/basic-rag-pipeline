"""Quickstart: evaluate RAG answers over a folder of documents with an LLM judge.

Wires the whole pipeline end to end over every supported file in ``data/``:

    data/ -> DoclingLoader -> FixedSizeIndexer -> ChromaStore
          -> DataStoreRetriever -> GroundedGenerator -> LLMJudgeEvaluator

Drop any supported file (PDF/DOCX/PPTX/MD/HTML/...) into ``data/`` and it
gets loaded and indexed — the folder is scanned recursively.

For each sample question it retrieves context, generates a grounded
answer with the local LM Studio model, and scores the answer for
faithfulness (to the retrieved context) and correctness (against a
reference answer).

Prerequisites:
    pip install -e ".[lmstudio,chroma,docs]"
    Start LM Studio's local server with a model loaded
    (default: qwen/qwen2.5-coder-14b at http://localhost:1234/v1).

Run:
    python evaluate/evaluate_quickstart.py [path/to/docs-folder-or-file]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a plain script without an editable install: put the
# project's src/ directory on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_pipeline.datastore import ChromaStore
from rag_pipeline.evaluator import LLMJudgeEvaluator
from rag_pipeline.generator import GroundedGenerator
from rag_pipeline.indexer import FixedSizeIndexer
from rag_pipeline.llm import LMStudioClient
from rag_pipeline.loader import AudioLoader, DoclingLoader
from rag_pipeline.retriever import DataStoreRetriever

# Sample (question, reference answer) pairs live in ``samples.json`` next to
# this script. Each entry has a ``question`` and a ``reference_answer``; the
# reference answers drive the judge's correctness metric.
SAMPLES_PATH = Path(__file__).resolve().parent / "samples.json"


def load_samples(path: Path = SAMPLES_PATH) -> list[tuple[str, str]]:
    """Load (question, reference_answer) pairs from a JSON file.

    Entries with a blank question or reference answer are skipped, so the
    empty template that ships with the repo doesn't produce a run.
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        question = item["question"].strip()
        reference = item["reference_answer"].strip()
        if question and reference:
            samples.append((question, reference))
    return samples


def main(source: str = "data") -> None:
    print(f"Loading documents from {source!r} ...")
    documents = DoclingLoader([source]).load()
    # Transcribe any audio in the same folder; skip if [audio] isn't installed.
    try:
        documents += AudioLoader([source]).load()
    except ImportError:
        print("  (audio support not installed — pip install 'rag-pipeline[audio]')")
    if not documents:
        print(f"  No supported documents found in {source!r}. Add a file and retry.")
        return
    total_chars = sum(len(d.content) for d in documents)
    print(f"  {len(documents)} document(s), {total_chars:,} characters")

    print("Indexing and storing ...")
    chunks = FixedSizeIndexer(chunk_size=1000, chunk_overlap=200).index(documents)
    store = ChromaStore(collection_name="corpus")
    store.add(chunks)
    print(f"  {store.count()} chunks stored")

    llm = LMStudioClient()
    retriever = DataStoreRetriever(store)
    generator = GroundedGenerator(llm)
    evaluator = LLMJudgeEvaluator(llm)

    samples = load_samples()
    if not samples:
        print(
            f"\nNo evaluation samples found. Add question/reference_answer "
            f"pairs to {SAMPLES_PATH} and retry."
        )
        return

    for question, reference in samples:
        print("\n" + "=" * 70)
        print(f"Q: {question}")

        context = retriever.retrieve(question, top_k=4)
        try:
            response = generator.generate(question, context)
            result = evaluator.evaluate(response, reference_answer=reference)
        except Exception as exc:  # noqa: BLE001 - surface a friendly hint
            print(
                "\nLLM call failed. Is LM Studio's local server running with a "
                "model loaded?\n"
                "  Default endpoint: http://localhost:1234/v1\n"
                f"  Error: {exc}"
            )
            return

        print(f"\nAnswer:    {response.answer}")
        print(f"Reference: {reference}")
        sources = ", ".join(
            f"{c.chunk.metadata.get('chunk_index', '?')}@{c.score:.2f}" for c in context
        )
        print(f"Context:   {len(context)} chunks (chunk_index@score: {sources})")
        metrics = ", ".join(f"{k}={v:.2f}" for k, v in result.metrics.items())
        print(f"Scores:    {metrics}")
        print(f"Reasoning: {result.details.get('reasoning', '')}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
