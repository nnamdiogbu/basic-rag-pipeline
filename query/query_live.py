"""Interactive RAG query tool: ask questions about your documents live.

Loads and indexes everything in a folder (PDFs/DOCX/... via docling, audio
via Whisper), then drops into a prompt where you type questions. Each
question is answered by the local LM Studio model, grounded in the
retrieved context.

By default the index is in-memory and discarded on exit. Pass --persist
DIR to keep it on disk and skip re-indexing on the next run.

Prerequisites:
    pip install -e ".[lmstudio,chroma,docs]"   # add ,audio for audio files
    Start LM Studio's local server with a model loaded
    (default: qwen/qwen2.5-coder-14b at http://localhost:1234/v1).

Run:
    python query/query_live.py                  # query ./data, in-memory
    python query/query_live.py -s path/to/docs
    python query/query_live.py --persist ./chroma_db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_pipeline.datastore import ChromaStore
from rag_pipeline.generator import GroundedGenerator
from rag_pipeline.indexer import FixedSizeIndexer
from rag_pipeline.llm import LMStudioClient
from rag_pipeline.loader import AudioLoader, DoclingLoader
from rag_pipeline.retriever import DataStoreRetriever


def load_documents(source: str):
    documents = DoclingLoader([source]).load()
    try:
        documents += AudioLoader([source]).load()
    except ImportError:
        print("  (audio support not installed — pip install 'rag-pipeline[audio]')")
    return documents


def build_store(source: str, persist_directory: str | None) -> ChromaStore:
    store = ChromaStore(collection_name="corpus", persist_directory=persist_directory)
    if store.count() > 0:
        print(f"Reusing existing index ({store.count()} chunks).")
        return store
    print(f"Loading documents from {source!r} ...")
    documents = load_documents(source)
    if not documents:
        print(f"  No supported documents found in {source!r}.")
        return store
    print(f"  {len(documents)} document(s); chunking and indexing ...")
    chunks = FixedSizeIndexer(chunk_size=1000, chunk_overlap=200).index(documents)
    store.add(chunks)
    print(f"  {store.count()} chunks indexed.")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about your documents.")
    parser.add_argument(
        "-s", "--source", default="data", help="folder or file to index (default: data)"
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=4, help="chunks to retrieve per question (default: 4)"
    )
    parser.add_argument(
        "--persist",
        metavar="DIR",
        default=None,
        help="persist the index to DIR instead of in-memory (skips re-indexing next run)",
    )
    args = parser.parse_args()

    store = build_store(args.source, args.persist)
    if store.count() == 0:
        return

    retriever = DataStoreRetriever(store)
    generator = GroundedGenerator(LMStudioClient())

    print("\nReady — ask a question ('exit' or Ctrl-D to quit).")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        context = retriever.retrieve(query, top_k=args.top_k)
        try:
            response = generator.generate(query, context)
        except Exception as exc:  # noqa: BLE001 - surface a friendly hint
            print(
                "\nLLM call failed. Is LM Studio's local server running with a "
                "model loaded?\n"
                "  Default endpoint: http://localhost:1234/v1\n"
                f"  Error: {exc}"
            )
            continue

        print(f"\n{response.answer}")
        sources = ", ".join(
            f"{Path(c.chunk.doc_id).name}#{c.chunk.metadata.get('chunk_index', '?')}@{c.score:.2f}"
            for c in context
        )
        print(f"\n[sources: {sources}]")


if __name__ == "__main__":
    main()
