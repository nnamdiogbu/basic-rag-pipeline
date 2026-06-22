# basic-rag-pipeline

An educational RAG (Retrieval-Augmented Generation) pipeline, built piece
by piece. The structure comes first: six small interfaces define the
pipeline, and real implementations get added one at a time.

## Architecture

Six small interfaces define the pipeline, and any implementation of one
can be swapped without touching the others. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the data-flow diagram, the
component table, and the data models that flow between the pieces.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Optional extras per source type (kept out of the core install because
they pull in heavy dependencies):

```sh
pip install -e ".[docs]"   # docling, for PDF/DOCX/PPTX/Markdown/... loading
```

The tests in `tests/test_interfaces.py` wire naive fake implementations
through all five interfaces end-to-end — a working miniature of the
pipeline and a template for real implementations to come.
# basic-rag-pipeline
# basic-rag-pipeline
