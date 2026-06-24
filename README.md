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

## Adding your data

The `data/` directory is empty by default — populate it with your own
documents. Drop any supported file (PDF, DOCX, PPTX, Markdown, HTML, and
audio such as WAV/MP3 with the `[audio]` extra) into `data/` and it gets
loaded and indexed; the folder is scanned recursively.

```sh
cp ~/Documents/my-handbook.pdf data/
```

> Note: `data/` is git-ignored, so your documents stay local and are not
> committed.

## Evaluation

`evaluate/evaluate_quickstart.py` runs the full pipeline end to end over
everything in `data/` and scores the generated answers for **faithfulness**
(to the retrieved context) and **correctness** (against a reference answer)
using an LLM judge.

The evaluation questions and reference answers live in
[`evaluate/samples.json`](evaluate/samples.json), separate from the code so
you can edit them without touching Python. It ships with a single empty
template entry — fill it in with questions about *your* documents:

```json
[
  {
    "question": "What is the maximum number of vacation days per year?",
    "reference_answer": "20 days."
  },
  {
    "question": "Who approves expense reports over $500?",
    "reference_answer": "The department manager."
  }
]
```

Each entry needs a `question` and a `reference_answer`. Add as many entries
as you like — the evaluator runs every one.

Prerequisites: install an LLM backend extra and start the local server
(default LM Studio at `http://localhost:1234/v1`):

```sh
pip install -e ".[lmstudio,chroma,docs]"
```

Then run:

```sh
python3 evaluate/evaluate_quickstart.py            # evaluates over data/
python3 evaluate/evaluate_quickstart.py path/to/docs   # or a custom folder/file
```
