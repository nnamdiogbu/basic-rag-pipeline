# Architecture

The pipeline is six interfaces and the data models that flow between
them. Any implementation of one interface can be swapped without
touching the others.

## Data flow

```
                     indexing time
┌────────┐   ┌────────────────┐   ┌─────────┐   ┌───────────┐
│ source │──▶│ DocumentLoader │──▶│ Indexer │──▶│ DataStore │
└────────┘   └────────────────┘   └─────────┘   └───────────┘
              Document[]            Chunk[]       stored + embedded
                                                       ▲
                     query time                        │ search(query)
┌───────┐   ┌───────────┐   ┌───────────────────┐      │
│ query │──▶│ Retriever │──────────────────────────────┘
└───────┘   └─────┬─────┘
                  │ ScoredChunk[]
            ┌─────▼─────────────┐
            │ ResponseGenerator │
            └─────┬─────────────┘
                  │ GeneratedResponse
            ┌─────▼─────┐
            │ Evaluator │
            └─────┬─────┘
                  │ EvaluationResult
```

## Components

Each lives in its own package under `src/rag_pipeline/`, with its
interface in `base.py` and implementations added alongside.

| Package | Interface | Responsibility | Implementations |
| --- | --- | --- | --- |
| `loader/` | `DocumentLoader` | Extract documents from an external source | `DoclingLoader`, `AudioLoader` |
| `indexer/` | `Indexer` | Split documents into chunks | `FixedSizeIndexer` |
| `datastore/` | `DataStore` | Store chunks; embed and similarity-search over a query | `ChromaStore` |
| `retriever/` | `Retriever` | Map a query to relevant chunks | `DataStoreRetriever` |
| `generator/` | `ResponseGenerator` | Produce an answer grounded in retrieved context | `GroundedGenerator` |
| `evaluator/` | `Evaluator` | Score a response (retrieval quality, faithfulness, correctness) | `LLMJudgeEvaluator` |

A seventh, supporting interface backs the generator:

| Package | Interface | Responsibility | Implementations |
| --- | --- | --- | --- |
| `llm/` | `LLMClient` | Complete a prompt — a swappable model backend | `LMStudioClient` |

## Data models

Defined in `models.py` — the "wires" every interface speaks in:

| Type | Carries |
| --- | --- |
| `Document` | `doc_id`, `content` (plain text), `metadata` |
| `Chunk` | `chunk_id`, `doc_id`, `content`, optional `embedding`, `metadata` |
| `ScoredChunk` | a `Chunk` plus its relevance `score` (higher is better) |
| `GeneratedResponse` | `query`, `answer`, the `context` chunks used, `metadata` |
| `EvaluationResult` | `metrics` (name → value), `details` |

## Embedding

Embedding lives **inside** the `DataStore`: `add` embeds chunk content
and `search` embeds the query, so both sides use the same model.
`ChromaStore` delegates this to Chroma's built-in embedding function
(default `all-MiniLM-L6-v2`). Because of this, `search` takes query
**text**, not a vector, and chunks carry no embedding through the
pipeline.

## Language model

`GroundedGenerator` owns the provider-agnostic work — assembling the
prompt from context and the grounding system instruction — and delegates
the actual completion to an injected `LLMClient`. The model is therefore
hot-swappable: pass `LMStudioClient` (a local LM Studio server via the
optional `[lmstudio]` extra) or any other `LLMClient` implementation
without touching the generator. The same applies to `LLMJudgeEvaluator`.
