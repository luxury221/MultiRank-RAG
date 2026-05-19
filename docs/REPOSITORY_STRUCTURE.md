# Repository Structure

This repository is organized as a complete multimodal RAG project rather than a one-off experiment folder.

## Top-Level Layout

```text
multirank_rag/  Core Python package with stable import paths
backend/      FastAPI application package
web/          React + Vite frontend
scripts/      CLI-oriented pipeline, evaluation, and research utilities
configs/      Environment and model configuration examples
data/         Small sample inputs and local question files
docs/         Architecture, evaluation, and engineering documentation
outputs/      Generated runtime artifacts, ignored by Git
external/     External cloned dependencies, ignored by Git
```

## Backend Package

The backend is split by responsibility:

```text
backend/app.py                 FastAPI app factory and router registration
backend/config.py              Paths, CORS, env helpers, provider settings
backend/routers/               HTTP endpoints only
backend/schemas/               Pydantic request/response schemas
backend/jobs/                  Upload job state and status persistence
backend/services/              RAG orchestration, retrieval, visual caption, frontend serialization
backend/utils/                 Small file and formatting helpers
```

This keeps the API layer thin. PDF parsing, visual evidence, GraphRAG, retrieval, reranking, evidence-chain generation, and frontend result normalization live in services rather than in the route functions.

## Core Package

`multirank_rag/` is the stable package boundary for application code:

```text
multirank_rag/common.py        Shared path, CSV/JSONL, and text helpers
multirank_rag/parsing/         PDF parsing and structured chunking facade
multirank_rag/graph/           Structure graph and GraphRAG facade
multirank_rag/vision/          Visual evidence enrichment facade
multirank_rag/retrieval/       Embedding index, hybrid retrieval, KG index
multirank_rag/rerank/          MultiRank reranking and answer helpers
multirank_rag/evidence/        Evidence chain and evidence card APIs
multirank_rag/evaluation/      Retrieval and evidence-chain metrics
multirank_rag/models/          Model gateway wrappers
```

The first extraction phase keeps the original CLI scripts stable and exposes their maintained functionality through package facades. Future work can continue moving implementation details from `scripts/*.py` into these subpackages without changing backend imports.

## Scripts Layer

`scripts/` is intentionally kept as a command-line layer. It contains stable entrypoints for reproducible research runs, plus historical experiment utilities. The main project logic is documented and grouped in [scripts/CATALOG.md](../scripts/CATALOG.md).

The important distinction is:

- **Core pipeline scripts** are part of the maintained project workflow.
- **Evaluation scripts** measure retrieval, reranking, and evidence-chain quality.
- **Dataset preparation scripts** build local benchmarks and sample data.
- **Competition/legacy scripts** are retained for traceability but are not the main project interface.

Future refactors can gradually move more implementation details from `scripts/*.py` into `multirank_rag/` while keeping the CLI wrappers stable.
