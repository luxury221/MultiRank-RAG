# Repository Structure

This repository is organized as a complete multimodal RAG project rather than a one-off experiment folder.

## Top-Level Layout

```text
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

## Scripts Layer

`scripts/` is intentionally kept as a command-line layer. It contains stable entrypoints for reproducible research runs, plus historical experiment utilities. The main project logic is documented and grouped in [scripts/CATALOG.md](../scripts/CATALOG.md).

The important distinction is:

- **Core pipeline scripts** are part of the maintained project workflow.
- **Evaluation scripts** measure retrieval, reranking, and evidence-chain quality.
- **Dataset preparation scripts** build local benchmarks and sample data.
- **Competition/legacy scripts** are retained for traceability but are not the main project interface.

Future refactors can gradually move shared logic from `scripts/*.py` into a dedicated `multirank_rag/` package while keeping the CLI wrappers stable.
