# Backend API

The backend is a FastAPI service for online PDF analysis and evidence-chain generation.

## Structure

```text
backend/app.py                 FastAPI app factory and router registration
backend/config.py              Paths, CORS, environment helpers, provider settings
backend/routers/               HTTP endpoints only
backend/schemas/               Pydantic request/response schemas
backend/jobs/                  Upload job directories, status files, logs
backend/services/              PDF/RAG pipeline, retrieval, visual caption, frontend serialization
backend/utils/                 Small reusable helpers
multirank_rag/                 Stable package imports for core RAG capabilities
```

The route layer is intentionally thin. Heavy work such as parsing, visual evidence enrichment, GraphRAG indexing, retrieval, reranking, evidence-chain generation, and card generation is orchestrated in `backend/services/` and imported through the stable `multirank_rag/` package boundary.

## Responsibilities

1. Receive uploaded PDF files and user questions.
2. Parse PDFs into evidence nodes.
3. Enrich visual evidence when configured.
4. Build a per-upload GraphRAG index from nodes and document edges.
5. Build retrieval candidates and G4 reranking results.
6. Generate evidence chains and evidence cards.
7. Return structured job status for the frontend.

## Run

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

## API

```text
GET  /api/health
POST /api/analyze
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/files/{path}
```

The main upload flow is:

```text
POST /api/analyze
  -> routers/analyze.py validates form data and creates a job
  -> jobs/store.py persists status and logs
  -> services/pipeline.py runs parse -> visual -> graph -> GraphRAG -> retrieve -> rerank -> chain -> card
  -> routers/jobs.py returns status, result JSON, and generated files
```

## Environment

Common environment variables:

```text
RAG_PDF_PARSER=mineru
MINERU_API_MODE=cloud
MINERU_API_URL=https://mineru.net/api/v4
MINERU_API_KEY=<your-mineru-api-key>
RAG_BACKEND_PIPELINE_MODE=quality
RAG_BACKEND_PIPELINE_VARIANT=V5-online-quality
RAG_BACKEND_CANDIDATE_RETRIEVER=multiroute
RAG_BACKEND_RERANK_RETRIEVER=multiroute
RAG_BACKEND_CONTEXT_EXPANSION=true
RAG_BACKEND_ADAPTIVE_RERANK_BOOST=true
RAG_BACKEND_GRAPH_CONTEXT_BOOST=true
RAG_BACKEND_EVIDENCE_GUARD=true
RAG_BACKEND_ENHANCED_CONTEXT_EDGES=true
RAG_BACKEND_ENABLE_KG=1
RAG_KG_DIR=outputs/graphrag
RAG_MODEL_PROVIDER=ark
RAG_EMBEDDING_PROVIDER=ark
RAG_EMBEDDING_MODEL=doubao-embedding-vision-250615
RAG_ANSWER_PROVIDER=ark
RAG_ANSWER_MODEL=<your-answer-model-endpoint>
```

When `MINERU_API_KEY` is set, uploaded PDFs are parsed through MinerU OpenAPI. Without it, the backend falls back to local MinerU CLI; configure `MINERU_BIN`, `MINERU_BACKEND`, `MINERU_METHOD`, and related model paths according to your local installation.

The backend can route model calls through direct cloud APIs, Xinference, or a local OpenAI-compatible server:

```text
RAG_MODEL_PROVIDER=ark
RAG_MODEL_PROVIDER=xinference
RAG_MODEL_PROVIDER=openai_compatible
```

Use `RAG_EMBEDDING_PROVIDER`, `RAG_ANSWER_PROVIDER`, `RAG_VISUAL_CAPTION_PROVIDER`, and `RAG_RERANK_PROVIDER` to override individual modules. See `docs/MODEL_GATEWAY.md` for complete examples.

For local development, put private keys in the repository root `.env` file:

```text
MINERU_API_MODE=cloud
MINERU_API_URL=https://mineru.net/api/v4
MINERU_API_KEY=your-mineru-api-key
```

The `.env` file is ignored by Git.

Runtime job outputs are written to:

```text
outputs/upload_jobs/
```

These outputs are local artifacts and are not committed to Git.
