# Backend API

The backend is a FastAPI service for online PDF analysis and evidence-chain generation.

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

## Environment

Common environment variables:

```text
RAG_PDF_PARSER=mineru
MINERU_API_MODE=cloud
MINERU_API_URL=https://mineru.net/api/v4
MINERU_API_KEY=<your-mineru-api-key>
RAG_BACKEND_CANDIDATE_RETRIEVER=fusion
RAG_BACKEND_RERANK_RETRIEVER=fusion
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
