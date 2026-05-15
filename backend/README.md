# Backend API

The backend is a FastAPI service for online PDF analysis and evidence-chain generation.

## Responsibilities

1. Receive uploaded PDF files and user questions.
2. Parse PDFs into evidence nodes.
3. Enrich visual evidence when configured.
4. Build retrieval candidates and G4 reranking results.
5. Generate evidence chains and evidence cards.
6. Return structured job status for the frontend.

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
RAG_BACKEND_CANDIDATE_RETRIEVER=fusion
RAG_BACKEND_RERANK_RETRIEVER=fusion
RAG_BACKEND_ENABLE_KG=1
RAG_EMBEDDING_MODEL=doubao-embedding-vision-250615
RAG_ANSWER_MODEL=<your-answer-model-endpoint>
```

When using MinerU locally, configure `MINERU_BIN`, `MINERU_BACKEND`, `MINERU_METHOD`, and related model paths according to your local installation.

Runtime job outputs are written to:

```text
outputs/upload_jobs/
```

These outputs are local artifacts and are not committed to Git.

