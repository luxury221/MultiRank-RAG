# Docker and CI/CD Guide

This guide describes the deployment path for the complex-document QA demo.

## Local Docker Run

Build and start both services:

```bash
docker compose up --build -d
```

Open the application:

```text
http://127.0.0.1:8080
```

Health checks:

```bash
curl http://127.0.0.1:8080/api/health
curl http://127.0.0.1:8765/api/health
docker compose ps
docker compose logs -f backend
docker compose logs -f frontend
```

Stop services:

```bash
docker compose down
```

## Runtime Layout

```text
frontend container
  nginx serves React static files
  /api/* proxies to backend:8765

backend container
  FastAPI serves upload, job status, generated files
  /app/data is mounted read-only from ./data
  /app/outputs is mounted read-write from ./outputs
```

The local compose file uses a lightweight default mode:

```text
RAG_PDF_PARSER=native
RAG_BACKEND_CANDIDATE_RETRIEVER=lexical
RAG_BACKEND_RERANK_RETRIEVER=lexical
RAG_VISUAL_CAPTION_PROVIDER=local
RAG_ANSWER_PROVIDER=fallback
```

This avoids large local model dependencies in Docker. For a quality/cloud run, set provider variables in `.env` and adjust `docker-compose.yml` or `docker-compose.images.yml`.

## CI/CD Flow

The GitHub Actions workflow is stored at:

```text
.github/workflows/docker-ci-cd.yml
```

On pull requests it:

```text
1. Installs backend lightweight dependencies.
2. Runs Python compile checks.
3. Installs frontend dependencies.
4. Builds the React frontend.
5. Builds Docker images without pushing them.
```

On pushes to `main` it additionally pushes images to GHCR:

```text
ghcr.io/<owner>/multirank-rag-backend:latest
ghcr.io/<owner>/multirank-rag-frontend:latest
ghcr.io/<owner>/multirank-rag-backend:sha-<commit>
ghcr.io/<owner>/multirank-rag-frontend:sha-<commit>
```

## Server Deployment With Published Images

On a server, keep these files:

```text
docker-compose.images.yml
.env
data/
outputs/
```

Start with published images:

```bash
docker compose -f docker-compose.images.yml pull
docker compose -f docker-compose.images.yml up -d
```

Update to the latest image:

```bash
docker compose -f docker-compose.images.yml pull
docker compose -f docker-compose.images.yml up -d
docker compose -f docker-compose.images.yml ps
```

Roll back to a commit-tagged image by setting:

```bash
export BACKEND_IMAGE=ghcr.io/luxury221/multirank-rag-backend:sha-<commit>
export FRONTEND_IMAGE=ghcr.io/luxury221/multirank-rag-frontend:sha-<commit>
docker compose -f docker-compose.images.yml up -d
```

## Secrets

Do not commit `.env`.

Use `.env` only on the deployment host for private values:

```text
MINERU_API_KEY=...
ARK_API_KEY=...
RAG_ANSWER_MODEL=...
```

GitHub Actions uses `GITHUB_TOKEN` to push images to GHCR. External deployment credentials, if added later, should be stored in GitHub Secrets.
