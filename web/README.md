# Frontend

The frontend is a React + Vite application for the multimodal RAG evidence workflow.

## Features

- Select built-in documents or upload a PDF.
- Ask a custom question or choose a prepared question.
- View the generated answer.
- Inspect evidence cards and ranked supporting evidence.
- Open visual evidence files served by the backend.

## Run in Development

```bash
cd web
npm install
npm run dev
```

Default development URL:

```text
http://127.0.0.1:5173
```

The frontend expects the backend at:

```text
http://127.0.0.1:8765
```

## Build

```bash
cd web
npm run build
```

The build output is written to `web/dist/`, which is ignored by Git.

## Docker Deployment

From the repository root:

```bash
docker compose up --build -d
```

Open:

```text
http://127.0.0.1:8080
```

In Docker, Nginx serves the frontend and proxies `/api/*` to the FastAPI backend. See `../docs/DOCKER_CICD.md` for the full Docker and CI/CD workflow.
