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

