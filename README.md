# MultiRank-RAG

MultiRank-RAG is a multimodal RAG system for complex PDF documents, product manuals, and image-text knowledge bases. It focuses on one core goal: answering questions with a traceable evidence chain instead of returning an isolated response.

The system parses PDF layout, converts text/tables/figures/captions into unified evidence nodes, builds document and GraphRAG relations, retrieves multimodal evidence, reranks it with query-aware structural signals, and finally generates grounded answers and visual evidence cards.

## Highlights

- **Complex PDF parsing**: supports MinerU and native parsing, preserving page, section, bbox, table, figure, caption, and layout metadata.
- **Unified evidence schema**: models text, title, table, figure, caption, equation, and page nodes in one retrieval graph.
- **Multimodal evidence enrichment**: adds image crops, OCR, visual captions, key objects, visual summaries, and QA-oriented evidence fields.
- **GraphRAG layer**: builds document structure edges, entity links, semantic relations, and community summaries for graph-aware retrieval.
- **Hybrid retrieval**: combines BM25, lexical matching, embeddings, visual evidence, and GraphRAG/KG signals.
- **MultiRank reranking**: compares V0-V4 / G0-G4 variants with semantic similarity, PPR, bridge scores, reference matching, visual grounding, adaptive routing, and evidence-chain signals.
- **Evidence-chain answer generation**: generates final answers from V4 chains only, with evidence citations and `<PIC:node_id>` markers for visual evidence.
- **Full-stack demo**: includes FastAPI backend and React frontend for uploading PDFs, asking questions, and inspecting evidence cards.

## System Architecture

```text
PDF / Manual / Knowledge Base
        |
        v
PDF Parser
  - MinerU / native parser
  - page, section, bbox, table, figure, caption
        |
        v
Evidence Node Builder
  - outputs/parsed/nodes.jsonl
  - outputs/parsed/edges.jsonl
        |
        v
Visual Evidence Enrichment
  - image crop
  - OCR / caption / key objects / QA evidence
        |
        v
GraphRAG Builder
  - document structure graph
  - semantic entity graph
  - relation and community summaries
        |
        v
Hybrid Retrieval
  - BM25 / lexical / embedding / visual / GraphRAG
        |
        v
MultiRank Reranking
  - G0 raw retrieval
  - G1 semantic rerank
  - G2 semantic + PPR
  - G3 bridge + reference
  - G4 adaptive multimodal evidence chain
        |
        v
Evidence Chain / Grounded Answer / Evidence Card / API
```

Detailed design notes are available in:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/GRAPHRAG.md](docs/GRAPHRAG.md)
- [docs/MODEL_GATEWAY.md](docs/MODEL_GATEWAY.md)
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- [docs/EVALUATION.md](docs/EVALUATION.md)

## Repository Layout

```text
backend/      FastAPI service for PDF upload, analysis jobs, and evidence-chain APIs
web/          React + Vite frontend for document selection, Q&A, and evidence visualization
scripts/      Offline pipeline scripts for parsing, GraphRAG, retrieval, reranking, generation, and diagnostics
configs/      Environment examples and chunking templates
data/         Local questions, sample data, and PDF input folders
docs/         Architecture notes, evaluation design, and project documentation
outputs/      Generated artifacts, ignored by Git except placeholders
external/     External cloned dependencies, ignored by Git
```

The public datasets and benchmark files in this repository are only used as engineering references. The project itself is designed as a general complex-document multimodal RAG system, not as a dataset-specific submission script.

## Quick Start

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the offline pipeline on local questions:

```bash
python scripts/06_run_pipeline.py \
  --questions data/questions.csv \
  --candidate-k 50 \
  --rerank-k 10
```

Run a lightweight sample check:

```bash
python scripts/06_run_pipeline.py \
  --sample \
  --retriever bm25 \
  --candidate-k 10 \
  --rerank-k 3 \
  --rerank-methods G4
```

Run the V4 evidence-chain experiment with answer generation:

```bash
python scripts/40_run_main_experiment.py \
  --dataset-name sample \
  --variants V4 \
  --build-chains \
  --generate-answers \
  --answer-provider none
```

Use `--answer-provider ark`, `--answer-provider xinference`, or `--answer-provider openai_compatible` when model services are configured.

## Web Demo

Start the backend:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

Start the frontend:

```bash
cd web
npm install
npm run dev
```

The frontend supports uploading PDFs, selecting built-in documents, asking custom or preset questions, and viewing evidence cards with ranked supporting evidence.

## Important Outputs

```text
outputs/parsed/nodes.jsonl                  Unified evidence nodes
outputs/parsed/edges.jsonl                  Document structure edges
outputs/graphrag/entities.jsonl             GraphRAG entities
outputs/graphrag/relations.jsonl            GraphRAG relations
outputs/graphrag/communities.jsonl          Community summaries
outputs/rankings/candidates.csv             Retrieved candidate evidence
outputs/rankings/reranked.csv               G0-G4 reranking results
outputs/evidence_chains/chains.jsonl        Question-level evidence chains
outputs/evidence_chains/answers.csv         Chain-grounded final answers
outputs/evidence_cards/                     Rendered evidence card assets
```

## Configuration

Use `.env.example` and `configs/doubao_optimized.env.example` as references. Put real keys in a local `.env` file, which is ignored by Git.

Common configuration:

```bash
RAG_PDF_PARSER=mineru
MINERU_API_MODE=cloud
MINERU_API_URL=https://mineru.net/api/v4
MINERU_API_KEY=<your-mineru-api-key>

RAG_MODEL_PROVIDER=ark
RAG_EMBEDDING_PROVIDER=ark
RAG_EMBEDDING_MODEL=doubao-embedding-vision-250615
RAG_ANSWER_PROVIDER=ark
RAG_ANSWER_MODEL=<your-doubao-endpoint-id>

RAG_KG_DIR=outputs/graphrag
RAG_BACKEND_CANDIDATE_RETRIEVER=fusion
RAG_BACKEND_RERANK_RETRIEVER=fusion
```

Supported model gateway modes:

```text
ark                 Direct Ark / Doubao / DashScope-compatible provider
xinference          Local or remote Xinference OpenAI-compatible gateway
openai_compatible   Any local OpenAI-compatible service
```

Visual captioning, embedding, answer generation, and reranking can be configured independently through `RAG_VISUAL_CAPTION_PROVIDER`, `RAG_EMBEDDING_PROVIDER`, `RAG_ANSWER_PROVIDER`, and `RAG_RERANK_PROVIDER`.

## Evaluation

The project uses two complementary evaluation layers:

- **Retrieval and reranking metrics**: Recall@k, MRR, nDCG, evidence hit, modality hit.
- **Evidence-chain metrics**: chain presence, gold-node coverage, page hit, modality coverage, visual grounding, cross-modal support, and relation support.

This separation is intentional: MultiRank-RAG aims not only to retrieve the correct chunk, but also to organize a coherent, inspectable evidence path for the final answer.

## Documentation

- [scripts/README.md](scripts/README.md): script entry points and offline pipeline usage.
- [backend/README.md](backend/README.md): backend API and job flow.
- [web/README.md](web/README.md): frontend setup and UI behavior.
- [docs/REPOSITORY_GUIDE.md](docs/REPOSITORY_GUIDE.md): repository maintenance notes.

## Current Status

The project currently provides a complete research prototype: PDF upload, parsing, multimodal node construction, GraphRAG indexing, hybrid retrieval, MultiRank reranking, evidence-chain generation, grounded answer generation, and frontend evidence visualization. Future work focuses on stronger visual grounding, more robust PDF parsing, richer GraphRAG relation extraction, and answer-level evaluation with LLM judges.
