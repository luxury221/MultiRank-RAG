# Scripts

`scripts/` is the command-line layer for MultiRank-RAG. It contains maintained pipeline entrypoints, evaluation tools, and dataset preparation helpers.

For a categorized file-by-file index, see [CATALOG.md](CATALOG.md).

The root-level script paths are kept stable so existing experiment commands still work. Shared production-facing imports live in `multirank_rag/`, API behavior lives in `backend/`, while long-running research workflows remain here as explicit CLI tools.

## Main Entry

```bash
python scripts/06_run_pipeline.py --questions data/questions.csv
```

Useful options:

```bash
--sample                    Run with data/sample
--skip-parse                Reuse existing parsed nodes
--skip-visual               Skip image/visual caption enrichment
--skip-graphrag             Skip GraphRAG construction
--candidate-retriever       fusion | hybrid | embedding | lexical | bm25 | kg
--rerank-retriever          fusion | hybrid | embedding | lexical | bm25 | kg
--rerank-methods            G0,G1,G2,G3,G4
--kg-dir                    GraphRAG/KG directory, default outputs/graphrag
```

## Core Pipeline

```text
01_parse_pdf.py                 Parse PDFs into evidence nodes
02_build_graph.py               Build document structure edges
10_build_visual_evidence.py     Add image crops, OCR/caption/QA visual evidence
23_build_graphrag.py            Build GraphRAG entities, relations and communities
03_retrieve_candidates.py       Retrieve candidate evidence nodes
04_rerank.py                    Run G0-G4 MultiRank reranking
09_build_evidence_chains.py     Build question-level evidence chains
34_generate_chain_answers.py    Generate final answers grounded by evidence chains
11_build_evidence_cards.py      Render evidence card assets
12_check_evidence_cards.py      Check card completeness
42_enhance_multimodal_nodes.py  Enhance table and visual nodes
50_export_experiment_summary.py Export experiment summaries to Excel/CSV
52_self_correct_evidence.py     Verify, merge, or replace evidence chains
```

## Script Groups

```text
Core pipeline       01, 02, 03, 04, 06, 09, 10, 11, 23, 34, 40, 42, 52
Shared libraries    pipeline_common.py, embedding_index.py, rerank_lib.py, ark_clients.py, query_expansion.py
Evaluation          05, 08, 12_evaluate, 14, 50
Data preparation    07, 41
```

The recommended user-facing path is the main pipeline plus backend/frontend.

## Main Experiment

Run the V0-V4 experiment and let V4 generate chain-grounded answers:

```bash
python scripts/40_run_main_experiment.py \
  --dataset-name sample \
  --variants V0,V1,V2,V3,V4 \
  --build-chains \
  --generate-answers \
  --answer-provider ark
```

Use `--answer-provider none` for deterministic offline output without calling an API. The generated answer files are written under `V4/evidence_chains/answers.csv` and `answers.jsonl`.

## GraphRAG

Run GraphRAG alone:

```bash
python scripts/23_build_graphrag.py \
  --nodes outputs/parsed/nodes.jsonl \
  --edges outputs/parsed/edges.jsonl \
  --output-dir outputs/graphrag
```

Important outputs:

```text
outputs/graphrag/entities.jsonl
outputs/graphrag/relations.jsonl
outputs/graphrag/communities.jsonl
outputs/graphrag/entity_links.jsonl
```

`03_retrieve_candidates.py` and `04_rerank.py` can consume this directory through:

```bash
--kg-dir outputs/graphrag
```

## Model Gateway

Offline scripts read the same `.env` provider settings as the backend:

```text
RAG_MODEL_PROVIDER=ark | xinference | openai_compatible
RAG_EMBEDDING_PROVIDER=local | ark | xinference | openai_compatible
RAG_VISUAL_CAPTION_PROVIDER=local | qwen | doubao | xinference | openai_compatible
RAG_ANSWER_PROVIDER=fallback | ark | xinference | openai_compatible
RAG_RERANK_PROVIDER=xinference | openai_compatible
```

See `docs/MODEL_GATEWAY.md` for complete examples.

## Diagnostics

```text
05_evaluate.py                  Retrieval/rerank metrics
08_compare_methods.py           Compare G methods by question type
14_chunk_quality_report.py      Chunk quality report
```

Generated artifacts are written to `outputs/` and are ignored by Git by default.

## Current Recommended Flow

The current recommended research pipeline is:

```text
ABECD + Evidence Guard + SelfCorrect merge-v2
```

This means:

```text
A  context expansion
B  query-adaptive reranking
E  GraphRAG graph/context enhancement
C  table-structure enhancement
D  visual evidence enhancement
Guard  evidence-chain completeness constraints
SelfCorrect merge-v2  primary/fallback evidence verification and merge
```

Use `52_self_correct_evidence.py` after a stable primary run and a multiroute fallback run when you want the final evidence chains to keep high-precision ordering while supplementing missing visual/table/bridge evidence.
