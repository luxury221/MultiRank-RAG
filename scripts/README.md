# Scripts

`scripts/` contains the offline pipeline for parsing PDFs, building graph indexes, retrieving evidence, reranking, and generating evidence chains/cards.

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
```

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
24_ablate_retrieval.py          Retrieval ablation
31_diagnose_submission_quality.py  Output quality diagnosis
```

Generated artifacts are written to `outputs/` and are ignored by Git by default.
