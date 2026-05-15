# Scripts

`scripts/` 存放项目的离线流水线和实验工具。核心脚本按端到端流程排列。

## Core Pipeline

```text
01_parse_pdf.py
  Parse PDF files into evidence nodes.

02_build_graph.py
  Build adjacency edges between evidence nodes.

03_retrieve_candidates.py
  Retrieve candidate evidence with BM25, lexical, embedding, KG, or fusion.

04_rerank.py
  Apply G0-G4 MultiRank reranking.

09_build_evidence_chains.py
  Assemble evidence chains for questions.

10_build_visual_evidence.py
  Enrich figure/table nodes with visual captions, OCR and QA evidence.

11_build_evidence_cards.py
  Render evidence cards for UI and demonstrations.

13_export_frontend_data.py
  Export static data consumed by the frontend.

31_diagnose_submission_quality.py
  Diagnose output structure and common answer quality risks.
```

## One-Command Pipeline

```bash
python scripts/06_run_pipeline.py \
  --questions data/questions.csv \
  --candidate-k 50 \
  --rerank-k 10
```

Useful switches:

```bash
--skip-parse
--skip-visual
--skip-kg
--candidate-retriever fusion
--rerank-retriever fusion
--chunk-template auto
```

## Implementation Notes

- `rerank_lib.py` contains most retrieval and reranking scoring logic.
- `embedding_index.py` wraps embedding generation and cache management.
- `ark_clients.py` wraps Ark / Doubao-compatible API calls.
- Data-specific import and benchmark scripts are kept here for reproducibility, but they are not the center of the project architecture.

