# Script Catalog

`scripts/` is the command-line workspace for the project. The files remain at the top level for backward-compatible commands, but they are grouped conceptually as follows.

Production-facing imports should prefer `multirank_rag/`. The scripts below are still maintained as CLI entrypoints and research utilities.

## Core Pipeline

| Script | Purpose |
|---|---|
| `01_parse_pdf.py` | Parse PDFs into multimodal evidence nodes. |
| `02_build_graph.py` | Build document structure edges. |
| `03_retrieve_candidates.py` | Retrieve candidate evidence nodes. |
| `04_rerank.py` | Run G0-G4 MultiRank reranking. |
| `06_run_pipeline.py` | End-to-end local pipeline wrapper. |
| `09_build_evidence_chains.py` | Build question-level evidence chains. |
| `10_build_visual_evidence.py` | Generate image crops, OCR/caption, and visual evidence. |
| `11_build_evidence_cards.py` | Render evidence card images. |
| `13_export_frontend_data.py` | Export artifacts consumed by the frontend demo. |
| `23_build_graphrag.py` | Build GraphRAG entities, relations, and communities. |
| `34_generate_chain_answers.py` | Generate final answers from evidence chains. |
| `40_run_main_experiment.py` | Main experiment runner for V0-V4/G0-G4 variants. |
| `42_enhance_multimodal_nodes.py` | Enhance table and visual nodes. |
| `52_self_correct_evidence.py` | Verify, merge, or replace evidence chains. |

Query planning and answer self-correction are implemented in `rerank_lib.py` and exposed through `multirank_rag/rerank/` and `multirank_rag/evidence/answers.py`.

## Shared Libraries

| Script | Purpose |
|---|---|
| `pipeline_common.py` | Shared file, CSV/JSONL, text, and path helpers. |
| `embedding_index.py` | Embedding provider abstraction and cached vector index. |
| `rerank_lib.py` | Retrieval, GraphRAG scoring, MultiRank, and answer helpers. |
| `ark_clients.py` | Ark/Doubao, Xinference, and OpenAI-compatible clients. |
| `query_expansion.py` | Optional product-aware query expansion helper. |

## Evaluation and Diagnostics

| Script | Purpose |
|---|---|
| `05_evaluate.py` | Retrieval and reranking metrics. |
| `08_compare_methods.py` | Compare G0-G4 methods by question type. |
| `12_check_evidence_cards.py` | Check generated evidence card completeness. |
| `12_evaluate_evidence_chains.py` | Evidence-chain metrics. |
| `14_chunk_quality_report.py` | Chunk quality diagnostics. |
| `15_visualize_layout_bboxes.py` | Layout/bbox visualization helper. |
| `50_export_experiment_summary.py` | Export experiment summaries to Excel/CSV. |

## Data and Benchmark Preparation

| Script | Purpose |
|---|---|
| `07_write_multimodal_questions.py` | Generate project question CSV files. |
| `41_prepare_benchmark.py` | Prepare benchmark subsets. |
