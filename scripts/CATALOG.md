# Script Catalog

`scripts/` is the command-line workspace for the project. The files remain at the top level for backward-compatible commands, but they are grouped conceptually as follows.

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

## Shared Libraries

| Script | Purpose |
|---|---|
| `pipeline_common.py` | Shared file, CSV/JSONL, text, and path helpers. |
| `embedding_index.py` | Embedding provider abstraction and cached vector index. |
| `rerank_lib.py` | Retrieval, GraphRAG scoring, MultiRank, and answer helpers. |
| `ark_clients.py` | Ark/Doubao, Xinference, and OpenAI-compatible clients. |

## Evaluation and Diagnostics

| Script | Purpose |
|---|---|
| `05_evaluate.py` | Retrieval and reranking metrics. |
| `08_compare_methods.py` | Compare G0-G4 methods by question type. |
| `12_check_evidence_cards.py` | Check generated evidence card completeness. |
| `12_evaluate_evidence_chains.py` | Evidence-chain metrics. |
| `14_chunk_quality_report.py` | Chunk quality diagnostics. |
| `15_visualize_layout_bboxes.py` | Layout/bbox visualization helper. |
| `24_ablate_retrieval.py` | Retrieval ablation for the project pipeline. |
| `31_diagnose_submission_quality.py` | Historical answer-quality diagnostics. |
| `50_export_experiment_summary.py` | Export experiment summaries to Excel/CSV. |

## Data and Benchmark Preparation

| Script | Purpose |
|---|---|
| `07_write_multimodal_questions.py` | Generate project question CSV files. |
| `19_build_after_sales_kb.py` | Build sample after-sales knowledge base. |
| `22_enrich_images.py` | Enrich image nodes for local datasets. |
| `23_build_kg.py` | Build earlier KG artifacts. |
| `27_build_union_candidates.py` | Build union candidate pools. |
| `28_product_route_questions.py` | Route product/manual questions. |
| `41_prepare_benchmark.py` | Prepare benchmark subsets. |

## Competition and Historical Utilities

These scripts are preserved for experiment traceability. They are not the main product interface.

| Script | Purpose |
|---|---|
| `16_import_datafountain.py` | Import historical DataFountain-format data. |
| `17_generate_datafountain_submission.py` | Historical submission generator. |
| `18_generate_datafountain_submission_llm.py` | Historical LLM submission generator. |
| `20_generate_after_sales_submission.py` | Historical after-sales answer export. |
| `21_generate_competition_submission_llm.py` | Historical competition LLM generation. |
| `22_enrich_datafountain_images.py` | Historical DataFountain image enrichment. |
| `23_build_datafountain_kg.py` | Historical DataFountain KG builder. |
| `24_ablate_datafountain_retrieval.py` | Historical DataFountain ablation. |
| `25_build_calibrated_submission.py` | Historical calibrated export. |
| `26_judge_submissions.py` | Historical LLM judge helper. |
| `29_refine_submission_answers.py` | Historical answer refinement. |
| `30_postprocess_datafountain_submission.py` | Historical post-processing. |
| `32_build_targeted_submission.py` | Historical targeted export. |
| `33_build_last_submission.py` | Local one-off export, ignored by Git. |
| `datafountain_query_expansion.py` | Historical query expansion helper. |
