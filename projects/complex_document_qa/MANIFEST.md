# Complex Document QA Manifest

## Product Code

```text
backend/
web/
multirank_rag/
scripts/01_parse_pdf.py
scripts/02_build_graph.py
scripts/03_retrieve_candidates.py
scripts/04_rerank.py
scripts/06_run_pipeline.py
scripts/09_build_evidence_chains.py
scripts/10_build_visual_evidence.py
scripts/11_build_evidence_cards.py
scripts/23_build_graphrag.py
scripts/34_generate_chain_answers.py
scripts/40_run_main_experiment.py
scripts/42_enhance_multimodal_nodes.py
scripts/50_export_experiment_summary.py
scripts/52_self_correct_evidence.py
```

## Shared Libraries

```text
scripts/pipeline_common.py
scripts/embedding_index.py
scripts/rerank_lib.py
scripts/ark_clients.py
scripts/query_expansion.py
```

## Project Documents

```text
README.md
docs/ARCHITECTURE.md
docs/GRAPHRAG.md
docs/MODEL_GATEWAY.md
docs/EVALUATION.md
docs/EXPERIMENTS.md
docs/PUBLIC_BENCHMARK_RESULTS.md
projects/complex_document_qa/reports/开题报告.docx
```

## Demo Data

```text
data/sample/
data/pdfs/README.md
data/questions.csv
data/question.csv
```

真实 PDF、用户上传文件、模型权重、运行输出和私有 API 配置保持本地管理。
