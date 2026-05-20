# Main Experiments

The main experiment is designed as an ablation study. Its goal is not only to report a final answer score, but to show which module improves evidence retrieval, visual grounding, GraphRAG reasoning, and evidence-chain readiness.

## Variants

```text
V0  Vanilla text-only RAG
    All evidence nodes are treated as text nodes. Visual fields and graph signals are removed.

V1  + Structured evidence nodes
    Keeps text/table/figure/caption/page node types, but removes visual caption/crop fields.

V2  + Visual evidence fields
    Keeps visual captions, crops, OCR-like fields, key objects, and QA evidence fields.

V3  + GraphRAG retrieval signal
    Builds a GraphRAG entity/relation/community index and lets fusion retrieval use it.

V4  Full MultiRank-RAG evidence chain
    Uses GraphRAG plus G4 MultiRank reranking, visual grounding, chain signals, and optional model rerank.

V5  Enhanced MultiRank-RAG (ABECD + Guard)
    Adds context expansion, adaptive rerank boost, graph-context boost, existing table/visual evidence fields,
    and evidence guard on top of the V4 G4 chain pipeline.
```

This lets the report explain improvements as:

```text
V0 -> V1: structured PDF chunking / node typing
V1 -> V2: multimodal visual evidence enrichment
V2 -> V3: GraphRAG relationship modeling
V3 -> V4: MultiRank reranking and evidence-chain organization
V4 -> V5: ABECD enhanced retrieval/rerank plus evidence guard
```

## Quick Smoke Test

This uses the built-in sample data and avoids external model calls by forcing BM25:

```bash
python scripts/40_run_main_experiment.py \
  --dataset-name sample \
  --run-name smoke \
  --variants V0,V1,V2,V3,V4,V5 \
  --retriever bm25 \
  --candidate-k 10 \
  --rerank-k 3 \
  --clean
```

The summary will be written to:

```text
outputs/experiments/sample/smoke/main_experiment_summary.csv
```

## Running On A Converted Benchmark

Each benchmark should first be converted into the project common format:

```text
nodes.jsonl       evidence nodes: text/table/figure/caption/page/equation
questions.csv     question_id, doc_id, question, gold_node_ids/gold_pages/gold_modalities
```

Then run:

```bash
python scripts/40_run_main_experiment.py \
  --dataset-name m3docvqa \
  --nodes outputs/benchmarks/m3docvqa/nodes.jsonl \
  --questions outputs/benchmarks/m3docvqa/questions.csv \
  --run-name pilot_100 \
  --candidate-k 50 \
  --rerank-k 10 \
  --clean
```

## Recommended Main Benchmarks

```text
RAGBench        Basic text RAG capability
MultiHop-RAG    Multi-hop and cross-document retrieval
M3DocVQA        Multimodal PDF / document VQA
T²-RAGBench     Text-table structured RAG
```

ALCE-style citation and faithfulness metrics can be added on top of these outputs, especially for V4 evidence-chain evaluation.

## Public Benchmark Run

The current repository records a full public benchmark ablation on four converted datasets:

```text
RAGBench eManual      text/manual QA
T2/FinQA              table and numeric reasoning
MMLongBench-Doc       multimodal long-document grounding
MultiHop-RAG          multi-hop and cross-document retrieval
```

To control variables, the run fixed candidate retrieval to BM25 and disabled answer generation:

```bash
python scripts/40_run_main_experiment.py \
  --dataset-name <dataset_name> \
  --nodes outputs/benchmarks/<dataset_name>/nodes.jsonl \
  --questions outputs/benchmarks/<dataset_name>/questions.csv \
  --run-name public_bm25_v0v5_full_20260520 \
  --variants V0,V1,V2,V3,V4,V5 \
  --retriever bm25 \
  --candidate-k 50 \
  --rerank-k 10 \
  --build-chains \
  --answer-provider none \
  --clean
```

Tracked summary files:

```text
docs/PUBLIC_BENCHMARK_RESULTS.md
docs/public_benchmark_ablation_20260520.csv
```

Main findings:

| Dataset | Strongest signal |
|---|---|
| RAGBench eManual | Baseline is already saturated at Recall@5/10=1.000, useful as a negative control. |
| T2/FinQA | V5 improves nDCG@5 from 0.678 to 0.878 under the same BM25 candidate set. |
| MMLongBench-Doc | V1-V5 recover modality coverage and V2+ recover visual-grounding indicators; low recall shows BM25 is the bottleneck. |
| MultiHop-RAG | V4/V5 improve Recall@10 from 0.900 to 0.910; V5 improves evidence-chain score from 0.864 to 0.890. |

## Primary Metrics

The current evaluator reports:

```text
Recall@1 / Recall@3 / Recall@5 / Recall@10
MRR
nDCG@5
Evidence Hit
Modality Hit
Citation Correct
Visual Grounding Hit
Visual Caption Hit
Evidence Chain Ready
Average Rerank Time
```

For the final report, the highest-signal table is usually:

```text
variant, recall_at_5, recall_at_10, mrr, ndcg_at_5,
evidence_hit, visual_grounding_hit, evidence_chain_ready
```

## Lightweight Smoke Snapshot

The repository keeps generated artifacts out of Git. The earlier lightweight `data/sample` snapshot is kept only as a smoke experiment for checking whether the pipeline can run end to end; the public benchmark section above is the current main ablation evidence.

```bash
python scripts/40_run_main_experiment.py \
  --dataset-name sample \
  --run-name readme_ablation_20260519 \
  --variants V0,V1,V2,V3,V4 \
  --retriever bm25 \
  --candidate-k 10 \
  --rerank-k 3 \
  --build-chains \
  --answer-provider none \
  --clean
```

Retrieval and reranking summary:

| Variant | Label | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | Evidence Hit | Modality Hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | Vanilla text-only RAG | 0.600 | 0.800 | 0.800 | 0.800 | 0.700 | 0.726 | 0.800 | 0.400 |
| V1 | + structured evidence nodes | 0.600 | 1.000 | 1.000 | 1.000 | 0.767 | 0.826 | 1.000 | 1.000 |
| V2 | + visual evidence fields | 0.600 | 1.000 | 1.000 | 1.000 | 0.767 | 0.826 | 1.000 | 1.000 |
| V3 | + GraphRAG retrieval signal | 0.600 | 1.000 | 1.000 | 1.000 | 0.767 | 0.826 | 1.000 | 1.000 |
| V4 | Full MultiRank-RAG evidence chain | 0.800 | 1.000 | 1.000 | 1.000 | 0.900 | 0.926 | 1.000 | 1.000 |

V4 evidence-chain summary:

| Metric | Value |
|---|---:|
| chain_present | 1.000 |
| avg_step_count | 5.000 |
| gold_node_coverage | 1.000 |
| gold_page_hit | 1.000 |
| gold_modality_coverage | 1.000 |
| visual_grounding_hit | 0.400 |
| cross_modal_hit | 1.000 |
| relation_support | 1.000 |
| evidence_chain_score | 0.916 |

Main observation: V1 shows that structured evidence nodes are the largest immediate gain over text-only chunking. V4 improves ranking quality further, especially Recall@1, MRR, and nDCG@5. The remaining bottleneck is fine-grained visual grounding, which should be evaluated on a larger multimodal benchmark after the smoke test.
