# Public Benchmark Ablation Results

This report records the full V0-V5 public benchmark run used for the repository documentation. Generated artifacts under `outputs/` are intentionally ignored by Git; this file and `docs/public_benchmark_ablation_20260520.csv` keep the reproducible summary inside the repository.

## Run Setup

| Item | Value |
|---|---|
| Run name | `public_bm25_v0v5_full_20260520` |
| Date | 2026-05-20 |
| Variants | `V0,V1,V2,V3,V4,V5` |
| Retriever | `bm25` fixed for all variants |
| Candidate / rerank k | `50 / 10` |
| Answer generation | disabled, `--answer-provider none` |
| Evidence chains | enabled, with evidence guard in V5 |

## Variant Definitions

| Variant | Meaning |
|---|---|
| V0 | Vanilla text-only RAG baseline |
| V1 | Structured evidence nodes: text/table/figure/caption/page types |
| V2 | Visual evidence fields: caption/OCR/visual summary/QA evidence |
| V3 | GraphRAG retrieval signal |
| V4 | Full MultiRank-RAG evidence chain with G4 reranking |
| V5 | Enhanced MultiRank-RAG with ABECD + Evidence Guard |

## Main Retrieval Metrics

### RAGBench eManual

Task focus: text/manual QA.

| Variant | Method | N | R@1 | R@5 | R@10 | MRR | nDCG@5 | Evidence Hit | Modality Hit | Visual Grounding | Chain Ready |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | G1 | 66 | 0.833 | 1.000 | 1.000 | 0.899 | 1.183 | 1.000 | 1.000 | 0.000 | 0.924 |
| V1 | G1 | 66 | 0.833 | 1.000 | 1.000 | 0.899 | 1.183 | 1.000 | 1.000 | 0.000 | 0.924 |
| V2 | G1 | 66 | 0.833 | 1.000 | 1.000 | 0.899 | 1.183 | 1.000 | 1.000 | 0.000 | 0.924 |
| V3 | G1 | 66 | 0.833 | 1.000 | 1.000 | 0.899 | 1.183 | 1.000 | 1.000 | 0.000 | 0.924 |
| V4 | G4 | 66 | 0.833 | 1.000 | 1.000 | 0.899 | 1.183 | 1.000 | 1.000 | 0.000 | 0.924 |
| V5 | G4 | 66 | 0.833 | 1.000 | 1.000 | 0.894 | 1.156 | 1.000 | 1.000 | 0.000 | 0.924 |

### T2/FinQA

Task focus: table and numeric reasoning.

| Variant | Method | N | R@1 | R@5 | R@10 | MRR | nDCG@5 | Evidence Hit | Modality Hit | Visual Grounding | Chain Ready |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | G1 | 100 | 0.830 | 0.990 | 1.000 | 0.908 | 0.678 | 0.990 | 1.000 | 0.000 | 0.000 |
| V1 | G1 | 100 | 0.830 | 0.990 | 1.000 | 0.908 | 0.678 | 0.990 | 1.000 | 0.000 | 0.000 |
| V2 | G1 | 100 | 0.840 | 0.990 | 1.000 | 0.913 | 0.683 | 0.990 | 1.000 | 0.000 | 0.000 |
| V3 | G1 | 100 | 0.840 | 0.990 | 1.000 | 0.913 | 0.683 | 0.990 | 1.000 | 0.000 | 0.000 |
| V4 | G4 | 100 | 0.840 | 0.990 | 1.000 | 0.913 | 0.695 | 0.990 | 1.000 | 0.000 | 0.000 |
| V5 | G4 | 100 | 0.840 | 0.990 | 1.000 | 0.895 | 0.878 | 0.990 | 1.000 | 0.000 | 0.000 |

### MMLongBench-Doc

Task focus: multimodal long-document grounding.

| Variant | Method | N | R@1 | R@5 | R@10 | MRR | nDCG@5 | Evidence Hit | Modality Hit | Visual Grounding | Chain Ready |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | G1 | 100 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 | 0.000 | 0.000 | 0.000 |
| V1 | G1 | 100 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 | 1.000 | 0.000 | 0.000 |
| V2 | G1 | 100 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 | 1.000 | 1.000 | 0.070 |
| V3 | G1 | 100 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 | 1.000 | 1.000 | 0.070 |
| V4 | G4 | 100 | 0.070 | 0.070 | 0.090 | 0.073 | 0.070 | 0.070 | 1.000 | 1.000 | 0.070 |
| V5 | G4 | 100 | 0.070 | 0.070 | 0.090 | 0.073 | 0.070 | 0.070 | 1.000 | 1.000 | 0.070 |

### MultiHop-RAG

Task focus: multi-hop and cross-document retrieval.

| Variant | Method | N | R@1 | R@5 | R@10 | MRR | nDCG@5 | Evidence Hit | Modality Hit | Visual Grounding | Chain Ready |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | G1 | 100 | 0.670 | 0.890 | 0.900 | 0.755 | 0.615 | 0.890 | 1.000 | 0.000 | 0.850 |
| V1 | G1 | 100 | 0.670 | 0.890 | 0.900 | 0.755 | 0.615 | 0.890 | 1.000 | 0.000 | 0.850 |
| V2 | G1 | 100 | 0.670 | 0.890 | 0.900 | 0.755 | 0.615 | 0.890 | 1.000 | 0.000 | 0.850 |
| V3 | G1 | 100 | 0.670 | 0.890 | 0.900 | 0.755 | 0.615 | 0.890 | 1.000 | 0.000 | 0.850 |
| V4 | G4 | 100 | 0.660 | 0.890 | 0.910 | 0.752 | 0.616 | 0.890 | 1.000 | 0.000 | 0.850 |
| V5 | G4 | 100 | 0.660 | 0.890 | 0.910 | 0.752 | 0.615 | 0.890 | 1.000 | 0.000 | 0.850 |

## Evidence Chain Metrics

| Dataset | Variant | Chain Score | Gold Node Coverage | Avg Steps |
|---|---|---:|---:|---:|
| RAGBench eManual | V4 | 0.981 | 1.000 | 3.000 |
| RAGBench eManual | V5 | 0.981 | 1.000 | 4.340 |
| T2/FinQA | V4 | 0.981 | 0.950 | 5.000 |
| T2/FinQA | V5 | 0.979 | 0.965 | 4.920 |
| MMLongBench-Doc | V4 | 0.530 | 0.015 | 4.760 |
| MMLongBench-Doc | V5 | 0.527 | 0.009 | 2.830 |
| MultiHop-RAG | V4 | 0.864 | 0.662 | 4.960 |
| MultiHop-RAG | V5 | 0.890 | 0.730 | 4.960 |

## Observations

- On **T2/FinQA**, V5 raises nDCG@5 from 0.678 to 0.878 under the same BM25 candidate set, showing that table/context-aware reranking improves evidence ordering even when Recall@5 is already saturated.
- On **MMLongBench-Doc**, V1-V5 recover modality coverage and V2+ recover visual-grounding indicators, but absolute recall stays low. This is a retrieval bottleneck under BM25 and points to the next upgrade: embedding-vision or ColPali-style page retrieval.
- On **MultiHop-RAG**, V4/V5 improve Recall@10 from 0.900 to 0.910. V5 improves evidence-chain score from 0.864 to 0.890 and gold-node coverage from 0.662 to 0.730 compared with V4.
- On **RAGBench eManual**, the baseline is already saturated at Recall@5/10. V5 does not improve this simple text/manual setting, which is useful as a negative control showing the enhanced pipeline is most valuable for structured, multimodal, or multi-hop cases.

## Reproduction Command

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
