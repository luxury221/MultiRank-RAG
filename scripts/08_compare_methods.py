from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from pipeline_common import as_float, clean_text, ensure_project_dirs, read_csv, resolve_path, write_csv


COMPARISON_FIELDS = [
    "question_id",
    "question_type",
    "doc_id",
    "question",
    "baseline",
    "target",
    "recall_at_5_delta",
    "ndcg_at_5_delta",
    "mrr_delta",
    "baseline_top1_node",
    "baseline_top1_type",
    "target_top1_node",
    "target_top1_type",
    "status",
]

TYPE_FIELDS = [
    "question_type",
    "num_questions",
    "baseline",
    "target",
    "avg_recall_at_5_delta",
    "avg_ndcg_at_5_delta",
    "avg_mrr_delta",
    "improved",
    "regressed",
    "unchanged",
]


def delta_status(ndcg_delta: float, recall_delta: float) -> str:
    if ndcg_delta > 1e-9 or recall_delta > 0:
        return "improved"
    if ndcg_delta < -1e-9 or recall_delta < 0:
        return "regressed"
    return "unchanged"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two ranking methods question by question.")
    parser.add_argument("--questions", default="data/questions.csv")
    parser.add_argument("--metrics", default="outputs/metrics/per_question_metrics.csv")
    parser.add_argument("--baseline", default="G1")
    parser.add_argument("--target", default="G3")
    parser.add_argument("--output", default="outputs/metrics/g1_g3_comparison.csv")
    parser.add_argument("--type-output", default="outputs/metrics/type_comparison.csv")
    args = parser.parse_args()

    ensure_project_dirs()
    questions = {row.get("question_id", ""): row for row in read_csv(args.questions)}
    metrics = read_csv(args.metrics)
    by_q_method = {(row.get("question_id", ""), row.get("method", "")): row for row in metrics}

    rows: list[dict[str, Any]] = []
    for question_id, question in sorted(questions.items()):
        baseline = by_q_method.get((question_id, args.baseline))
        target = by_q_method.get((question_id, args.target))
        if not baseline or not target:
            continue
        recall_delta = as_float(target.get("recall_at_5")) - as_float(baseline.get("recall_at_5"))
        ndcg_delta = as_float(target.get("ndcg_at_5")) - as_float(baseline.get("ndcg_at_5"))
        mrr_delta = as_float(target.get("mrr")) - as_float(baseline.get("mrr"))
        rows.append(
            {
                "question_id": question_id,
                "question_type": question.get("question_type", ""),
                "doc_id": question.get("doc_id", ""),
                "question": question.get("question", ""),
                "baseline": args.baseline,
                "target": args.target,
                "recall_at_5_delta": round(recall_delta, 6),
                "ndcg_at_5_delta": round(ndcg_delta, 6),
                "mrr_delta": round(mrr_delta, 6),
                "baseline_top1_node": baseline.get("top1_node", ""),
                "baseline_top1_type": baseline.get("top1_type", ""),
                "target_top1_node": target.get("top1_node", ""),
                "target_top1_type": target.get("top1_type", ""),
                "status": delta_status(ndcg_delta, recall_delta),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clean_text(row.get("question_type"))].append(row)

    type_rows: list[dict[str, Any]] = []
    for question_type, group in sorted(grouped.items()):
        total = max(1, len(group))
        type_rows.append(
            {
                "question_type": question_type,
                "num_questions": len(group),
                "baseline": args.baseline,
                "target": args.target,
                "avg_recall_at_5_delta": round(
                    sum(as_float(row.get("recall_at_5_delta")) for row in group) / total, 6
                ),
                "avg_ndcg_at_5_delta": round(sum(as_float(row.get("ndcg_at_5_delta")) for row in group) / total, 6),
                "avg_mrr_delta": round(sum(as_float(row.get("mrr_delta")) for row in group) / total, 6),
                "improved": sum(1 for row in group if row.get("status") == "improved"),
                "regressed": sum(1 for row in group if row.get("status") == "regressed"),
                "unchanged": sum(1 for row in group if row.get("status") == "unchanged"),
            }
        )

    write_csv(args.output, rows, COMPARISON_FIELDS)
    write_csv(args.type_output, type_rows, TYPE_FIELDS)
    print(f"Wrote comparison rows to {resolve_path(args.output)}")
    print(f"Wrote type comparison rows to {resolve_path(args.type_output)}")


if __name__ == "__main__":
    main()
