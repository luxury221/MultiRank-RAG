from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any

from pipeline_common import (
    DEFAULT_QUESTIONS,
    DEFAULT_RANKINGS,
    DEFAULT_SUMMARY,
    as_float,
    clean_text,
    ensure_project_dirs,
    read_csv,
    resolve_path,
    split_ints,
    split_multi,
    write_csv,
)


PER_QUESTION_FIELDS = [
    "question_id",
    "question_type",
    "method",
    "top1_node",
    "top1_type",
    "top1_page",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_5",
    "evidence_hit",
    "modality_hit",
    "citation_correct",
    "visual_required",
    "visual_grounding_hit",
    "visual_caption_hit",
    "evidence_chain_ready",
    "rerank_time_ms",
]

SUMMARY_FIELDS = [
    "method",
    "num_questions",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_5",
    "evidence_hit",
    "modality_hit",
    "citation_correct",
    "visual_required_questions",
    "visual_grounding_hit",
    "visual_caption_hit",
    "evidence_chain_ready",
    "avg_rerank_time_ms",
]


VISUAL_MODALITIES = {"table", "figure", "caption"}


def is_relevant(row: dict[str, Any], gold_nodes: set[str], gold_pages: set[int], gold_modalities: set[str]) -> bool:
    node_id = clean_text(row.get("node_id"))
    if gold_nodes:
        return node_id in gold_nodes
    page = int(float(row.get("page") or 0))
    node_type = clean_text(row.get("node_type"))
    if gold_pages and gold_modalities:
        return page in gold_pages and node_type in gold_modalities
    if gold_pages:
        return page in gold_pages
    return False


def ndcg_at_k(relevances: list[int], ideal_relevant_count: int, k: int = 5) -> float:
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel:
            dcg += 1.0 / math.log2(idx + 1)
    ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, min(ideal_relevant_count, k) + 1))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    return text in {"1", "true", "yes", "y"}


def question_requires_visual(question: dict[str, str], gold_modalities: set[str]) -> bool:
    qtype = clean_text(question.get("question_type"))
    query = clean_text(question.get("question")).lower()
    if gold_modalities & VISUAL_MODALITIES:
        return True
    visual_markers = [
        "table",
        "figure",
        "fig.",
        "chart",
        "plot",
        "\u8868",
        "\u8868\u683c",
        "\u56fe",
        "\u56fe\u8868",
        "\u56fe\u6587",
        "\u8de8\u6a21\u6001",
    ]
    return any(marker in qtype or marker in query for marker in visual_markers)


def evaluate_question_method(
    question: dict[str, str],
    rows: list[dict[str, str]],
    method: str,
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row.get("rank") or 0))
    gold_nodes = set(split_multi(question.get("gold_node_ids")))
    gold_pages = split_ints(question.get("gold_pages"))
    gold_modalities = {item.lower() for item in split_multi(question.get("gold_modalities"))}
    rel = [1 if is_relevant(row, gold_nodes, gold_pages, gold_modalities) else 0 for row in rows]
    first_hit_rank = next((idx for idx, value in enumerate(rel, start=1) if value), None)
    top5 = rows[:5]
    modality_hit = 0
    if gold_modalities:
        modality_hit = int(any(clean_text(row.get("node_type")).lower() in gold_modalities for row in top5))
    citation_correct = 0
    if gold_pages:
        citation_correct = int(any(int(float(row.get("page") or 0)) in gold_pages for row in top5))
    visual_required = int(question_requires_visual(question, gold_modalities))
    visual_rows = [row for row in top5 if clean_text(row.get("node_type")).lower() in VISUAL_MODALITIES]
    visual_grounding_hit = int(
        bool(visual_rows) and any(as_bool(row.get("has_visual_crop")) for row in visual_rows)
    )
    visual_caption_hit = int(
        bool(visual_rows) and any(as_bool(row.get("has_visual_caption")) for row in visual_rows)
    )
    evidence_chain_ready = int(any(rel[:5]) and (not visual_required or visual_grounding_hit))
    top1 = rows[0] if rows else {}
    ideal_count = len(gold_nodes) if gold_nodes else 1
    return {
        "question_id": question.get("question_id", ""),
        "question_type": question.get("question_type", ""),
        "method": method,
        "top1_node": top1.get("node_id", ""),
        "top1_type": top1.get("node_type", ""),
        "top1_page": top1.get("page", ""),
        "recall_at_1": int(any(rel[:1])),
        "recall_at_3": int(any(rel[:3])),
        "recall_at_5": int(any(rel[:5])),
        "recall_at_10": int(any(rel[:10])),
        "mrr": round(1.0 / first_hit_rank, 6) if first_hit_rank else 0.0,
        "ndcg_at_5": round(ndcg_at_k(rel, ideal_count, 5), 6),
        "evidence_hit": int(any(rel[:5])),
        "modality_hit": modality_hit,
        "citation_correct": citation_correct,
        "visual_required": visual_required,
        "visual_grounding_hit": visual_grounding_hit if visual_required else 0,
        "visual_caption_hit": visual_caption_hit if visual_required else 0,
        "evidence_chain_ready": evidence_chain_ready,
        "rerank_time_ms": round(max(as_float(row.get("rerank_time_ms")) for row in rows), 3) if rows else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary: list[dict[str, Any]] = []
    metric_fields = [
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "evidence_hit",
        "modality_hit",
        "citation_correct",
        "evidence_chain_ready",
    ]
    for method, method_rows in sorted(grouped.items()):
        item: dict[str, Any] = {"method": method, "num_questions": len(method_rows)}
        for field in metric_fields:
            item[field] = round(sum(as_float(row.get(field)) for row in method_rows) / max(1, len(method_rows)), 6)
        visual_rows = [row for row in method_rows if as_float(row.get("visual_required")) > 0]
        item["visual_required_questions"] = len(visual_rows)
        for field in ["visual_grounding_hit", "visual_caption_hit"]:
            item[field] = round(sum(as_float(row.get(field)) for row in visual_rows) / max(1, len(visual_rows)), 6)
        item["avg_rerank_time_ms"] = round(
            sum(as_float(row.get("rerank_time_ms")) for row in method_rows) / max(1, len(method_rows)),
            3,
        )
        summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate G0-G3 ranking outputs.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS.relative_to(DEFAULT_QUESTIONS.parents[1])))
    parser.add_argument("--rankings", default=str(DEFAULT_RANKINGS.relative_to(DEFAULT_RANKINGS.parents[2])))
    parser.add_argument("--per-question-output", default="outputs/metrics/per_question_metrics.csv")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY.relative_to(DEFAULT_SUMMARY.parents[2])))
    args = parser.parse_args()

    ensure_project_dirs()
    questions = {row.get("question_id", ""): row for row in read_csv(args.questions) if clean_text(row.get("question"))}
    ranking_rows = read_csv(args.rankings)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ranking_rows:
        grouped[(row.get("question_id", ""), row.get("method", ""))].append(row)

    per_question: list[dict[str, Any]] = []
    for (question_id, method), rows in sorted(grouped.items()):
        question = questions.get(question_id)
        if not question:
            continue
        per_question.append(evaluate_question_method(question, rows, method))

    write_csv(args.per_question_output, per_question, PER_QUESTION_FIELDS)
    summary = summarize(per_question)
    write_csv(args.summary_output, summary, SUMMARY_FIELDS)
    print(f"Wrote per-question metrics to {resolve_path(args.per_question_output)}")
    print(f"Wrote summary metrics to {resolve_path(args.summary_output)}")
    if not per_question:
        print("No metrics produced. Check rankings and gold labels in the question file.")


if __name__ == "__main__":
    main()
