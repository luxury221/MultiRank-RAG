from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from pipeline_common import (
    as_float,
    clean_text,
    ensure_project_dirs,
    read_csv,
    read_jsonl,
    resolve_path,
    split_ints,
    split_multi,
    write_csv,
)
from rerank_lib import adaptive_rag_route


VISUAL_MODALITIES = {"table", "figure", "caption"}
CHAIN_PER_QUESTION_FIELDS = [
    "question_id",
    "question_type",
    "adaptive_route",
    "chain_present",
    "step_count",
    "unique_node_types",
    "gold_node_coverage",
    "gold_page_hit",
    "gold_modality_coverage",
    "visual_required",
    "visual_grounding_hit",
    "cross_modal_hit",
    "relation_support",
    "chain_length_score",
    "evidence_chain_score",
]

CHAIN_SUMMARY_FIELDS = [
    "route",
    "num_questions",
    "chain_present",
    "avg_step_count",
    "gold_node_coverage",
    "gold_page_hit",
    "gold_modality_coverage",
    "visual_grounding_hit",
    "cross_modal_hit",
    "relation_support",
    "evidence_chain_score",
]


def _safe_page(value: Any) -> int | None:
    try:
        text = clean_text(value)
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def question_requires_visual(question: dict[str, Any], gold_modalities: set[str]) -> bool:
    if gold_modalities & VISUAL_MODALITIES:
        return True
    blob = clean_text(f"{question.get('question_type', '')} {question.get('question', '')}").lower()
    return any(
        marker in blob
        for marker in (
            "table",
            "figure",
            "fig.",
            "chart",
            "plot",
            "multimodal",
            "cross-modal",
            "\u8868",
            "\u8868\u683c",
            "\u56fe",
            "\u56fe\u8868",
            "\u56fe\u6587",
            "\u8de8\u6a21\u6001",
        )
    )


def step_has_visual_grounding(step: dict[str, Any]) -> bool:
    if clean_text(step.get("crop_image_path")) or clean_text(step.get("page_image_path")):
        return True
    if clean_text(step.get("bbox")) or clean_text(step.get("bbox_source")):
        return True
    return bool(clean_text(step.get("visual_caption")) or clean_text(step.get("visual_summary")))


def evaluate_chain(question: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    gold_nodes = set(split_multi(question.get("gold_node_ids")))
    gold_pages = split_ints(question.get("gold_pages"))
    gold_modalities = {item.lower() for item in split_multi(question.get("gold_modalities"))}
    route = adaptive_rag_route(question)

    step_nodes = {clean_text(step.get("node_id")) for step in steps if clean_text(step.get("node_id"))}
    step_pages = {_safe_page(step.get("page")) for step in steps}
    step_pages = {page for page in step_pages if page is not None}
    step_types = {clean_text(step.get("node_type")).lower() for step in steps if clean_text(step.get("node_type"))}
    visual_required = question_requires_visual(question, gold_modalities)

    if gold_nodes:
        gold_node_coverage = len(step_nodes & gold_nodes) / max(1, len(gold_nodes))
    elif gold_pages:
        gold_node_coverage = 1.0 if gold_pages and step_pages & gold_pages else 0.0
    else:
        gold_node_coverage = 1.0 if steps else 0.0

    gold_page_hit = 1.0 if gold_pages and step_pages & gold_pages else (1.0 if not gold_pages and steps else 0.0)
    if gold_modalities:
        gold_modality_coverage = len(step_types & gold_modalities) / max(1, len(gold_modalities))
    else:
        gold_modality_coverage = 1.0 if steps else 0.0

    visual_grounding_hit = 0.0
    if visual_required:
        visual_steps = [step for step in steps if clean_text(step.get("node_type")).lower() in VISUAL_MODALITIES]
        visual_grounding_hit = 1.0 if any(step_has_visual_grounding(step) for step in visual_steps) else 0.0
    else:
        visual_grounding_hit = 1.0 if steps else 0.0

    if route == "multihop_graph":
        cross_modal_hit = 1.0 if len(step_nodes) >= 2 else 0.0
    else:
        cross_modal_needed = visual_required or route in {"cross_modal", "structured_table", "visual_grounding"}
        cross_modal_hit = 1.0 if (not cross_modal_needed or len(step_types) >= 2) else 0.0
    relation_steps = [
        step
        for step in steps[1:]
        if clean_text(step.get("relation")) or clean_text(step.get("role")) in {"graph_neighbor", "visual_companion"}
    ]
    relation_support = len(relation_steps) / max(1, len(steps) - 1) if len(steps) > 1 else 0.0
    if 2 <= len(steps) <= 5:
        chain_length_score = 1.0
    elif len(steps) == 1:
        chain_length_score = 0.55
    elif len(steps) > 5:
        chain_length_score = 0.75
    else:
        chain_length_score = 0.0

    evidence_chain_score = (
        0.38 * gold_node_coverage
        + 0.14 * gold_page_hit
        + 0.16 * gold_modality_coverage
        + 0.14 * visual_grounding_hit
        + 0.10 * cross_modal_hit
        + 0.05 * relation_support
        + 0.03 * chain_length_score
    )
    return {
        "question_id": question.get("question_id", ""),
        "question_type": question.get("question_type", ""),
        "adaptive_route": route,
        "chain_present": int(bool(steps)),
        "step_count": len(steps),
        "unique_node_types": len(step_types),
        "gold_node_coverage": round(gold_node_coverage, 6),
        "gold_page_hit": round(gold_page_hit, 6),
        "gold_modality_coverage": round(gold_modality_coverage, 6),
        "visual_required": int(visual_required),
        "visual_grounding_hit": round(visual_grounding_hit, 6),
        "cross_modal_hit": round(cross_modal_hit, 6),
        "relation_support": round(relation_support, 6),
        "chain_length_score": round(chain_length_score, 6),
        "evidence_chain_score": round(evidence_chain_score, 6),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clean_text(row.get("adaptive_route")) or "all"].append(row)
        grouped["all"].append(row)

    summary_rows: list[dict[str, Any]] = []
    for route, group in sorted(grouped.items()):
        item: dict[str, Any] = {"route": route, "num_questions": len(group)}
        item["avg_step_count"] = round(sum(as_float(row.get("step_count")) for row in group) / max(1, len(group)), 6)
        for field in [
            "chain_present",
            "gold_node_coverage",
            "gold_page_hit",
            "gold_modality_coverage",
            "visual_grounding_hit",
            "cross_modal_hit",
            "relation_support",
            "evidence_chain_score",
        ]:
            item[field] = round(sum(as_float(row.get(field)) for row in group) / max(1, len(group)), 6)
        summary_rows.append(item)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated evidence chains.")
    parser.add_argument("--questions", default="data/questions.csv")
    parser.add_argument("--chains", default="outputs/evidence_chains/chains.jsonl")
    parser.add_argument("--per-question-output", default="outputs/evidence_chains/chain_eval_per_question.csv")
    parser.add_argument("--summary-output", default="outputs/evidence_chains/chain_eval_summary.csv")
    args = parser.parse_args()

    ensure_project_dirs()
    questions = {row.get("question_id", ""): row for row in read_csv(args.questions) if clean_text(row.get("question"))}
    chain_rows = read_jsonl(args.chains)

    per_question: list[dict[str, Any]] = []
    for chain in chain_rows:
        qid = clean_text(chain.get("question_id"))
        question = questions.get(qid)
        if not question:
            continue
        steps = chain.get("steps") if isinstance(chain.get("steps"), list) else []
        per_question.append(evaluate_chain(question, steps))

    write_csv(args.per_question_output, per_question, CHAIN_PER_QUESTION_FIELDS)
    write_csv(args.summary_output, summarize(per_question), CHAIN_SUMMARY_FIELDS)
    print(f"Wrote chain per-question metrics to {resolve_path(args.per_question_output)}")
    print(f"Wrote chain summary metrics to {resolve_path(args.summary_output)}")
    if not per_question:
        print("No chain metrics produced. Check --questions and --chains.")


if __name__ == "__main__":
    main()
