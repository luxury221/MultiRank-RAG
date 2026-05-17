from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline_common import (
    clean_text,
    ensure_project_dirs,
    preview,
    read_csv,
    read_jsonl,
    resolve_path,
    write_csv,
    write_jsonl,
)
from rerank_lib import adaptive_rag_route


RANKING_EXTRA_FIELDS = [
    "correction_source",
    "correction_reason",
    "verifier_score",
    "primary_verifier_score",
    "fallback_verifier_score",
    "primary_verifier_flags",
    "fallback_verifier_flags",
]

DECISION_FIELDS = [
    "question_id",
    "question_type",
    "adaptive_route",
    "selected_source",
    "primary_score",
    "fallback_score",
    "primary_flags",
    "fallback_flags",
    "reason",
    "primary_top1",
    "fallback_top1",
    "selected_top1",
]


VISUAL_TYPES = {"table", "figure", "caption"}


def group_rankings(rows: list[dict[str, str]], method: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if clean_text(row.get("method")) == method:
            grouped[clean_text(row.get("question_id"))].append(row)
    for qid in grouped:
        grouped[qid].sort(key=lambda row: int(float(row.get("rank") or 0)))
    return grouped


def group_chains(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = clean_text(row.get("question_id"))
        if qid:
            grouped[qid] = row
    return grouped


def wants_from_question(question: dict[str, Any]) -> dict[str, bool]:
    qtype = clean_text(question.get("question_type")).casefold()
    query = clean_text(question.get("question")).casefold()
    blob = f"{qtype} {query}"
    wants_table = "table" in blob or "tabular" in blob or "表" in blob
    wants_figure = any(term in blob for term in ("image", "figure", "fig.", "chart", "plot", "diagram", "图"))
    wants_cross = (
        "cross" in blob
        or "multi" in blob
        or "text-table-image" in qtype
        or (wants_table and wants_figure)
    )
    wants_visual = wants_table or wants_figure or wants_cross
    return {
        "table": wants_table,
        "figure": wants_figure,
        "cross": wants_cross,
        "visual": wants_visual,
    }


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        text = clean_text(value)
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def row_has_visual_grounding(row: dict[str, Any]) -> bool:
    return bool(
        clean_text(row.get("crop_image_path"))
        or clean_text(row.get("page_image_path"))
        or clean_text(row.get("visual_title"))
        or clean_text(row.get("qa_evidence"))
        or clean_text(row.get("visual_caption"))
        or clean_text(row.get("visual_summary"))
        or clean_text(row.get("has_visual_caption")) in {"1", "true", "yes"}
        or clean_text(row.get("has_visual_crop")) in {"1", "true", "yes"}
    )


def step_has_visual_grounding(step: dict[str, Any]) -> bool:
    return bool(
        clean_text(step.get("crop_image_path"))
        or clean_text(step.get("page_image_path"))
        or clean_text(step.get("bbox"))
        or clean_text(step.get("visual_caption"))
        or clean_text(step.get("visual_summary"))
    )


def route_diversity(rows: list[dict[str, str]]) -> float:
    routes: set[str] = set()
    for row in rows[:5]:
        for item in clean_text(row.get("source_routes")).split("|"):
            if item:
                routes.add(item)
    return min(1.0, len(routes) / 5.0)


def margin_score(rows: list[dict[str, str]]) -> float:
    if len(rows) < 2:
        return 0.2 if rows else 0.0
    margin = as_float(rows[0].get("score")) - as_float(rows[1].get("score"))
    return max(0.0, min(1.0, margin * 3.0))


def verifier(
    question: dict[str, Any],
    rows: list[dict[str, str]],
    chain: dict[str, Any] | None,
) -> dict[str, Any]:
    route = adaptive_rag_route(question)
    wants = wants_from_question(question)
    top5 = rows[:5]
    top1 = top5[0] if top5 else {}
    steps = chain.get("steps") if isinstance(chain, dict) and isinstance(chain.get("steps"), list) else []
    ranking_types = {clean_text(row.get("node_type")).lower() for row in top5 if clean_text(row.get("node_type"))}
    chain_types = {clean_text(step.get("node_type")).lower() for step in steps if clean_text(step.get("node_type"))}
    all_types = ranking_types | chain_types

    has_table = "table" in all_types
    has_figure = bool({"figure", "caption"} & all_types)
    has_text = "text" in all_types
    visual_rows = [row for row in top5 if clean_text(row.get("node_type")).lower() in VISUAL_TYPES]
    visual_steps = [step for step in steps if clean_text(step.get("node_type")).lower() in VISUAL_TYPES]
    grounded_visual = any(row_has_visual_grounding(row) for row in visual_rows) or any(
        step_has_visual_grounding(step) for step in visual_steps
    )

    required = 0
    satisfied = 0
    if wants["table"]:
        required += 1
        satisfied += int(has_table)
    if wants["figure"]:
        required += 1
        satisfied += int(has_figure and grounded_visual)
    if wants["cross"]:
        required += 1
        satisfied += int(has_text and (has_table or has_figure) and (not wants["figure"] or grounded_visual))
    if not wants["visual"]:
        required += 1
        satisfied += int(has_text)

    requirement_score = satisfied / max(1, required)
    step_score = min(1.0, len(steps) / 4.0) if steps else min(0.5, len(top5) / 10.0)
    relation_steps = [
        step
        for step in steps[1:]
        if clean_text(step.get("relation")) or clean_text(step.get("role")) in {"graph_neighbor", "visual_companion"}
    ]
    relation_score = len(relation_steps) / max(1, len(steps) - 1) if len(steps) > 1 else 0.0
    top1_type = clean_text(top1.get("node_type")).lower()
    top1_type_match = 0.0
    if wants["table"] and top1_type == "table":
        top1_type_match = 1.0
    elif wants["figure"] and top1_type in {"figure", "caption"}:
        top1_type_match = 1.0
    elif not wants["visual"] and top1_type == "text":
        top1_type_match = 1.0
    elif top1_type == "text":
        top1_type_match = 0.65
    elif top1_type in VISUAL_TYPES and wants["visual"]:
        top1_type_match = 0.72

    semantic_score = 0.55 * as_float(top1.get("sim_score")) + 0.25 * as_float(top1.get("chain_score")) + 0.20 * as_float(
        top1.get("visual_score")
    )
    semantic_score = max(0.0, min(1.0, semantic_score))
    diversity = route_diversity(rows)
    margin = margin_score(rows)
    score = (
        0.34 * requirement_score
        + 0.22 * semantic_score
        + 0.14 * step_score
        + 0.10 * relation_score
        + 0.08 * top1_type_match
        + 0.07 * diversity
        + 0.05 * margin
    )

    flags: list[str] = []
    if not rows:
        flags.append("no_ranking")
    if not steps:
        flags.append("no_chain")
    if wants["table"] and not has_table:
        flags.append("missing_table")
    if wants["figure"] and not has_figure:
        flags.append("missing_figure")
    if wants["figure"] and has_figure and not grounded_visual:
        flags.append("ungrounded_figure")
    if wants["cross"] and not (has_text and (has_table or has_figure)):
        flags.append("missing_cross_modal_pair")
    if not wants["visual"] and top1_type in VISUAL_TYPES:
        flags.append("visual_top1_for_text_query")
    if semantic_score < 0.42:
        flags.append("low_semantic_confidence")
    if requirement_score < 1.0:
        flags.append("requirement_not_satisfied")

    severe = {
        "no_ranking",
        "no_chain",
        "missing_table",
        "missing_figure",
        "ungrounded_figure",
        "missing_cross_modal_pair",
        "visual_top1_for_text_query",
    }
    needs_correction = bool(severe & set(flags)) or score < 0.58
    return {
        "route": route,
        "score": round(score, 6),
        "flags": flags,
        "needs_correction": needs_correction,
        "top1": clean_text(top1.get("node_id")),
        "top1_type": top1_type,
        "requirement_score": round(requirement_score, 6),
        "semantic_score": round(semantic_score, 6),
        "step_score": round(step_score, 6),
    }


def severe_flags(flags: set[str]) -> set[str]:
    severe = {
        "no_ranking",
        "no_chain",
        "missing_table",
        "missing_figure",
        "ungrounded_figure",
        "missing_cross_modal_pair",
        "visual_top1_for_text_query",
    }
    return flags & severe


def choose_source(primary: dict[str, Any], fallback: dict[str, Any], min_gain: float) -> tuple[str, str]:
    primary_score = as_float(primary.get("score"))
    fallback_score = as_float(fallback.get("score"))
    primary_flags = set(primary.get("flags") or [])
    fallback_flags = set(fallback.get("flags") or [])
    primary_severe = severe_flags(primary_flags)
    fallback_severe = severe_flags(fallback_flags)
    fixed_severe = bool(primary_severe) and len(fallback_severe) < len(primary_severe)

    if "visual_top1_for_text_query" in primary_severe and not fallback_severe and fallback_score >= primary_score - 0.02:
        return "fallback", "fallback fixes severe verifier flags"
    if primary_score < 0.52 and fixed_severe and fallback_score >= primary_score + min_gain:
        return "fallback", "fallback verifier score is higher"
    if fixed_severe and fallback_score >= primary_score - 0.04:
        return "merge", "primary kept; fallback supplies missing evidence"
    if primary_score < 0.50 and fallback_score >= primary_score + 0.02:
        return "fallback", "primary verifier score is low"
    return "primary", "primary verifier is sufficient"


def needed_visual_types(flags: list[str]) -> set[str]:
    needed: set[str] = set()
    if "missing_table" in flags:
        needed.add("table")
    if "missing_figure" in flags or "ungrounded_figure" in flags:
        needed.update({"figure", "caption"})
    if "missing_cross_modal_pair" in flags:
        needed.update({"text", "table", "figure", "caption"})
    return needed


def merge_chains(
    primary_chain: dict[str, Any] | None,
    fallback_chain: dict[str, Any] | None,
    primary_v: dict[str, Any],
    max_steps: int = 5,
) -> dict[str, Any] | None:
    if not primary_chain:
        return fallback_chain
    if not fallback_chain:
        return primary_chain
    primary_steps = primary_chain.get("steps") if isinstance(primary_chain.get("steps"), list) else []
    fallback_steps = fallback_chain.get("steps") if isinstance(fallback_chain.get("steps"), list) else []
    if not primary_steps:
        return fallback_chain

    needed = needed_visual_types(primary_v.get("flags") or [])
    seen = {clean_text(step.get("node_id")) for step in primary_steps if clean_text(step.get("node_id"))}
    merged = [dict(step) for step in primary_steps]

    additions: list[dict[str, Any]] = []
    for step in fallback_steps:
        node_id = clean_text(step.get("node_id"))
        node_type = clean_text(step.get("node_type")).lower()
        if not node_id or node_id in seen:
            continue
        if needed and node_type not in needed:
            continue
        if node_type in {"figure", "caption"} and not step_has_visual_grounding(step):
            continue
        item = dict(step)
        item["role"] = "visual_companion" if node_type in VISUAL_TYPES else item.get("role", "context_text")
        item["relation"] = clean_text(item.get("relation")) or "self-correction evidence"
        item["reason"] = (
            "自我修正阶段从多路召回结果中补入该证据，用于补全第一轮证据链缺失的图表/跨模态信息。"
        )
        additions.append(item)
        seen.add(node_id)
        if len(additions) >= 2:
            break

    if not additions:
        return primary_chain

    # Preserve the primary main evidence; replace lower-priority trailing context first.
    keep = merged[:]
    while len(keep) + len(additions) > max_steps and len(keep) > 1:
        remove_index = next(
            (
                idx
                for idx in range(len(keep) - 1, 0, -1)
                if clean_text(keep[idx].get("role")) in {"context_text", "graph_neighbor"}
            ),
            len(keep) - 1,
        )
        keep.pop(remove_index)
    merged_steps = keep + additions
    for idx, step in enumerate(merged_steps, start=1):
        step["chain_step"] = idx

    chain = dict(primary_chain)
    chain["steps"] = merged_steps
    roles = " -> ".join(clean_text(step.get("role")) for step in merged_steps)
    pages = sorted({clean_text(step.get("page")) for step in merged_steps if clean_text(step.get("page"))})
    chain["summary"] = f"{chain.get('question_id', '')} 自我修正证据链: {roles}; 涉及页码: {', '.join(pages)}。"
    return chain


def annotate_rankings(
    rows: list[dict[str, str]],
    source: str,
    reason: str,
    primary_v: dict[str, Any],
    fallback_v: dict[str, Any],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    verifier_score = max(primary_v["score"], fallback_v["score"]) if source == "merge" else (
        primary_v["score"] if source == "primary" else fallback_v["score"]
    )
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["correction_source"] = source
        item["correction_reason"] = reason
        item["verifier_score"] = verifier_score
        item["primary_verifier_score"] = primary_v["score"]
        item["fallback_verifier_score"] = fallback_v["score"]
        item["primary_verifier_flags"] = "|".join(primary_v["flags"])
        item["fallback_verifier_flags"] = "|".join(fallback_v["flags"])
        annotated.append(item)
    return annotated


def annotate_chain(chain: dict[str, Any] | None, source: str, reason: str, primary_v: dict[str, Any], fallback_v: dict[str, Any]) -> dict[str, Any]:
    if not chain:
        return {}
    item = dict(chain)
    item["correction_source"] = source
    item["correction_reason"] = reason
    item["verifier_score"] = max(primary_v["score"], fallback_v["score"]) if source == "merge" else (
        primary_v["score"] if source == "primary" else fallback_v["score"]
    )
    item["primary_verifier_score"] = primary_v["score"]
    item["fallback_verifier_score"] = fallback_v["score"]
    item["primary_verifier_flags"] = primary_v["flags"]
    item["fallback_verifier_flags"] = fallback_v["flags"]
    return item


def write_decision_report(path: str | Path, decisions: list[dict[str, Any]]) -> None:
    write_csv(path, decisions, DECISION_FIELDS)


def flatten_chain_steps(chains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chain in chains:
        steps = chain.get("steps") if isinstance(chain.get("steps"), list) else []
        for step in steps:
            rows.append(dict(step))
    return rows


def write_chain_markdown(path: str | Path, chains: list[dict[str, Any]]) -> None:
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Self-Corrected Evidence Chains\n\n")
        for chain in chains:
            f.write(f"## {chain.get('question_id', '')} | {chain.get('question_type', '')}\n\n")
            f.write(f"**Question**: {chain.get('question', '')}\n\n")
            f.write(f"**Correction**: {chain.get('correction_source', '')} - {chain.get('correction_reason', '')}\n\n")
            if clean_text(chain.get("summary")):
                f.write(f"**Summary**: {chain.get('summary')}\n\n")
            steps = chain.get("steps") if isinstance(chain.get("steps"), list) else []
            for step in steps:
                f.write(
                    f"{step.get('chain_step', '')}. `{step.get('node_id', '')}` "
                    f"({step.get('node_type', '')}, p.{step.get('page', '')}) - {step.get('role', '')}\n\n"
                )
                f.write(f"   - relation: {step.get('relation', '')}\n")
                f.write(f"   - reason: {step.get('reason', '')}\n")
                if clean_text(step.get("visual_summary")):
                    f.write(f"   - visual: {preview(step.get('visual_summary'), 220)}\n")
                f.write(f"   - evidence: {preview(step.get('content_preview'), 220)}\n\n")


def output_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    for extra in RANKING_EXTRA_FIELDS:
        if extra not in fields:
            fields.append(extra)
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence-aware self-correction over two completed RAG runs.")
    parser.add_argument("--questions", default="data/questions.csv")
    parser.add_argument("--primary-dir", required=True, help="Primary V4 directory with reranked.csv and evidence_chains.")
    parser.add_argument("--fallback-dir", required=True, help="Fallback V4 directory with reranked.csv and evidence_chains.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="G4")
    parser.add_argument("--min-gain", type=float, default=0.06)
    args = parser.parse_args()

    ensure_project_dirs()
    questions = [row for row in read_csv(args.questions) if clean_text(row.get("question_id"))]
    primary_dir = resolve_path(args.primary_dir)
    fallback_dir = resolve_path(args.fallback_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_rankings = group_rankings(read_csv(primary_dir / "reranked.csv"), args.method)
    fallback_rankings = group_rankings(read_csv(fallback_dir / "reranked.csv"), args.method)
    primary_chains = group_chains(read_jsonl(primary_dir / "evidence_chains/chains.jsonl"))
    fallback_chains = group_chains(read_jsonl(fallback_dir / "evidence_chains/chains.jsonl"))

    selected_rankings: list[dict[str, Any]] = []
    selected_chains: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()

    for question in questions:
        qid = clean_text(question.get("question_id"))
        primary_rows = primary_rankings.get(qid, [])
        fallback_rows = fallback_rankings.get(qid, [])
        primary_chain = primary_chains.get(qid)
        fallback_chain = fallback_chains.get(qid)
        primary_v = verifier(question, primary_rows, primary_chain)
        fallback_v = verifier(question, fallback_rows, fallback_chain)
        source, reason = choose_source(primary_v, fallback_v, args.min_gain)
        if source == "fallback":
            rows = fallback_rows
            chain = fallback_chain
        elif source == "merge":
            rows = primary_rows
            chain = merge_chains(primary_chain, fallback_chain, primary_v)
        else:
            rows = primary_rows
            chain = primary_chain
        selected_rankings.extend(annotate_rankings(rows, source, reason, primary_v, fallback_v))
        selected_chain = annotate_chain(chain, source, reason, primary_v, fallback_v)
        if selected_chain:
            selected_chains.append(selected_chain)
        source_counts[source] += 1
        flag_counts.update(primary_v["flags"])
        decisions.append(
            {
                "question_id": qid,
                "question_type": question.get("question_type", ""),
                "adaptive_route": primary_v["route"],
                "selected_source": source,
                "primary_score": primary_v["score"],
                "fallback_score": fallback_v["score"],
                "primary_flags": "|".join(primary_v["flags"]),
                "fallback_flags": "|".join(fallback_v["flags"]),
                "reason": reason,
                "primary_top1": primary_v["top1"],
                "fallback_top1": fallback_v["top1"],
                "selected_top1": (fallback_v if source == "fallback" else primary_v)["top1"],
            }
        )

    write_csv(output_dir / "reranked.csv", selected_rankings, output_fields(selected_rankings))
    chain_dir = output_dir / "evidence_chains"
    write_jsonl(chain_dir / "chains.jsonl", selected_chains)
    chain_steps = flatten_chain_steps(selected_chains)
    write_csv(chain_dir / "chain_steps.csv", chain_steps, output_fields(chain_steps) if chain_steps else None)
    write_chain_markdown(chain_dir / "evidence_chains.md", selected_chains)
    write_decision_report(output_dir / "self_correction_decisions.csv", decisions)
    summary = [
        {"metric": "questions", "value": len(questions)},
        {"metric": "selected_primary", "value": source_counts.get("primary", 0)},
        {"metric": "selected_merge", "value": source_counts.get("merge", 0)},
        {"metric": "selected_fallback", "value": source_counts.get("fallback", 0)},
    ]
    summary.extend({"metric": f"primary_flag:{flag}", "value": count} for flag, count in flag_counts.most_common())
    write_csv(output_dir / "self_correction_summary.csv", summary, ["metric", "value"])
    print(f"Wrote corrected rankings to {output_dir / 'reranked.csv'}")
    print(f"Wrote corrected chains to {chain_dir / 'chains.jsonl'}")
    print(f"Wrote decisions to {output_dir / 'self_correction_decisions.csv'}")
    print(
        f"Selected primary={source_counts.get('primary', 0)} "
        f"merge={source_counts.get('merge', 0)} fallback={source_counts.get('fallback', 0)}"
    )


if __name__ == "__main__":
    main()
