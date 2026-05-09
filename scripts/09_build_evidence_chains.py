from __future__ import annotations

import argparse
import re
from collections import defaultdict
from typing import Any

from pipeline_common import (
    as_float,
    clean_text,
    ensure_project_dirs,
    preview,
    read_csv,
    read_jsonl,
    resolve_path,
    split_multi,
    write_csv,
    write_jsonl,
)
from rerank_lib import build_graph, extract_document_refs, _looks_like_toc_entry


CHAIN_FIELDS = [
    "question_id",
    "doc_id",
    "question_type",
    "question",
    "chain_step",
    "role",
    "node_id",
    "node_type",
    "page",
    "relation",
    "score",
    "sim_score",
    "bridge_score",
    "ref_score",
    "visual_score",
    "source_ref",
    "page_image_path",
    "crop_image_path",
    "bbox",
    "bbox_source",
    "visual_summary",
    "visual_caption",
    "reason",
    "content_preview",
]


ROLE_LABELS = {
    "main_evidence": "主证据",
    "explicit_reference": "显式编号证据",
    "table_or_figure": "图表节点",
    "caption": "图注/表题",
    "context_text": "上下文解释",
    "graph_neighbor": "图关系补充",
    "visual_companion": "视觉伴随证据",
}


EDGE_PRIORITY = {
    "text_ref_table": 1.0,
    "text_ref_figure": 1.0,
    "table_caption": 0.95,
    "figure_caption": 0.95,
    "section_title": 0.45,
    "same_section": 0.35,
    "parent_section": 0.48,
    "chunk_sequence": 0.18,
    "same_page": 0.25,
    "belongs_to_page": 0.15,
}

VISUAL_NODE_TYPES = {"table", "figure", "caption"}
VISUAL_QUERY_TERMS = (
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
)


def group_by_question_method(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("question_id", ""), row.get("method", ""))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: int(row.get("rank") or 0))
    return grouped


def node_score(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    return (
        as_float(row.get("score")) * 0.55
        + as_float(row.get("ref_score")) * 0.25
        + as_float(row.get("bridge_score")) * 0.2
    )


def question_wants_visual(question: dict[str, str]) -> bool:
    blob = clean_text(
        f"{question.get('question_type', '')} {question.get('question', '')} {question.get('gold_modalities', '')}"
    ).lower()
    return any(term in blob for term in VISUAL_QUERY_TERMS)


def chain_has_visual_node(steps: list[dict[str, Any]]) -> bool:
    return any(clean_text(step.get("node_type")) in VISUAL_NODE_TYPES for step in steps)


def node_has_qwen_visual(node: dict[str, Any]) -> bool:
    return bool(clean_text(node.get("visual_caption")) or clean_text(node.get("qa_evidence")))


def visual_node_score(
    node_id: str,
    node: dict[str, Any],
    row: dict[str, str] | None,
    seed_pages: set[str],
    preferred_pages: set[str],
    relation_bonus: float,
) -> float:
    node_type = clean_text(node.get("node_type"))
    page = clean_text(node.get("page"))
    score = 0.0
    score += {"table": 0.35, "figure": 0.35, "caption": 0.28}.get(node_type, 0.0)
    score += 0.25 if clean_text(node.get("crop_image_path")) else 0.0
    score += 0.25 if node_has_qwen_visual(node) else 0.0
    score += 0.18 if page in preferred_pages else 0.0
    score += 0.12 if page in seed_pages else 0.0
    score += relation_bonus
    if row:
        score += 0.25 * as_float(row.get("visual_score"))
        score += 0.15 * as_float(row.get("score"))
        rank = int(float(row.get("rank") or 99))
        score += max(0.0, 0.12 * (1.0 - (rank - 1) / 10.0))
    return score


def visual_completion_candidates(
    question: dict[str, str],
    steps: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    graph,
    ranking_by_node: dict[str, dict[str, str]],
    preferred_pages: set[str],
    question_refs: set[tuple[str, str]],
) -> list[tuple[float, str, str, str]]:
    if not steps:
        return []
    doc_id = clean_text(question.get("doc_id"))
    seed_ids = [clean_text(step.get("node_id")) for step in steps if clean_text(step.get("node_id"))]
    seed_pages = {clean_text(step.get("page")) for step in steps if clean_text(step.get("page"))}
    target_pages = preferred_pages or seed_pages
    candidates: dict[str, tuple[float, str, str]] = {}

    def add_candidate(node_id: str, relation: str, reason: str, relation_bonus: float) -> None:
        node = nodes_by_id.get(node_id, {})
        if not node or _looks_like_toc_entry(node.get("content", "")):
            return
        if clean_text(node.get("node_type")) not in VISUAL_NODE_TYPES:
            return
        if doc_id and clean_text(node.get("doc_id")) != doc_id:
            return
        if question_refs and preferred_pages:
            page = clean_text(node.get("page"))
            if page not in preferred_pages and node_refs_conflict(node, question_refs):
                return
        score = visual_node_score(
            node_id,
            node,
            ranking_by_node.get(node_id),
            seed_pages,
            preferred_pages,
            relation_bonus,
        )
        previous = candidates.get(node_id)
        if not previous or score > previous[0]:
            candidates[node_id] = (score, relation, reason)

    for seed_id in seed_ids[:3]:
        if seed_id not in graph:
            continue
        for neighbor in graph.neighbors(seed_id):
            edge_data = graph.get_edge_data(seed_id, neighbor, default={})
            edge_types = list(edge_data.get("edge_types") or [edge_data.get("edge_type", "related")])
            relation = relation_label(edge_types)
            edge_bonus = max(EDGE_PRIORITY.get(edge_type, 0.1) for edge_type in edge_types)
            add_candidate(
                neighbor,
                relation or "\u56fe\u7ed3\u6784\u89c6\u89c9\u8865\u5168",
                "\u8be5\u89c6\u89c9\u8282\u70b9\u4e0e\u4e3b\u8bc1\u636e\u901a\u8fc7\u56fe\u5173\u7cfb\u76f8\u8fde\uff0c\u7528\u4e8e\u8865\u5168\u8868\u683c\u3001\u56fe\u50cf\u6216\u56fe\u6ce8\u8bc1\u636e\u3002",
                edge_bonus,
            )

    for node_id, node in nodes_by_id.items():
        if doc_id and clean_text(node.get("doc_id")) != doc_id:
            continue
        if target_pages and clean_text(node.get("page")) not in target_pages:
            continue
        add_candidate(
            node_id,
            "\u540c\u9875\u89c6\u89c9\u8865\u5168",
            "\u8be5\u89c6\u89c9\u8282\u70b9\u4e0e\u4e3b\u8bc1\u636e\u5904\u5728\u540c\u4e00\u9875\uff0c\u63d0\u4f9b\u53ef\u89c6\u5316\u88c1\u526a\u548c Qwen \u89c6\u89c9\u6458\u8981\u3002",
            0.08,
        )

    ranked = [
        (score, node_id, relation, reason)
        for node_id, (score, relation, reason) in candidates.items()
        if node_id not in {clean_text(step.get("node_id")) for step in steps}
    ]
    ranked.sort(reverse=True)
    return ranked


def node_ref_hits(node: dict[str, Any], question_refs: set[tuple[str, str]]) -> set[tuple[str, str]]:
    if not question_refs or _looks_like_toc_entry(node.get("content", "")):
        return set()
    text = f"{node.get('content', '')} {node.get('source_ref', '')}"
    return extract_document_refs(text) & question_refs


def node_refs_conflict(node: dict[str, Any], question_refs: set[tuple[str, str]]) -> bool:
    if not question_refs or _looks_like_toc_entry(node.get("content", "")):
        return False
    text = f"{node.get('content', '')} {node.get('source_ref', '')}"
    refs = extract_document_refs(text)
    question_kinds = {kind for kind, _ in question_refs}
    same_kind_refs = {ref for ref in refs if ref[0] in question_kinds}
    return bool(same_kind_refs and not same_kind_refs & question_refs)


def _ref_no_pattern(ref_no: str) -> str:
    ref_no = clean_text(ref_no).lower()
    if ref_no.startswith("s") and len(ref_no) > 1:
        return rf"s\s*{re.escape(ref_no[1:])}"
    return re.escape(ref_no)


def has_ref_label(node: dict[str, Any], refs: set[tuple[str, str]]) -> bool:
    if not refs:
        return False
    text = clean_text(node.get("content", "")).lower()
    for kind, ref_no in refs:
        no_pattern = _ref_no_pattern(ref_no)
        if kind == "table":
            if re.search(rf"(?:\btable|表)\s*{no_pattern}\s*[:：.\-]", text, re.I):
                return True
        elif re.search(rf"(?:\bfig\.?|\bfigure|图|圖)\s*{no_pattern}\s*[:：.\-]", text, re.I):
            return True
    return False


def preferred_reference_pages(
    question_refs: set[tuple[str, str]],
    ranking_rows: list[dict[str, str]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> set[str]:
    if not question_refs:
        return set()
    exact_rows = [
        row
        for row in ranking_rows
        if row.get("node_id") in nodes_by_id and node_ref_hits(nodes_by_id[row["node_id"]], question_refs)
    ]
    label_rows = [row for row in exact_rows if has_ref_label(nodes_by_id[row["node_id"]], question_refs)]
    anchor_rows = label_rows or exact_rows[:1]
    return {
        clean_text(nodes_by_id[row["node_id"]].get("page"))
        for row in anchor_rows
        if clean_text(nodes_by_id[row["node_id"]].get("page"))
    }


def add_step(
    steps: list[dict[str, Any]],
    seen: set[str],
    question: dict[str, str],
    node: dict[str, Any],
    row: dict[str, Any] | None,
    role: str,
    relation: str,
    reason: str,
) -> None:
    node_id = clean_text(node.get("node_id"))
    if not node_id or node_id in seen:
        return
    if _looks_like_toc_entry(node.get("content", "")):
        return
    seen.add(node_id)
    steps.append(
        {
            "question_id": question.get("question_id", ""),
            "doc_id": question.get("doc_id", ""),
            "question_type": question.get("question_type", ""),
            "question": question.get("question", ""),
            "chain_step": len(steps) + 1,
            "role": role,
            "node_id": node_id,
            "node_type": node.get("node_type", ""),
            "page": node.get("page", ""),
            "relation": relation,
            "score": round(as_float(row.get("score")) if row else node_score(row), 6),
            "sim_score": round(as_float(row.get("sim_score")) if row else 0.0, 6),
            "bridge_score": round(as_float(row.get("bridge_score")) if row else 0.0, 6),
            "ref_score": round(as_float(row.get("ref_score")) if row else 0.0, 6),
            "visual_score": round(as_float(row.get("visual_score")) if row else 0.0, 6),
            "source_ref": node.get("source_ref", ""),
            "page_image_path": node.get("page_image_path", ""),
            "crop_image_path": node.get("crop_image_path", ""),
            "bbox": node.get("bbox", ""),
            "bbox_source": node.get("bbox_source", ""),
            "visual_summary": node.get("visual_summary", ""),
            "visual_caption": node.get("visual_caption", ""),
            "reason": reason,
            "content_preview": preview(node.get("content", ""), 220),
        }
    )


def role_for_node_type(node_type: str) -> str:
    if node_type in {"table", "figure"}:
        return "table_or_figure"
    if node_type == "caption":
        return "caption"
    return "context_text"


def relation_label(edge_types: list[str]) -> str:
    labels = {
        "text_ref_table": "正文引用表格",
        "text_ref_figure": "正文引用图片",
        "table_caption": "表格-表题",
        "figure_caption": "图片-图注",
        "section_title": "章节标题",
        "same_section": "同章节补充",
        "parent_section": "层级父子",
        "chunk_sequence": "前后文补充",
        "same_page": "同页补充",
        "belongs_to_page": "页面包含",
    }
    return " / ".join(labels.get(edge_type, edge_type) for edge_type in edge_types)


def neighbor_candidates(
    node_id: str,
    nodes_by_id: dict[str, dict[str, Any]],
    graph,
    ranking_by_node: dict[str, dict[str, str]],
) -> list[tuple[float, str, list[str]]]:
    if node_id not in graph:
        return []
    items: list[tuple[float, str, list[str]]] = []
    for neighbor in graph.neighbors(node_id):
        node = nodes_by_id.get(neighbor, {})
        if not node or _looks_like_toc_entry(node.get("content", "")):
            continue
        edge_data = graph.get_edge_data(node_id, neighbor, default={})
        edge_types = list(edge_data.get("edge_types") or [edge_data.get("edge_type", "related")])
        edge_weight = max(EDGE_PRIORITY.get(edge_type, 0.1) for edge_type in edge_types)
        rank_bonus = 0.0
        if neighbor in ranking_by_node:
            rank_bonus = max(0.0, 1.0 - (int(ranking_by_node[neighbor].get("rank") or 99) - 1) / 10.0)
        type_bonus = {"table": 0.25, "figure": 0.25, "caption": 0.2, "text": 0.1}.get(
            clean_text(node.get("node_type")),
            0.0,
        )
        items.append((edge_weight + rank_bonus + type_bonus, neighbor, edge_types))
    items.sort(reverse=True)
    return items


def build_chain_for_question(
    question: dict[str, str],
    ranking_rows: list[dict[str, str]],
    nodes_by_id: dict[str, dict[str, Any]],
    graph,
    max_steps: int,
) -> list[dict[str, Any]]:
    ranking_rows = [row for row in ranking_rows if row.get("node_id") in nodes_by_id]
    ranking_by_node = {row.get("node_id", ""): row for row in ranking_rows}
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()

    non_page = [
        row
        for row in ranking_rows
        if clean_text(nodes_by_id[row["node_id"]].get("node_type")) != "page"
        and not _looks_like_toc_entry(nodes_by_id[row["node_id"]].get("content", ""))
    ]
    main_row = non_page[0] if non_page else (ranking_rows[0] if ranking_rows else None)
    if main_row:
        node = nodes_by_id[main_row["node_id"]]
        add_step(
            steps,
            seen,
            question,
            node,
            main_row,
            "main_evidence",
            "G4 Top-1",
            "G4 综合相似度、图结构、编号引用与 Qwen 视觉证据后的首位证据。",
        )

    question_refs = extract_document_refs(question.get("question", ""))
    preferred_pages = preferred_reference_pages(question_refs, ranking_rows, nodes_by_id)
    wants_visual = question_wants_visual(question)
    ref_rows = sorted(
        [
            row
            for row in ranking_rows
            if as_float(row.get("ref_score")) > 0
            and not node_refs_conflict(nodes_by_id[row["node_id"]], question_refs)
        ],
        key=lambda row: (as_float(row.get("ref_score")), as_float(row.get("score"))),
        reverse=True,
    )
    if preferred_pages:
        focused_ref_rows = [
            row
            for row in ref_rows
            if clean_text(nodes_by_id[row["node_id"]].get("page")) in preferred_pages
            or has_ref_label(nodes_by_id[row["node_id"]], question_refs)
        ]
        if focused_ref_rows:
            ref_rows = focused_ref_rows
    for row in ref_rows:
        if len(steps) >= max_steps:
            break
        if wants_visual and not chain_has_visual_node(steps) and len(steps) >= max_steps - 1:
            break
        node = nodes_by_id[row["node_id"]]
        role = "explicit_reference" if question_refs else role_for_node_type(clean_text(node.get("node_type")))
        add_step(
            steps,
            seen,
            question,
            node,
            row,
            role,
            "Figure/Table 编号匹配",
            "问题中出现显式图表编号，该节点与编号或相邻图表区域匹配。",
        )

    if wants_visual and not chain_has_visual_node(steps) and len(steps) < max_steps:
        for _, node_id, relation, reason in visual_completion_candidates(
            question,
            steps,
            nodes_by_id,
            graph,
            ranking_by_node,
            preferred_pages,
            question_refs,
        )[:1]:
            if len(steps) >= max_steps:
                break
            node = nodes_by_id[node_id]
            add_step(
                steps,
                seen,
                question,
                node,
                ranking_by_node.get(node_id),
                "visual_companion",
                relation,
                reason,
            )

    modality_rows = [
        row
        for row in ranking_rows
        if clean_text(nodes_by_id[row["node_id"]].get("node_type")) in {"table", "figure", "caption"}
    ]
    if question_refs:
        modality_rows = [
            row
            for row in modality_rows
            if node_ref_hits(nodes_by_id[row["node_id"]], question_refs)
            or (
                clean_text(nodes_by_id[row["node_id"]].get("page")) in preferred_pages
                and not node_refs_conflict(nodes_by_id[row["node_id"]], question_refs)
            )
        ]
    modality_rows.sort(
        key=lambda row: (
            as_float(row.get("ref_score")),
            as_float(row.get("bridge_score")),
            as_float(row.get("score")),
        ),
        reverse=True,
    )
    for row in modality_rows:
        if len(steps) >= max_steps:
            break
        node = nodes_by_id[row["node_id"]]
        add_step(
            steps,
            seen,
            question,
            node,
            row,
            role_for_node_type(clean_text(node.get("node_type"))),
            "多模态节点补充",
            "该节点提供表格、图片或图注层面的直接证据。",
        )

    if wants_visual and len(steps) < max_steps:
        for _, node_id, relation, reason in visual_completion_candidates(
            question,
            steps,
            nodes_by_id,
            graph,
            ranking_by_node,
            preferred_pages,
            question_refs,
        )[:1]:
            if len(steps) >= max_steps or node_id in seen:
                break
            node = nodes_by_id[node_id]
            add_step(
                steps,
                seen,
                question,
                node,
                ranking_by_node.get(node_id),
                "visual_companion",
                relation,
                reason,
            )

    seed_ids = [step["node_id"] for step in steps[:2]]
    for seed_id in seed_ids:
        if len(steps) >= max_steps:
            break
        for _, neighbor_id, edge_types in neighbor_candidates(seed_id, nodes_by_id, graph, ranking_by_node):
            if len(steps) >= max_steps:
                break
            if wants_visual and not chain_has_visual_node(steps) and len(steps) >= max_steps - 1:
                break
            if neighbor_id in seen:
                continue
            node = nodes_by_id[neighbor_id]
            if question_refs:
                if preferred_pages and clean_text(node.get("page")) not in preferred_pages:
                    continue
                if node_refs_conflict(node, question_refs):
                    continue
            row = ranking_by_node.get(neighbor_id)
            add_step(
                steps,
                seen,
                question,
                node,
                row,
                "graph_neighbor",
                relation_label(edge_types),
                "该节点与前序证据存在图边关系，用于补全跨模态上下文。",
            )

    text_rows = [
        row
        for row in ranking_rows
        if clean_text(nodes_by_id[row["node_id"]].get("node_type")) == "text"
        and not _looks_like_toc_entry(nodes_by_id[row["node_id"]].get("content", ""))
        and not node_refs_conflict(nodes_by_id[row["node_id"]], question_refs)
    ]
    for row in text_rows:
        if len(steps) >= max_steps:
            break
        if wants_visual and not chain_has_visual_node(steps) and len(steps) >= max_steps - 1:
            break
        node = nodes_by_id[row["node_id"]]
        add_step(
            steps,
            seen,
            question,
            node,
            row,
            "context_text",
            "语义上下文",
            "该文本段提供回答问题所需的解释性上下文。",
        )

    return steps


def chain_summary(question: dict[str, str], steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "未能构建证据链。"
    role_names = " -> ".join(ROLE_LABELS.get(step["role"], step["role"]) for step in steps)
    pages = sorted({str(step.get("page")) for step in steps if clean_text(step.get("page"))})
    return f"{question.get('question_id')} 证据链: {role_names}; 涉及页码: {', '.join(pages)}。"


def write_markdown(path: str, questions: list[dict[str, str]], grouped_steps: dict[str, list[dict[str, Any]]]) -> None:
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# G4 Evidence Chains\n\n")
        for question in questions:
            qid = question.get("question_id", "")
            steps = grouped_steps.get(qid, [])
            f.write(f"## {qid} | {question.get('question_type', '')}\n\n")
            f.write(f"**问题**: {question.get('question', '')}\n\n")
            f.write(f"**链路摘要**: {chain_summary(question, steps)}\n\n")
            for step in steps:
                role = ROLE_LABELS.get(step["role"], step["role"])
                f.write(
                    f"{step['chain_step']}. **{role}** `{step['node_id']}` "
                    f"({step['node_type']}, p.{step['page']}) - {step['relation']}\n\n"
                )
                f.write(f"   - 原因: {step['reason']}\n")
                if clean_text(step.get("crop_image_path")):
                    f.write(f"   - 视觉裁剪: {step['crop_image_path']}\n")
                if clean_text(step.get("visual_summary")):
                    f.write(f"   - 视觉摘要: {preview(step['visual_summary'], 260)}\n")
                f.write(f"   - 证据: {step['content_preview']}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence chains from visual-enhanced reranking outputs.")
    parser.add_argument("--questions", default="data/questions.csv")
    parser.add_argument("--nodes", default="outputs/parsed/nodes.jsonl")
    parser.add_argument("--edges", default="outputs/parsed/edges.jsonl")
    parser.add_argument("--rankings", default="outputs/rankings/reranked.csv")
    parser.add_argument("--method", default="G4")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--output-jsonl", default="outputs/evidence_chains/chains.jsonl")
    parser.add_argument("--output-csv", default="outputs/evidence_chains/chain_steps.csv")
    parser.add_argument("--output-md", default="outputs/evidence_chains/evidence_chains.md")
    args = parser.parse_args()

    ensure_project_dirs()
    questions = [row for row in read_csv(args.questions) if clean_text(row.get("question"))]
    nodes = read_jsonl(args.nodes)
    edges = read_jsonl(args.edges)
    rankings = read_csv(args.rankings)
    nodes_by_id = {node.get("node_id", ""): node for node in nodes if node.get("node_id")}
    graph = build_graph(nodes, edges)
    ranking_groups = group_by_question_method(rankings)

    all_steps: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    grouped_steps: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        qid = question.get("question_id", "")
        steps = build_chain_for_question(
            question,
            ranking_groups.get((qid, args.method), []),
            nodes_by_id,
            graph,
            max_steps=args.max_steps,
        )
        grouped_steps[qid] = steps
        all_steps.extend(steps)
        chain_rows.append(
            {
                "question_id": qid,
                "doc_id": question.get("doc_id", ""),
                "question_type": question.get("question_type", ""),
                "question": question.get("question", ""),
                "answer": question.get("answer", ""),
                "gold_node_ids": ";".join(split_multi(question.get("gold_node_ids"))),
                "summary": chain_summary(question, steps),
                "steps": steps,
            }
        )

    write_jsonl(args.output_jsonl, chain_rows)
    write_csv(args.output_csv, all_steps, CHAIN_FIELDS)
    write_markdown(args.output_md, questions, grouped_steps)
    print(f"Wrote {len(chain_rows)} evidence chains to {resolve_path(args.output_jsonl)}")
    print(f"Wrote {len(all_steps)} evidence chain steps to {resolve_path(args.output_csv)}")
    print(f"Wrote markdown evidence chains to {resolve_path(args.output_md)}")


if __name__ == "__main__":
    main()
