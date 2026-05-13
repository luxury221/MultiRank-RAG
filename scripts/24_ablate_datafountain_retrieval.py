from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from embedding_index import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingIndex,
)
from pipeline_common import clean_text, read_csv, read_jsonl, resolve_path, write_csv
from rerank_lib import load_kg_index, rank_question, retrieve_candidates


DEFAULT_QUESTIONS = "outputs/after_sales_kb/questions.csv"
DEFAULT_NODES = "outputs/after_sales_kb/nodes.jsonl"
DEFAULT_EDGES = "outputs/after_sales_kb/edges.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/after_sales_kb/ablation"


SUMMARY_FIELDS = [
    "variant",
    "questions",
    "retriever",
    "kg",
    "embedding",
    "policy_questions",
    "policy_top1_policy_rate",
    "manual_visual_questions",
    "manual_visual_top5_visual_rate",
    "top5_visual_caption_rate",
    "avg_top1_score",
    "avg_top1_kg_score",
]


RANKING_FIELDS = [
    "variant",
    "question_id",
    "question",
    "rank",
    "node_id",
    "node_type",
    "score",
    "kg_score",
    "has_visual_caption",
    "source_ref",
    "content_preview",
]


SERVICE_TERMS = (
    "退货",
    "换货",
    "退款",
    "无理由",
    "运费",
    "发票",
    "开票",
    "物流",
    "包装",
    "破损",
    "售后",
    "维修",
    "保修",
    "人为损坏",
    "return",
    "refund",
    "exchange",
    "invoice",
    "warranty",
    "repair",
)

MANUAL_VISUAL_TERMS = (
    "如何",
    "怎么",
    "步骤",
    "安装",
    "拆卸",
    "更换",
    "清洁",
    "连接",
    "设置",
    "调节",
    "按钮",
    "接口",
    "指示灯",
    "图",
    "图片",
    "install",
    "remove",
    "replace",
    "clean",
    "connect",
    "setup",
    "button",
    "diagram",
)

VISUAL_TYPES = {"figure", "table", "caption"}


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    blob = clean_text(text).casefold()
    return any(term.casefold() in blob for term in terms)


def is_service_question(question: dict[str, Any]) -> bool:
    return contains_any(f"{question.get('question_type', '')} {question.get('question', '')}", SERVICE_TERMS)


def is_manual_visual_question(question: dict[str, Any]) -> bool:
    blob = f"{question.get('question_type', '')} {question.get('question', '')}"
    return not is_service_question(question) and contains_any(blob, MANUAL_VISUAL_TERMS)


def variant_specs(include_embedding: bool) -> list[dict[str, Any]]:
    specs = [
        {"variant": "lexical", "retriever": "lexical", "kg": False, "embedding": False},
        {"variant": "bm25", "retriever": "bm25", "kg": False, "embedding": False},
        {"variant": "fusion_no_kg", "retriever": "fusion", "kg": False, "embedding": False},
        {"variant": "fusion_kg", "retriever": "fusion", "kg": True, "embedding": False},
    ]
    if include_embedding:
        specs.extend(
            [
                {"variant": "embedding", "retriever": "embedding", "kg": False, "embedding": True},
                {"variant": "fusion_embedding_kg", "retriever": "fusion", "kg": True, "embedding": True},
            ]
        )
    return specs


def summarize_variant(variant: str, rows: list[dict[str, Any]], questions: list[dict[str, Any]]) -> dict[str, Any]:
    by_q: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_q.setdefault(clean_text(row.get("question_id")), []).append(row)
    for qrows in by_q.values():
        qrows.sort(key=lambda row: int(row.get("rank") or 0))

    policy_questions = [q for q in questions if is_service_question(q)]
    visual_questions = [q for q in questions if is_manual_visual_question(q)]
    policy_hits = 0
    visual_hits = 0
    top5_caption_hits = 0
    top1_scores: list[float] = []
    top1_kg_scores: list[float] = []

    for question in questions:
        qid = clean_text(question.get("question_id"))
        qrows = by_q.get(qid, [])
        if not qrows:
            continue
        top1 = qrows[0]
        try:
            top1_scores.append(float(top1.get("score") or 0.0))
            top1_kg_scores.append(float(top1.get("kg_score") or 0.0))
        except (TypeError, ValueError):
            pass
        if question in policy_questions and clean_text(top1.get("node_id")).startswith("AS_POLICY"):
            policy_hits += 1
        top5 = qrows[:5]
        if any(clean_text(row.get("has_visual_caption")) in {"1", "true", "True"} for row in top5):
            top5_caption_hits += 1
        if question in visual_questions and any(clean_text(row.get("node_type")) in VISUAL_TYPES for row in top5):
            visual_hits += 1

    return {
        "variant": variant,
        "questions": len(questions),
        "policy_questions": len(policy_questions),
        "policy_top1_policy_rate": round(policy_hits / max(1, len(policy_questions)), 6),
        "manual_visual_questions": len(visual_questions),
        "manual_visual_top5_visual_rate": round(visual_hits / max(1, len(visual_questions)), 6),
        "top5_visual_caption_rate": round(top5_caption_hits / max(1, len(questions)), 6),
        "avg_top1_score": round(sum(top1_scores) / max(1, len(top1_scores)), 6),
        "avg_top1_kg_score": round(sum(top1_kg_scores) / max(1, len(top1_kg_scores)), 6),
    }


def run_variant(
    spec: dict[str, Any],
    questions: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    top_k: int,
    candidate_k: int,
    kg_index: dict[str, Any],
    embedding_index: EmbeddingIndex | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    active_kg = kg_index if spec["kg"] else None
    active_embedding = embedding_index if spec["embedding"] else None
    for question in questions:
        candidates, _scores = retrieve_candidates(
            question,
            nodes,
            top_k=candidate_k,
            retriever=spec["retriever"],
            embedding_index=active_embedding,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            hybrid_alpha=args.hybrid_alpha,
            kg_index=active_kg,
        )
        candidate_rows = [{"node_id": node.get("node_id", ""), "rank": index} for index, node in enumerate(candidates, 1)]
        rank_nodes = nodes if args.full_rerank else candidates
        rank_edges = edges if args.full_rerank else []
        ranked = rank_question(
            question,
            rank_nodes,
            rank_edges,
            top_k=top_k,
            candidate_rows=candidate_rows,
            retriever=spec["retriever"],
            embedding_index=active_embedding,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            hybrid_alpha=args.hybrid_alpha,
            kg_index=active_kg,
        )
        for row in ranked:
            if row.get("method") != "G4":
                continue
            rows.append({"variant": spec["variant"], **row})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataFountain retrieval ablations for BM25/KG/embedding/fusion.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kg-dir", default="outputs/kg")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--node-limit", type=int, default=0, help="Optional quick-smoke cap on nodes.")
    parser.add_argument("--full-rerank", action="store_true", help="Use the full graph during G4 rerank. Slower.")
    parser.add_argument("--include-embedding", action="store_true")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache", default="outputs/embeddings")
    parser.add_argument("--embedding-device", default=DEFAULT_EMBEDDING_DEVICE)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--hybrid-alpha", type=float, default=0.7)
    args = parser.parse_args()

    questions = [row for row in read_csv(args.questions) if clean_text(row.get("question"))]
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
    nodes = read_jsonl(args.nodes)
    if args.node_limit and args.node_limit > 0:
        nodes = nodes[: args.node_limit]
    edges = read_jsonl(args.edges)
    kg_index = load_kg_index(args.kg_dir)
    embedding_index = (
        EmbeddingIndex.from_nodes(
            nodes,
            model_name=args.embedding_model,
            cache_dir=args.embedding_cache,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
        if args.include_embedding
        else None
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for spec in variant_specs(args.include_embedding):
        rows = run_variant(
            spec,
            questions,
            nodes,
            edges,
            args.top_k,
            args.candidate_k,
            kg_index,
            embedding_index,
            args,
        )
        write_csv(output_dir / f"{spec['variant']}_g4.csv", rows, RANKING_FIELDS)
        summary = summarize_variant(spec["variant"], rows, questions)
        summary.update({"retriever": spec["retriever"], "kg": int(spec["kg"]), "embedding": int(spec["embedding"])})
        summary_rows.append(summary)
        all_rows.extend(rows)
        print(f"{spec['variant']}: {summary}")

    write_csv(output_dir / "all_g4.csv", all_rows, RANKING_FIELDS)
    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    print(f"Wrote ablation outputs to {output_dir}")


if __name__ == "__main__":
    main()
