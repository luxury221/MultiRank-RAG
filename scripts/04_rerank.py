from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

from embedding_index import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingIndex,
)
from pipeline_common import (
    DEFAULT_CANDIDATES,
    DEFAULT_EDGES,
    DEFAULT_NODES,
    DEFAULT_QUESTIONS,
    DEFAULT_RANKINGS,
    clean_text,
    ensure_project_dirs,
    read_csv,
    read_jsonl,
    resolve_path,
    write_csv,
)
from rerank_lib import rank_question
from rerank_lib import load_kg_index
from datafountain_query_expansion import expand_question, load_routes, submission_id


RANKING_FIELDS = [
    "question_id",
    "doc_id",
    "question",
    "method",
    "rank",
    "node_id",
    "node_type",
    "page",
    "score",
    "sim_score",
    "ppr_score",
    "bridge_score",
    "ref_score",
    "visual_score",
    "chain_score",
    "domain_score",
    "kg_score",
    "model_rerank_score",
    "adaptive_route",
    "rerank_profile",
    "source_routes",
    "route_ranks",
    "has_visual_crop",
    "has_visual_caption",
    "visual_title",
    "qa_evidence",
    "crop_image_path",
    "page_image_path",
    "source_ref",
    "content_preview",
    "rerank_time_ms",
]

ALL_RERANK_METHODS = ("G0", "G1", "G2", "G3", "G4")


def parse_methods(value: str) -> list[str]:
    methods = [item.strip().upper() for item in value.split(",") if item.strip()]
    invalid = [method for method in methods if method not in ALL_RERANK_METHODS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown rerank method(s): {', '.join(invalid)}. "
            f"Valid choices: {', '.join(ALL_RERANK_METHODS)}"
        )
    return list(dict.fromkeys(methods)) or list(ALL_RERANK_METHODS)


def group_candidates(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("question_id", "")].append(row)
    for question_id in grouped:
        grouped[question_id].sort(key=lambda row: int(row.get("rank") or 0))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G0-G4 evidence reranking.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS.relative_to(DEFAULT_QUESTIONS.parents[1])))
    parser.add_argument("--nodes", default=str(DEFAULT_NODES.relative_to(DEFAULT_NODES.parents[2])))
    parser.add_argument("--edges", default=str(DEFAULT_EDGES.relative_to(DEFAULT_EDGES.parents[2])))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES.relative_to(DEFAULT_CANDIDATES.parents[2])))
    parser.add_argument("--output", default=str(DEFAULT_RANKINGS.relative_to(DEFAULT_RANKINGS.parents[2])))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.93)
    parser.add_argument("--beta", type=float, default=0.07)
    parser.add_argument("--lambda-s", type=float, default=0.85)
    parser.add_argument("--lambda-p", type=float, default=0.0)
    parser.add_argument("--lambda-b", type=float, default=0.15)
    parser.add_argument("--lambda-r", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument(
        "--retriever",
        choices=["fusion", "multiroute", "multi_route", "multi", "hybrid", "embedding", "lexical", "bm25", "kg"],
        default="fusion",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache", default="outputs/embeddings")
    parser.add_argument("--embedding-device", default=DEFAULT_EMBEDDING_DEVICE)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--hybrid-alpha", type=float, default=0.7, help="Embedding weight for --retriever hybrid.")
    parser.add_argument(
        "--kg-dir",
        default=os.getenv("RAG_KG_DIR", "outputs/graphrag"),
        help="GraphRAG/KG directory. Empty string disables graph scoring.",
    )
    parser.add_argument("--routes", default="", help="Optional DataFountain question route CSV for product-aware query expansion.")
    parser.add_argument("--expand-query", action="store_true", help="Append product route aliases to reranking queries.")
    parser.add_argument(
        "--context-expansion",
        action="store_true",
        help="Keep same-context table/figure/text companions in the reranking pool.",
    )
    parser.add_argument(
        "--adaptive-rerank-boost",
        action="store_true",
        help="Use stronger query-modality-aware G4 scoring for visual/table/cross-modal questions.",
    )
    parser.add_argument(
        "--graph-context-boost",
        action="store_true",
        help="Increase graph/PPR/bridge contribution for GraphRAG evidence-chain experiments.",
    )
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=parse_methods(os.getenv("RAG_RERANK_METHODS", ",".join(ALL_RERANK_METHODS))),
        help="Comma-separated methods to write, e.g. G0 or G0,G4.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse complete question rows already present in output.")
    args = parser.parse_args()

    ensure_project_dirs()
    questions = [row for row in read_csv(args.questions) if clean_text(row.get("question"))]
    nodes = read_jsonl(args.nodes)
    edges = read_jsonl(args.edges)
    candidates_by_qid = group_candidates(read_csv(args.candidates))
    embedding_index = None
    if args.retriever in {"embedding", "hybrid", "fusion", "multiroute", "multi_route", "multi"} and nodes:
        embedding_index = EmbeddingIndex.from_nodes(
            nodes,
            model_name=args.embedding_model,
            cache_dir=args.embedding_cache,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
    kg_index = load_kg_index(args.kg_dir)
    routes = load_routes(args.routes) if args.expand_query and args.routes else {}

    rows: list[dict[str, object]] = []
    completed_qids: set[str] = set()
    if args.resume and resolve_path(args.output).exists():
        existing_rows = read_csv(args.output)
        selected_methods = set(args.methods)
        expected_rows_per_question = args.top_k * len(selected_methods)
        counts = Counter(
            row.get("question_id", "")
            for row in existing_rows
            if row.get("question_id") and row.get("method") in selected_methods
        )
        completed_qids = {qid for qid, count in counts.items() if count >= expected_rows_per_question}
        rows = [
            row
            for row in existing_rows
            if row.get("question_id", "") in completed_qids and row.get("method") in selected_methods
        ]
        print(
            f"Resuming from {resolve_path(args.output)}; "
            f"kept {len(rows)} rows for {len(completed_qids)} complete questions.",
            flush=True,
        )

    print(f"Writing rerank methods: {','.join(args.methods)}", flush=True)
    processed = 0
    for index, question in enumerate(questions, start=1):
        qid = question.get("question_id", "")
        if qid in completed_qids:
            continue
        query_question = expand_question(question, routes.get(submission_id(qid))) if routes else question
        question_rows = rank_question(
            query_question,
            nodes,
            edges,
            top_k=args.top_k,
            candidate_rows=candidates_by_qid.get(qid),
            alpha=args.alpha,
            beta=args.beta,
            lambda_s=args.lambda_s,
            lambda_p=args.lambda_p,
            lambda_b=args.lambda_b,
            lambda_r=args.lambda_r,
            tau=args.tau,
            retriever=args.retriever,
            embedding_index=embedding_index,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            hybrid_alpha=args.hybrid_alpha,
            kg_index=kg_index,
            context_expansion=args.context_expansion,
            adaptive_rerank_boost=args.adaptive_rerank_boost,
            graph_context_boost=args.graph_context_boost,
        )
        for row in question_rows:
            if row.get("method") not in args.methods:
                continue
            row["question"] = question.get("question", "")
            rows.append(row)
        processed += 1
        if processed == 1 or processed % 10 == 0 or index == len(questions):
            print(f"Reranked {index}/{len(questions)} questions; processed={processed}; rows={len(rows)}", flush=True)
        if processed % 5 == 0:
            write_csv(args.output, rows, RANKING_FIELDS)

    write_csv(args.output, rows, RANKING_FIELDS)
    print(f"Wrote {len(rows)} ranking rows to {resolve_path(args.output)}")
    if not rows:
        print("No rankings produced. Check questions, nodes, and candidates.")


if __name__ == "__main__":
    main()
