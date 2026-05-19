from __future__ import annotations

import argparse
import os
from collections import Counter

from embedding_index import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingIndex,
)
from pipeline_common import (
    DEFAULT_CANDIDATES,
    DEFAULT_NODES,
    DEFAULT_QUESTIONS,
    clean_text,
    ensure_project_dirs,
    preview,
    read_csv,
    read_jsonl,
    resolve_path,
    write_csv,
)
from rerank_lib import load_kg_index, retrieve_candidates
from query_expansion import expand_question, load_routes, submission_id


CANDIDATE_FIELDS = [
    "question_id",
    "doc_id",
    "question",
    "rank",
    "node_id",
    "node_type",
    "page",
    "score",
    "retriever",
    "embedding_model",
    "query_plan",
    "query_plan_strategy",
    "required_modalities",
    "source_routes",
    "route_ranks",
    "source_ref",
    "content_preview",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve Top-K candidate evidence nodes for each question.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS.relative_to(DEFAULT_QUESTIONS.parents[1])))
    parser.add_argument("--nodes", default=str(DEFAULT_NODES.relative_to(DEFAULT_NODES.parents[2])))
    parser.add_argument("--output", default=str(DEFAULT_CANDIDATES.relative_to(DEFAULT_CANDIDATES.parents[2])))
    parser.add_argument("--top-k", type=int, default=50, help="Candidate pool size before reranking.")
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
    parser.add_argument("--routes", default="", help="Optional question route CSV for product-aware query expansion.")
    parser.add_argument("--expand-query", action="store_true", help="Append product route aliases to retrieval queries.")
    parser.add_argument(
        "--context-expansion",
        action="store_true",
        help="Add same-section/page table, figure, and neighboring text companions to the candidate pool.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse complete question rows already present in output.")
    args = parser.parse_args()

    ensure_project_dirs()
    questions = [row for row in read_csv(args.questions) if clean_text(row.get("question"))]
    nodes = read_jsonl(args.nodes)
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
        counts = Counter(row.get("question_id", "") for row in existing_rows if row.get("question_id"))
        completed_qids = {qid for qid, count in counts.items() if count >= args.top_k}
        rows = [row for row in existing_rows if row.get("question_id", "") in completed_qids]
        print(
            f"Resuming from {resolve_path(args.output)}; "
            f"kept {len(rows)} rows for {len(completed_qids)} complete questions.",
            flush=True,
        )

    work_items: list[tuple[int, dict[str, str], dict[str, str]]] = []
    for index, question in enumerate(questions, start=1):
        qid = question.get("question_id", "")
        if qid in completed_qids:
            continue
        query_question = expand_question(question, routes.get(submission_id(qid))) if routes else question
        work_items.append((index, question, query_question))

    precomputed_embedding_scores: list[dict[str, float] | None] = [None] * len(work_items)
    if embedding_index is not None and work_items:
        query_texts = [query_question.get("question", "") for _, _, query_question in work_items]
        precomputed_embedding_scores = embedding_index.score_many(query_texts, nodes=nodes)
        print(f"Precomputed embedding query scores for {len(work_items)} questions.", flush=True)

    processed = 0
    for item_index, (index, question, query_question) in enumerate(work_items):
        candidates, scores = retrieve_candidates(
            query_question,
            nodes,
            top_k=args.top_k,
            retriever=args.retriever,
            embedding_index=embedding_index,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            hybrid_alpha=args.hybrid_alpha,
            kg_index=kg_index,
            precomputed_embedding_scores=precomputed_embedding_scores[item_index],
            context_expansion=args.context_expansion,
        )
        source_routes = query_question.get("_multiroute_source_routes", {}) if isinstance(query_question, dict) else {}
        route_ranks = query_question.get("_multiroute_route_ranks", {}) if isinstance(query_question, dict) else {}
        query_plan = query_question.get("_query_plan", {}) if isinstance(query_question, dict) else {}
        for rank, node in enumerate(candidates, start=1):
            node_id = node.get("node_id", "")
            rows.append(
                {
                    "question_id": question.get("question_id", ""),
                    "doc_id": question.get("doc_id", ""),
                    "question": question.get("question", ""),
                    "rank": rank,
                    "node_id": node_id,
                    "node_type": node.get("node_type", ""),
                    "page": node.get("page", ""),
                    "score": round(scores.get(node_id, 0.0), 6),
                    "retriever": args.retriever,
                    "embedding_model": (
                        args.embedding_model
                        if args.retriever in {"embedding", "hybrid", "fusion", "multiroute", "multi_route", "multi"}
                        else ""
                    ),
                    "query_plan": (
                        f"route={query_plan.get('route', '')};"
                        f"strategy={query_plan.get('strategy', '')};"
                        f"modalities={','.join(query_plan.get('required_modalities') or [])}"
                    )
                    if query_plan
                    else "",
                    "query_plan_strategy": query_plan.get("strategy", "") if query_plan else "",
                    "required_modalities": ",".join(query_plan.get("required_modalities") or []) if query_plan else "",
                    "source_routes": source_routes.get(node_id, ""),
                    "route_ranks": route_ranks.get(node_id, ""),
                    "source_ref": node.get("source_ref", ""),
                    "content_preview": preview(node.get("content", "")),
                }
            )
        processed += 1
        if processed == 1 or processed % 10 == 0 or index == len(questions):
            print(f"Retrieved {index}/{len(questions)} questions; processed={processed}; rows={len(rows)}", flush=True)
        if processed % 5 == 0:
            write_csv(args.output, rows, CANDIDATE_FIELDS)

    write_csv(args.output, rows, CANDIDATE_FIELDS)
    print(f"Wrote {len(rows)} candidate rows to {resolve_path(args.output)}")
    if not questions:
        print("No questions found. Fill data/questions.csv or pass --questions data/sample/questions.csv.")
    if not nodes:
        print("No nodes found. Run scripts/01_parse_pdf.py first or pass --nodes data/sample/nodes.jsonl.")


if __name__ == "__main__":
    main()
