from __future__ import annotations

import argparse
from collections import defaultdict

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
    parser.add_argument("--retriever", choices=["fusion", "hybrid", "embedding", "lexical"], default="fusion")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache", default="outputs/embeddings")
    parser.add_argument("--embedding-device", default=DEFAULT_EMBEDDING_DEVICE)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--hybrid-alpha", type=float, default=0.7, help="Embedding weight for --retriever hybrid.")
    args = parser.parse_args()

    ensure_project_dirs()
    questions = [row for row in read_csv(args.questions) if clean_text(row.get("question"))]
    nodes = read_jsonl(args.nodes)
    edges = read_jsonl(args.edges)
    candidates_by_qid = group_candidates(read_csv(args.candidates))
    embedding_index = None
    if args.retriever in {"embedding", "hybrid", "fusion"} and nodes:
        embedding_index = EmbeddingIndex.from_nodes(
            nodes,
            model_name=args.embedding_model,
            cache_dir=args.embedding_cache,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )

    rows: list[dict[str, object]] = []
    for question in questions:
        qid = question.get("question_id", "")
        rows.extend(
            rank_question(
                question,
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
            )
        )

    write_csv(args.output, rows, RANKING_FIELDS)
    print(f"Wrote {len(rows)} ranking rows to {resolve_path(args.output)}")
    if not rows:
        print("No rankings produced. Check questions, nodes, and candidates.")


if __name__ == "__main__":
    main()
