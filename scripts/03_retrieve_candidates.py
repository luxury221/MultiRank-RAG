from __future__ import annotations

import argparse

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
from rerank_lib import retrieve_candidates


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
    "source_ref",
    "content_preview",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve Top-K candidate evidence nodes for each question.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS.relative_to(DEFAULT_QUESTIONS.parents[1])))
    parser.add_argument("--nodes", default=str(DEFAULT_NODES.relative_to(DEFAULT_NODES.parents[2])))
    parser.add_argument("--output", default=str(DEFAULT_CANDIDATES.relative_to(DEFAULT_CANDIDATES.parents[2])))
    parser.add_argument("--top-k", type=int, default=50, help="Candidate pool size before reranking.")
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
        candidates, scores = retrieve_candidates(
            question,
            nodes,
            top_k=args.top_k,
            retriever=args.retriever,
            embedding_index=embedding_index,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            hybrid_alpha=args.hybrid_alpha,
        )
        for rank, node in enumerate(candidates, start=1):
            rows.append(
                {
                    "question_id": question.get("question_id", ""),
                    "doc_id": question.get("doc_id", ""),
                    "question": question.get("question", ""),
                    "rank": rank,
                    "node_id": node.get("node_id", ""),
                    "node_type": node.get("node_type", ""),
                    "page": node.get("page", ""),
                    "score": round(scores.get(node.get("node_id", ""), 0.0), 6),
                    "retriever": args.retriever,
                    "embedding_model": args.embedding_model if args.retriever in {"embedding", "hybrid", "fusion"} else "",
                    "source_ref": node.get("source_ref", ""),
                    "content_preview": preview(node.get("content", "")),
                }
            )

    write_csv(args.output, rows, CANDIDATE_FIELDS)
    print(f"Wrote {len(rows)} candidate rows to {resolve_path(args.output)}")
    if not questions:
        print("No questions found. Fill data/questions.csv or pass --questions data/sample/questions.csv.")
    if not nodes:
        print("No nodes found. Run scripts/01_parse_pdf.py first or pass --nodes data/sample/nodes.jsonl.")


if __name__ == "__main__":
    main()
