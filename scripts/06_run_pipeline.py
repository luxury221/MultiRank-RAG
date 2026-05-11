from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from embedding_index import DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_EMBEDDING_DEVICE, DEFAULT_EMBEDDING_MODEL
from pipeline_common import ensure_project_dirs


ROOT = Path(__file__).resolve().parents[1]


def run_step(args: list[str]) -> None:
    print(">", " ".join(args))
    subprocess.run(args, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full parse/build/retrieve/rerank/evaluate pipeline.")
    parser.add_argument("--sample", action="store_true", help="Use data/sample instead of real PDFs.")
    parser.add_argument("--questions", default="data/questions.csv")
    parser.add_argument("--top-k", type=int, default=10, help="Backward-compatible alias for --rerank-k.")
    parser.add_argument("--candidate-k", type=int, default=50, help="Candidate pool size before reranking.")
    parser.add_argument("--rerank-k", type=int, default=None, help="Final rows kept per method after reranking.")
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument(
        "--parser",
        choices=["mineru", "native"],
        default=os.getenv("RAG_PDF_PARSER", "mineru"),
        help="PDF parser backend used by scripts/01_parse_pdf.py.",
    )
    parser.add_argument("--mineru-output-dir", default=os.getenv("RAG_MINERU_OUTPUT_DIR", "outputs/mineru"))
    parser.add_argument("--mineru-api-url", default=os.getenv("MINERU_API_URL", ""))
    parser.add_argument("--mineru-backend", default=os.getenv("MINERU_BACKEND", "pipeline"))
    parser.add_argument("--mineru-method", default=os.getenv("MINERU_METHOD", "auto"))
    parser.add_argument("--mineru-lang", default=os.getenv("MINERU_LANG", ""))
    parser.add_argument(
        "--chunk-template",
        choices=["auto", "general", "ai", "math", "finance", "medical"],
        default="auto",
        help="Paper-aware parser template used by scripts/01_parse_pdf.py.",
    )
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--visual-dpi", type=int, default=120)
    parser.add_argument("--visual-caption-provider", choices=["local", "qwen"], default="local")
    parser.add_argument("--visual-caption-model", default="")
    parser.add_argument("--visual-caption-device", default="auto")
    parser.add_argument("--visual-max-captions", type=int, default=0)
    parser.add_argument("--qwen-model", default="qwen-vl-plus")
    parser.add_argument("--qwen-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--qwen-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument(
        "--candidate-retriever",
        choices=["fusion", "hybrid", "embedding", "lexical"],
        default="fusion",
        help="Retriever used for G0 candidate generation.",
    )
    parser.add_argument(
        "--rerank-retriever",
        choices=["fusion", "hybrid", "embedding", "lexical"],
        default="fusion",
        help="Retriever used for G1-G4 similarity scores.",
    )
    parser.add_argument(
        "--retriever",
        choices=["fusion", "hybrid", "embedding", "lexical"],
        default="",
        help="Backward-compatible shortcut that sets both candidate and rerank retrievers.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache", default="outputs/embeddings")
    parser.add_argument("--embedding-device", default=DEFAULT_EMBEDDING_DEVICE)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--hybrid-alpha", type=float, default=0.7)
    args = parser.parse_args()
    rerank_k = args.rerank_k if args.rerank_k is not None else args.top_k
    candidate_retriever = args.retriever or args.candidate_retriever
    rerank_retriever = args.retriever or args.rerank_retriever

    ensure_project_dirs()
    py = sys.executable

    if args.sample:
        nodes = "data/sample/nodes.jsonl"
        questions = "data/sample/questions.csv"
        edges = "outputs/parsed/edges.jsonl"
        run_step([py, "scripts/02_build_graph.py", "--nodes", nodes, "--output", edges])
    else:
        nodes = "outputs/parsed/nodes.jsonl"
        questions = args.questions
        edges = "outputs/parsed/edges.jsonl"
        if not args.skip_parse:
            parse_step = [
                py,
                "scripts/01_parse_pdf.py",
                "--parser",
                args.parser,
                "--mineru-output-dir",
                args.mineru_output_dir,
                "--mineru-api-url",
                args.mineru_api_url,
                "--mineru-backend",
                args.mineru_backend,
                "--mineru-method",
                args.mineru_method,
                "--chunk-template",
                args.chunk_template,
                "--chunk-size",
                str(args.chunk_size),
            ]
            if args.mineru_lang:
                parse_step.extend(["--mineru-lang", args.mineru_lang])
            run_step(parse_step)
            run_step([py, "scripts/14_chunk_quality_report.py", "--nodes", nodes])
        if not args.skip_visual:
            visual_step = [
                py,
                "scripts/10_build_visual_evidence.py",
                "--nodes",
                nodes,
                "--output",
                nodes,
                "--dpi",
                str(args.visual_dpi),
            ]
            if args.visual_caption_model:
                visual_step.extend(
                    [
                        "--caption-provider",
                        args.visual_caption_provider,
                        "--caption-model",
                        args.visual_caption_model,
                        "--caption-device",
                        args.visual_caption_device,
                        "--max-captions",
                        str(args.visual_max_captions),
                    ]
                )
            elif args.visual_caption_provider == "qwen":
                visual_step.extend(
                    [
                        "--caption-provider",
                        "qwen",
                        "--qwen-model",
                        args.qwen_model,
                        "--qwen-base-url",
                        args.qwen_base_url,
                        "--qwen-api-key-env",
                        args.qwen_api_key_env,
                        "--max-captions",
                        str(args.visual_max_captions),
                    ]
                )
            run_step(visual_step)
        run_step([py, "scripts/02_build_graph.py", "--nodes", nodes, "--output", edges])

    run_step(
        [
            py,
            "scripts/03_retrieve_candidates.py",
            "--questions",
            questions,
            "--nodes",
            nodes,
            "--top-k",
            str(args.candidate_k),
            "--retriever",
            candidate_retriever,
            "--embedding-model",
            args.embedding_model,
            "--embedding-cache",
            args.embedding_cache,
            "--embedding-device",
            args.embedding_device,
            "--embedding-batch-size",
            str(args.embedding_batch_size),
            "--hybrid-alpha",
            str(args.hybrid_alpha),
        ]
    )
    run_step(
        [
            py,
            "scripts/04_rerank.py",
            "--questions",
            questions,
            "--nodes",
            nodes,
            "--edges",
            edges,
            "--top-k",
            str(rerank_k),
            "--retriever",
            rerank_retriever,
            "--embedding-model",
            args.embedding_model,
            "--embedding-cache",
            args.embedding_cache,
            "--embedding-device",
            args.embedding_device,
            "--embedding-batch-size",
            str(args.embedding_batch_size),
            "--hybrid-alpha",
            str(args.hybrid_alpha),
        ]
    )
    run_step([py, "scripts/05_evaluate.py", "--questions", questions])
    run_step([py, "scripts/08_compare_methods.py", "--questions", questions])
    run_step([py, "scripts/09_build_evidence_chains.py", "--questions", questions])
    run_step([py, "scripts/11_build_evidence_cards.py", "--questions", questions])
    run_step([py, "scripts/12_check_evidence_cards.py", "--questions", questions])


if __name__ == "__main__":
    main()
