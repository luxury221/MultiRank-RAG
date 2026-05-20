from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embedding_index import DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_EMBEDDING_DEVICE, DEFAULT_EMBEDDING_MODEL
from pipeline_common import clean_text, ensure_project_dirs, read_csv, read_jsonl, resolve_path, write_csv, write_jsonl


ROOT = Path(__file__).resolve().parents[1]

VISUAL_FIELDS = {
    "visual_caption",
    "visual_caption_model",
    "visual_caption_error",
    "visual_title",
    "visual_type",
    "key_objects",
    "ocr_text",
    "data_or_trends",
    "qa_evidence",
    "limitations",
    "visual_summary",
    "crop_image_path",
    "page_image_path",
    "bbox",
    "bbox_source",
}

VARIANT_FIELDS = [
    "dataset",
    "run_name",
    "variant",
    "label",
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
    "delta_recall_at_5_vs_v0",
    "delta_ndcg_at_5_vs_v0",
    "delta_mrr_vs_v0",
    "variant_dir",
]


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    label: str
    force_text_only: bool
    strip_visual: bool
    build_graph: bool
    build_graphrag: bool
    method: str
    default_retriever: str
    enable_model_rerank: bool
    build_chain: bool
    context_expansion: bool = False
    adaptive_rerank_boost: bool = False
    graph_context_boost: bool = False
    evidence_guard: bool = False


VARIANTS: dict[str, ExperimentVariant] = {
    "V0": ExperimentVariant(
        name="V0",
        label="Vanilla text-only RAG",
        force_text_only=True,
        strip_visual=True,
        build_graph=False,
        build_graphrag=False,
        method="G1",
        default_retriever="embedding",
        enable_model_rerank=False,
        build_chain=False,
    ),
    "V1": ExperimentVariant(
        name="V1",
        label="+ structured evidence nodes",
        force_text_only=False,
        strip_visual=True,
        build_graph=True,
        build_graphrag=False,
        method="G1",
        default_retriever="fusion",
        enable_model_rerank=False,
        build_chain=False,
    ),
    "V2": ExperimentVariant(
        name="V2",
        label="+ visual evidence fields",
        force_text_only=False,
        strip_visual=False,
        build_graph=True,
        build_graphrag=False,
        method="G1",
        default_retriever="fusion",
        enable_model_rerank=False,
        build_chain=False,
    ),
    "V3": ExperimentVariant(
        name="V3",
        label="+ GraphRAG retrieval signal",
        force_text_only=False,
        strip_visual=False,
        build_graph=True,
        build_graphrag=True,
        method="G1",
        default_retriever="fusion",
        enable_model_rerank=False,
        build_chain=False,
    ),
    "V4": ExperimentVariant(
        name="V4",
        label="Full MultiRank-RAG evidence chain",
        force_text_only=False,
        strip_visual=False,
        build_graph=True,
        build_graphrag=True,
        method="G4",
        default_retriever="fusion",
        enable_model_rerank=True,
        build_chain=True,
    ),
    "V5": ExperimentVariant(
        name="V5",
        label="Enhanced MultiRank-RAG (ABECD + Guard)",
        force_text_only=False,
        strip_visual=False,
        build_graph=True,
        build_graphrag=True,
        method="G4",
        default_retriever="multiroute",
        enable_model_rerank=True,
        build_chain=True,
        context_expansion=True,
        adaptive_rerank_boost=True,
        graph_context_boost=True,
        evidence_guard=True,
    ),
}


DATASET_PRESETS = {
    "sample": {
        "nodes": "data/sample/nodes.jsonl",
        "questions": "data/sample/questions.csv",
    },
    "current": {
        "nodes": "outputs/parsed/nodes.jsonl",
        "questions": "data/questions.csv",
    },
}


def parse_variant_list(value: str) -> list[str]:
    names = [item.strip().upper() for item in value.split(",") if item.strip()]
    invalid = [name for name in names if name not in VARIANTS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown variant(s): {', '.join(invalid)}")
    return list(dict.fromkeys(names)) or list(VARIANTS)


def remove_visual_sections(text: Any) -> str:
    text = clean_text(text)
    marker = "\n\nVisual summary:"
    if marker in text:
        text = text.split(marker, 1)[0]
    return clean_text(text)


def transform_nodes(nodes: list[dict[str, Any]], variant: ExperimentVariant) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for node in nodes:
        item = dict(node)
        item["content"] = remove_visual_sections(item.get("content", ""))
        if variant.strip_visual:
            for key in list(item):
                if key in VISUAL_FIELDS or any(key.endswith(f"_{field}") for field in VISUAL_FIELDS):
                    item.pop(key, None)
        if variant.force_text_only:
            item["node_type"] = "text"
            for key in [
                "layout_role",
                "structure_type",
                "chunk_strategy",
                "parent_chunk_id",
                "previous_node_id",
                "next_node_id",
                "explicit_refs",
            ]:
                item.pop(key, None)
        transformed.append(item)
    return transformed


def run_step(args: list[str], env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print(">", " ".join(args), flush=True)
    if dry_run:
        return
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def prepare_edges(
    py: str,
    variant: ExperimentVariant,
    nodes_path: Path,
    edges_path: Path,
    dry_run: bool,
    enhanced_context_edges: bool = False,
) -> None:
    if not variant.build_graph:
        if not dry_run:
            write_jsonl(edges_path, [])
        return
    cmd = [
            py,
            "scripts/02_build_graph.py",
            "--nodes",
            str(nodes_path),
            "--output",
            str(edges_path),
        ]
    if enhanced_context_edges:
        cmd.append("--enhanced-context-edges")
    run_step(cmd, dry_run=dry_run)


def build_graphrag(py: str, variant: ExperimentVariant, nodes_path: Path, edges_path: Path, kg_dir: Path, dry_run: bool) -> str:
    if not variant.build_graphrag:
        if not dry_run:
            kg_dir.mkdir(parents=True, exist_ok=True)
        return ""
    run_step(
        [
            py,
            "scripts/23_build_graphrag.py",
            "--nodes",
            str(nodes_path),
            "--edges",
            str(edges_path),
            "--output-dir",
            str(kg_dir),
        ],
        dry_run=dry_run,
    )
    return str(kg_dir)


def run_variant(args: argparse.Namespace, variant: ExperimentVariant, base_nodes: list[dict[str, Any]], py: str) -> list[dict[str, Any]]:
    output_root = resolve_path(args.output_dir) / args.dataset_name / args.run_name
    variant_dir = output_root / variant.name
    metrics_dir = variant_dir / "metrics"
    nodes_path = variant_dir / "nodes.jsonl"
    edges_path = variant_dir / "edges.jsonl"
    kg_dir = variant_dir / "graphrag"
    candidates_path = variant_dir / "candidates.csv"
    rankings_path = variant_dir / "reranked.csv"
    per_question_path = metrics_dir / "per_question_metrics.csv"
    summary_path = metrics_dir / "summary_metrics.csv"
    embedding_cache = resolve_path(args.embedding_cache_dir) if args.embedding_cache_dir else output_root / "embeddings"

    if not args.dry_run:
        if variant_dir.exists() and args.clean:
            shutil.rmtree(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(nodes_path, transform_nodes(base_nodes, variant))

    prepare_edges(
        py,
        variant,
        nodes_path,
        edges_path,
        args.dry_run,
        enhanced_context_edges=bool((args.graph_context_boost or variant.graph_context_boost) and variant.build_graph),
    )
    kg_arg = build_graphrag(py, variant, nodes_path, edges_path, kg_dir, args.dry_run)
    if not kg_arg:
        kg_arg = str(kg_dir)

    retriever = args.retriever or variant.default_retriever
    env = os.environ.copy()
    env["RAG_ENABLE_MODEL_RERANK"] = "1" if variant.enable_model_rerank else "0"

    retrieve_cmd = [
            py,
            "scripts/03_retrieve_candidates.py",
            "--questions",
            args.questions,
            "--nodes",
            str(nodes_path),
            "--output",
            str(candidates_path),
            "--top-k",
            str(args.candidate_k),
            "--retriever",
            retriever,
            "--embedding-model",
            args.embedding_model,
            "--embedding-cache",
            str(embedding_cache),
            "--embedding-device",
            args.embedding_device,
            "--embedding-batch-size",
            str(args.embedding_batch_size),
            "--hybrid-alpha",
            str(args.hybrid_alpha),
            "--kg-dir",
            kg_arg,
        ]
    if args.context_expansion or variant.context_expansion:
        retrieve_cmd.append("--context-expansion")
    run_step(retrieve_cmd, env=env, dry_run=args.dry_run)

    rerank_cmd = [
            py,
            "scripts/04_rerank.py",
            "--questions",
            args.questions,
            "--nodes",
            str(nodes_path),
            "--edges",
            str(edges_path),
            "--candidates",
            str(candidates_path),
            "--output",
            str(rankings_path),
            "--top-k",
            str(args.rerank_k),
            "--retriever",
            retriever,
            "--embedding-model",
            args.embedding_model,
            "--embedding-cache",
            str(embedding_cache),
            "--embedding-device",
            args.embedding_device,
            "--embedding-batch-size",
            str(args.embedding_batch_size),
            "--hybrid-alpha",
            str(args.hybrid_alpha),
            "--kg-dir",
            kg_arg,
            "--methods",
            variant.method,
        ]
    if args.context_expansion or variant.context_expansion:
        rerank_cmd.append("--context-expansion")
    if args.adaptive_rerank_boost or variant.adaptive_rerank_boost:
        rerank_cmd.append("--adaptive-rerank-boost")
    if args.graph_context_boost or variant.graph_context_boost:
        rerank_cmd.append("--graph-context-boost")
    run_step(rerank_cmd, env=env, dry_run=args.dry_run)
    run_step(
        [
            py,
            "scripts/05_evaluate.py",
            "--questions",
            args.questions,
            "--rankings",
            str(rankings_path),
            "--per-question-output",
            str(per_question_path),
            "--summary-output",
            str(summary_path),
        ],
        env=env,
        dry_run=args.dry_run,
    )

    if variant.build_chain and args.build_chains:
        chain_dir = variant_dir / "evidence_chains"
        chain_cmd = [
                py,
                "scripts/09_build_evidence_chains.py",
                "--questions",
                args.questions,
                "--nodes",
                str(nodes_path),
                "--edges",
                str(edges_path),
                "--rankings",
                str(rankings_path),
                "--method",
                variant.method,
                "--output-jsonl",
                str(chain_dir / "chains.jsonl"),
                "--output-csv",
                str(chain_dir / "chain_steps.csv"),
                "--output-md",
                str(chain_dir / "evidence_chains.md"),
            ]
        if args.evidence_guard or variant.evidence_guard:
            chain_cmd.append("--evidence-guard")
        run_step(chain_cmd, env=env, dry_run=args.dry_run)
        run_step(
            [
                py,
                "scripts/12_evaluate_evidence_chains.py",
                "--questions",
                args.questions,
                "--chains",
                str(chain_dir / "chains.jsonl"),
                "--per-question-output",
                str(chain_dir / "chain_eval_per_question.csv"),
                "--summary-output",
                str(chain_dir / "chain_eval_summary.csv"),
            ],
            env=env,
            dry_run=args.dry_run,
        )
        if args.generate_answers:
            answer_cmd = [
                py,
                "scripts/34_generate_chain_answers.py",
                "--chains",
                str(chain_dir / "chains.jsonl"),
                "--output-csv",
                str(chain_dir / "answers.csv"),
                "--output-jsonl",
                str(chain_dir / "answers.jsonl"),
                "--cache",
                str(resolve_path(args.answer_cache) if args.answer_cache else chain_dir / "answer_cache.jsonl"),
                "--provider",
                args.answer_provider,
                "--max-steps",
                str(args.answer_max_steps),
                "--max-tokens",
                str(args.answer_max_tokens),
                "--temperature",
                str(args.answer_temperature),
            ]
            if args.answer_model:
                answer_cmd.extend(["--model", args.answer_model])
            run_step(answer_cmd, env=env, dry_run=args.dry_run)

    if args.dry_run:
        return []
    summary_rows = read_csv(summary_path)
    for row in summary_rows:
        row.update(
            {
                "dataset": args.dataset_name,
                "run_name": args.run_name,
                "variant": variant.name,
                "label": variant.label,
                "variant_dir": str(variant_dir),
            }
        )
    return summary_rows


def add_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((row for row in rows if row.get("variant") == "V0"), None)
    if not baseline:
        for row in rows:
            row["delta_recall_at_5_vs_v0"] = ""
            row["delta_ndcg_at_5_vs_v0"] = ""
            row["delta_mrr_vs_v0"] = ""
        return rows

    def value(row: dict[str, Any], field: str) -> float:
        try:
            return float(row.get(field) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    for row in rows:
        row["delta_recall_at_5_vs_v0"] = round(value(row, "recall_at_5") - value(baseline, "recall_at_5"), 6)
        row["delta_ndcg_at_5_vs_v0"] = round(value(row, "ndcg_at_5") - value(baseline, "ndcg_at_5"), 6)
        row["delta_mrr_vs_v0"] = round(value(row, "mrr") - value(baseline, "mrr"), 6)
    return rows


def resolve_dataset_paths(args: argparse.Namespace) -> None:
    preset = DATASET_PRESETS.get(args.dataset_name)
    if preset:
        args.nodes = args.nodes or preset["nodes"]
        args.questions = args.questions or preset["questions"]
    if not args.nodes or not args.questions:
        raise SystemExit("Pass --nodes and --questions, or use --dataset-name sample/current.")
    args.nodes = str(resolve_path(args.nodes))
    args.questions = str(resolve_path(args.questions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main V0-V5 MultiRank-RAG ablation experiment.")
    parser.add_argument("--dataset-name", default="sample", help="sample, current, or a custom name.")
    parser.add_argument("--nodes", default="", help="Evidence nodes JSONL. Optional for sample/current.")
    parser.add_argument("--questions", default="", help="Question CSV. Optional for sample/current.")
    parser.add_argument("--output-dir", default="outputs/experiments")
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--variants", type=parse_variant_list, default=parse_variant_list("V0,V1,V2,V3,V4,V5"))
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument(
        "--retriever",
        choices=["", "fusion", "multiroute", "multi_route", "multi", "hybrid", "embedding", "lexical", "bm25", "kg"],
        default="",
        help="Optional override for all variants. Empty uses each variant default.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-cache-dir",
        default="",
        help="Shared embedding cache directory. Default: <output-dir>/<dataset>/<run-name>/embeddings.",
    )
    parser.add_argument("--embedding-device", default=DEFAULT_EMBEDDING_DEVICE)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--hybrid-alpha", type=float, default=0.7)
    parser.add_argument(
        "--context-expansion",
        action="store_true",
        help="Ablation A: expand candidates with same-context table/figure/text companions.",
    )
    parser.add_argument(
        "--adaptive-rerank-boost",
        action="store_true",
        help="Ablation B: enable stronger query-modality-aware reranking.",
    )
    parser.add_argument(
        "--graph-context-boost",
        action="store_true",
        help="Ablation E: build stronger context graph edges and increase graph rerank signals.",
    )
    parser.add_argument(
        "--evidence-guard",
        action="store_true",
        help="Filter noisy visual/table evidence when building evidence chains.",
    )
    parser.add_argument("--build-chains", action="store_true", help="Build evidence chains for V4.")
    parser.add_argument("--generate-answers", action="store_true", help="Generate final answers from V4 evidence chains.")
    parser.add_argument(
        "--answer-provider",
        default="none",
        choices=[
            "none",
            "ark",
            "doubao",
            "volcengine",
            "xinference",
            "openai_compatible",
            "openai-compatible",
            "local_openai",
            "local-server",
        ],
        help="Answer generation provider. Use none for deterministic extractive fallback.",
    )
    parser.add_argument("--answer-model", default="", help="Optional chat model override for answer generation.")
    parser.add_argument("--answer-cache", default="", help="Optional answer cache JSONL path.")
    parser.add_argument("--answer-max-steps", type=int, default=5)
    parser.add_argument("--answer-max-tokens", type=int, default=700)
    parser.add_argument("--answer-temperature", type=float, default=0.1)
    parser.add_argument("--clean", action="store_true", help="Remove each variant output directory before running it.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    resolve_dataset_paths(args)
    base_nodes = read_jsonl(args.nodes)
    if not base_nodes:
        raise SystemExit(f"No nodes found: {args.nodes}")
    if not read_csv(args.questions):
        raise SystemExit(f"No questions found: {args.questions}")

    py = sys.executable
    summary_rows: list[dict[str, Any]] = []
    for variant_name in args.variants:
        variant = VARIANTS[variant_name]
        print(f"\n=== {variant.name}: {variant.label} ===", flush=True)
        summary_rows.extend(run_variant(args, variant, base_nodes, py))

    if args.dry_run:
        return

    summary_rows = add_deltas(summary_rows)
    output_root = resolve_path(args.output_dir) / args.dataset_name / args.run_name
    summary_path = output_root / "main_experiment_summary.csv"
    write_csv(summary_path, summary_rows, VARIANT_FIELDS)
    print(f"\nWrote main experiment summary to {summary_path}")


if __name__ == "__main__":
    main()
