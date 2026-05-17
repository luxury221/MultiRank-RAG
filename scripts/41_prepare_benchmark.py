from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from pipeline_common import clean_text, ensure_project_dirs, resolve_path, write_csv, write_jsonl


QUESTION_FIELDS = [
    "question_id",
    "doc_id",
    "question",
    "answer",
    "question_type",
    "gold_node_ids",
    "gold_pages",
    "gold_modalities",
    "evidence_note",
]

DEFAULT_RAW_ROOT = os.environ.get("BENCHMARK_RAW_ROOT", r"D:\ai_models\benchmarks")

LOCAL_DATASET_DIRS = {
    "rungalileo/ragbench": "ragbench",
    "galileo-ai/ragbench": "ragbench",
    "grasson/t2-ragbench": "t2-ragbench-metadata",
    "G4KMU/t2-ragbench": "t2-ragbench-metadata",
    "VLM2Vec/MMLongBench-doc": "mmlongbench-doc-retrieval",
}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return [value]


def safe_id(value: Any, fallback: str = "item") -> str:
    text = clean_text(value)
    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text).strip("_")
    return text[:120] or fallback


def semicolon(items: list[Any]) -> str:
    return ";".join(clean_text(item) for item in items if clean_text(item))


def hf_download(repo_id: str, filename: str, raw_root: str | Path = DEFAULT_RAW_ROOT) -> Path:
    local_dir = LOCAL_DATASET_DIRS.get(repo_id)
    if local_dir:
        local_path = Path(raw_root) / local_dir / filename
        if local_path.exists():
            return local_path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Install it with `pip install huggingface_hub`.") from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def read_parquet(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas + pyarrow are required to read Hugging Face parquet files.") from exc
    return pd.read_parquet(path)


def relevant_doc_indices_from_ragbench(row: dict[str, Any], num_docs: int) -> list[int]:
    keys: list[str] = []
    for field in ["all_relevant_sentence_keys", "all_utilized_sentence_keys", "unsupported_response_sentence_keys"]:
        keys.extend(clean_text(item) for item in as_list(row.get(field)) if clean_text(item))
    for support in as_list(row.get("sentence_support_information")):
        keys.extend(clean_text(item) for item in as_list(support) if clean_text(item))
    indices: set[int] = set()
    for key in keys:
        match = re.match(r"(\d+)[A-Za-z]", key)
        if match:
            indices.add(int(match.group(1)))
    if not indices:
        indices = set(range(num_docs))
    return sorted(index for index in indices if 0 <= index < num_docs)


def prepare_ragbench(args: argparse.Namespace) -> None:
    filename = f"{args.subset}/{args.split}-00000-of-00001.parquet"
    parquet_path = hf_download("rungalileo/ragbench", filename, args.raw_root)
    df = read_parquet(parquet_path).head(args.max_questions)
    nodes: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []

    for row_index, raw in enumerate(df.to_dict("records"), start=1):
        row = dict(raw)
        raw_id = clean_text(row.get("id")) or str(row_index)
        doc_id = f"ragbench_{args.subset}_{safe_id(raw_id)}"
        documents = [clean_text(item) for item in as_list(row.get("documents")) if clean_text(item)]
        gold_indices = relevant_doc_indices_from_ragbench(row, len(documents))
        gold_nodes: list[str] = []
        for doc_index, document in enumerate(documents):
            node_id = f"{doc_id}_doc_{doc_index}"
            if doc_index in gold_indices:
                gold_nodes.append(node_id)
            nodes.append(
                {
                    "node_id": node_id,
                    "doc_id": doc_id,
                    "page": doc_index + 1,
                    "node_type": "text",
                    "content": document,
                    "source_ref": f"RAGBench/{args.subset}/{args.split}/doc_{doc_index}",
                    "section": args.subset,
                    "benchmark": "RAGBench",
                    "benchmark_subset": args.subset,
                }
            )
        questions.append(
            {
                "question_id": f"ragbench_{args.subset}_{safe_id(raw_id)}",
                "doc_id": doc_id,
                "question": clean_text(row.get("question")),
                "answer": clean_text(row.get("response")),
                "question_type": "basic_rag",
                "gold_node_ids": semicolon(gold_nodes),
                "gold_pages": "",
                "gold_modalities": "text",
                "evidence_note": "Gold nodes are derived from RAGBench relevant/utilized sentence keys.",
            }
        )

    write_benchmark(args.output_dir, nodes, questions, "RAGBench", {"subset": args.subset, "split": args.split})


def prepare_multihop(args: argparse.Namespace) -> None:
    root = resolve_path(args.multihop_dir)
    corpus_path = root / "dataset" / "corpus.json"
    queries_path = root / "dataset" / "MultiHopRAG.json"
    if not corpus_path.exists() or not queries_path.exists():
        raise SystemExit(
            f"MultiHop-RAG files not found under {root}. "
            "Clone https://github.com/yixuantt/MultiHop-RAG into external/MultiHop-RAG first."
        )
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    query_rows = json.loads(queries_path.read_text(encoding="utf-8"))[: args.max_questions]
    url_to_node: dict[str, str] = {}
    title_to_node: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []

    for index, item in enumerate(corpus):
        node_id = f"multihop_doc_{index}"
        title = clean_text(item.get("title"))
        url = clean_text(item.get("url"))
        body = clean_text(item.get("body"))
        source = clean_text(item.get("source"))
        content = clean_text(f"{title}\n\nSource: {source}\nURL: {url}\n\n{body}")
        nodes.append(
            {
                "node_id": node_id,
                "doc_id": "multihop_rag_corpus",
                "page": index + 1,
                "node_type": "text",
                "content": content,
                "source_ref": url or title,
                "section": clean_text(item.get("category")),
                "benchmark": "MultiHop-RAG",
                "published_at": clean_text(item.get("published_at")),
            }
        )
        if url:
            url_to_node[url] = node_id
        if title:
            title_to_node[title] = node_id

    questions: list[dict[str, Any]] = []
    for index, item in enumerate(query_rows, start=1):
        gold_nodes: list[str] = []
        for evidence in as_list(item.get("evidence_list")):
            if not isinstance(evidence, dict):
                continue
            node_id = url_to_node.get(clean_text(evidence.get("url"))) or title_to_node.get(clean_text(evidence.get("title")))
            if node_id and node_id not in gold_nodes:
                gold_nodes.append(node_id)
        questions.append(
            {
                "question_id": f"multihop_{index:04d}",
                "doc_id": "",
                "question": clean_text(item.get("query")),
                "answer": clean_text(item.get("answer")),
                "question_type": clean_text(item.get("question_type")) or "multi_hop",
                "gold_node_ids": semicolon(gold_nodes),
                "gold_pages": "",
                "gold_modalities": "text",
                "evidence_note": "Gold nodes are mapped from MultiHop-RAG evidence_list titles/URLs.",
            }
        )

    write_benchmark(args.output_dir, nodes, questions, "MultiHop-RAG", {"source": str(root)})


def t2_file_for(subset: str, split: str) -> str:
    subset = subset.strip()
    if subset == "ConvFinQA":
        return "data/ConvFinQA/turn_0.jsonl"
    return f"data/{subset}/{split}/metadata.jsonl"


def prepare_t2_ragbench(args: argparse.Namespace) -> None:
    filename = t2_file_for(args.subset, args.split)
    jsonl_path = hf_download("grasson/t2-ragbench", filename, args.raw_root)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.max_questions:
                break

    nodes_by_id: dict[str, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []
    for row in rows:
        context_id = f"t2_{args.subset}_{safe_id(row.get('context_id'))}"
        text_parts = [clean_text(row.get("pre_text")), clean_text(row.get("post_text"))]
        if not any(text_parts):
            text_parts = [clean_text(row.get("context"))]
        table_text = clean_text(row.get("table"))
        text_node_id = f"{context_id}_text"
        table_node_id = f"{context_id}_table"

        if clean_text(" ".join(text_parts)) and text_node_id not in nodes_by_id:
            nodes_by_id[text_node_id] = {
                "node_id": text_node_id,
                "doc_id": context_id,
                "page": clean_text(row.get("page_number")) or 1,
                "node_type": "text",
                "content": clean_text("\n\n".join(text_parts)),
                "source_ref": clean_text(row.get("file_name")) or context_id,
                "section": args.subset,
                "benchmark": "T2-RAGBench",
            }
        if table_text and table_node_id not in nodes_by_id:
            nodes_by_id[table_node_id] = {
                "node_id": table_node_id,
                "doc_id": context_id,
                "page": clean_text(row.get("page_number")) or 1,
                "node_type": "table",
                "content": table_text,
                "source_ref": clean_text(row.get("file_name")) or context_id,
                "section": args.subset,
                "benchmark": "T2-RAGBench",
                "visual_summary": "Financial table evidence extracted from T2-RAGBench metadata.",
            }
        gold_nodes = [node_id for node_id in [table_node_id, text_node_id] if node_id in nodes_by_id]
        modalities = ["table" if table_node_id in nodes_by_id else "", "text" if text_node_id in nodes_by_id else ""]
        questions.append(
            {
                "question_id": f"t2_{args.subset}_{safe_id(row.get('id'))}",
                "doc_id": "",
                "question": clean_text(row.get("question")),
                "answer": clean_text(row.get("original_answer") or row.get("program_answer")),
                "question_type": "text_table_rag",
                "gold_node_ids": semicolon(gold_nodes),
                "gold_pages": clean_text(row.get("page_number")),
                "gold_modalities": semicolon(modalities),
                "evidence_note": "Gold nodes are the source context text/table from T2-RAGBench.",
            }
        )

    write_benchmark(args.output_dir, list(nodes_by_id.values()), questions, "T2-RAGBench", {"subset": args.subset, "split": args.split})


def prepare_mmlongbench_doc(args: argparse.Namespace) -> None:
    query_path = hf_download("VLM2Vec/MMLongBench-doc", "queries/test-00000-of-00001.parquet", args.raw_root)
    qrels_path = hf_download("VLM2Vec/MMLongBench-doc", "qrels/test-00000-of-00001.parquet", args.raw_root)
    queries = read_parquet(query_path).head(args.max_questions)
    qrels = read_parquet(qrels_path)
    selected_qids = set(int(value) for value in queries["query-id"].tolist())
    qrels = qrels[qrels["query-id"].isin(selected_qids)]
    qrels = qrels[qrels["score"] > 0]
    required_corpus_ids = set(int(value) for value in qrels["corpus-id"].tolist())

    output_root = resolve_path(args.output_dir)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    nodes_by_id: dict[int, dict[str, Any]] = {}
    for shard in range(3):
        if not required_corpus_ids:
            break
        corpus_path = hf_download(
            "VLM2Vec/MMLongBench-doc",
            f"corpus/test-{shard:05d}-of-00003.parquet",
            args.raw_root,
        )
        corpus = read_parquet(corpus_path)
        corpus = corpus[corpus["corpus-id"].isin(required_corpus_ids)]
        for row in corpus.to_dict("records"):
            corpus_id = int(row["corpus-id"])
            image = row.get("image") or {}
            image_path = image_dir / f"corpus_{corpus_id}.png"
            data = image.get("bytes") if isinstance(image, dict) else None
            if data and not image_path.exists():
                image_path.write_bytes(data)
            node_id = f"mmlongbench_doc_corpus_{corpus_id}"
            nodes_by_id[corpus_id] = {
                "node_id": node_id,
                "doc_id": "",
                "page": corpus_id + 1,
                "node_type": "figure",
                "content": f"MMLongBench-Doc page image. corpus_id={corpus_id}",
                "source_ref": f"MMLongBench-Doc/corpus/{corpus_id}",
                "crop_image_path": str(image_path),
                "page_image_path": str(image_path),
                "visual_summary": f"Document page image from MMLongBench-Doc corpus id {corpus_id}.",
                "benchmark": "MMLongBench-Doc",
            }
        required_corpus_ids -= set(nodes_by_id)

    qrels_by_qid: dict[int, list[int]] = {}
    for row in qrels.to_dict("records"):
        qid = int(row["query-id"])
        corpus_id = int(row["corpus-id"])
        if corpus_id in nodes_by_id:
            qrels_by_qid.setdefault(qid, []).append(corpus_id)

    questions: list[dict[str, Any]] = []
    for row in queries.to_dict("records"):
        qid = int(row["query-id"])
        gold_nodes = [nodes_by_id[corpus_id]["node_id"] for corpus_id in qrels_by_qid.get(qid, [])]
        questions.append(
            {
                "question_id": f"mmlongbench_doc_{qid}",
                "doc_id": "",
                "question": clean_text(row.get("query")),
                "answer": "",
                "question_type": "multimodal_document_retrieval",
                "gold_node_ids": semicolon(gold_nodes),
                "gold_pages": "",
                "gold_modalities": "figure",
                "evidence_note": "Gold nodes come from MMLongBench-Doc qrels. This adapter evaluates retrieval, not answer EM.",
            }
        )

    write_benchmark(args.output_dir, list(nodes_by_id.values()), questions, "MMLongBench-Doc", {"split": "test"})


def write_benchmark(
    output_dir: str | Path,
    nodes: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    name: str,
    metadata: dict[str, Any],
) -> None:
    output_root = resolve_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "nodes.jsonl", nodes)
    write_csv(output_root / "questions.csv", questions, QUESTION_FIELDS)
    report = {
        "benchmark": name,
        "num_nodes": len(nodes),
        "num_questions": len(questions),
        "metadata": metadata,
        "nodes": str(output_root / "nodes.jsonl"),
        "questions": str(output_root / "questions.csv"),
    }
    (output_root / "dataset_info.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {name}: nodes={len(nodes)}, questions={len(questions)}")
    print(f"Output: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/convert public benchmarks into MultiRank-RAG format.")
    parser.add_argument(
        "--dataset",
        choices=["ragbench", "multihop-rag", "t2-ragbench", "mmlongbench-doc"],
        required=True,
    )
    parser.add_argument("--output-dir", default="", help="Default: outputs/benchmarks/<dataset>")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--subset", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--multihop-dir", default="external/MultiHop-RAG")
    parser.add_argument(
        "--raw-root",
        default=DEFAULT_RAW_ROOT,
        help="Local raw benchmark root. Default: BENCHMARK_RAW_ROOT or D:\\ai_models\\benchmarks.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    if not args.output_dir:
        args.output_dir = f"outputs/benchmarks/{args.dataset}"
    if args.dataset == "ragbench":
        args.subset = args.subset or "emanual"
        prepare_ragbench(args)
    elif args.dataset == "multihop-rag":
        prepare_multihop(args)
    elif args.dataset == "t2-ragbench":
        args.subset = args.subset or "FinQA"
        args.split = args.split or "dev"
        prepare_t2_ragbench(args)
    elif args.dataset == "mmlongbench-doc":
        prepare_mmlongbench_doc(args)


if __name__ == "__main__":
    main()
