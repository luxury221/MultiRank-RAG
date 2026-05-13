from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline_common import as_float, clean_text, preview, read_csv, read_jsonl, resolve_path, write_csv
from rerank_lib import _alias_in_text, node_embedding_text, node_retrieval_text


FIELDS = [
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
    "union_sources",
    "route_product",
    "route_score",
]


PRODUCT_DOC_HINTS = {
    "air_conditioner": ("空调", "air conditioner"),
    "drill": ("电钻", "drill"),
    "air_purifier": ("空气净化器", "air purifier"),
    "hair_dryer": ("吹风机", "blower", "hair dryer"),
    "dishwasher": ("洗碗机", "dishwasher"),
    "exercise_bike": ("健身单车", "exercise bike"),
    "steam_cleaner": ("蒸汽清洁机", "steam cleaner"),
    "kids_motorcycle": ("儿童电动摩托车", "motorcycle"),
    "refrigerator": ("冰箱", "refrigerator"),
    "jetski": ("摩托艇", "jetski", "waverunner", "watercraft"),
    "ergonomic_chair": ("人体工学椅", "ergonomic chair"),
    "keyboard": ("功能键盘", "keyboard"),
    "oven": ("烤箱", "oven"),
    "camera": ("相机", "camera"),
    "thermostat": ("温控器", "thermostat"),
    "fitness_tracker": ("健身追踪器", "fitness tracker"),
    "water_pump": ("水泵", "water pump"),
    "generator": ("发电机", "generator"),
    "vr_headset": ("VR", "vr headset"),
    "mouse": ("鼠标", "mouse"),
    "earphones": ("耳机", "earphone", "earbud", "headphone"),
    "ereader": ("电子书", "阅读器", "ereader", "e-reader"),
    "fax": ("传真", "fax"),
    "grill": ("烤架", "grill"),
    "landline": ("座机", "landline", "handset", "base station"),
    "lawn_mower": ("割草", "lawn mower", "mower"),
    "microwave": ("微波炉", "microwave"),
    "motherboard": ("主板", "motherboard"),
    "pressure_cooker": ("压力锅", "空气炸锅", "pressure cooker", "air fryer"),
    "vacuum": ("扫地", "吸尘", "vacuum"),
    "snowmobile": ("雪地摩托", "snowmobile"),
    "tv_radio": ("电视", "收音", "television", "radio", "dvd"),
    "toothbrush": ("牙刷", "toothbrush"),
}


def load_routes(path: str | Path) -> dict[str, dict[str, str]]:
    routes: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        qid = clean_text(row.get("question_id"))
        if qid:
            routes[qid] = row
    return routes


def node_product_score(product: str, node: dict[str, Any]) -> float:
    if not product:
        return 0.0
    hints = PRODUCT_DOC_HINTS.get(product, ())
    if not hints:
        return 0.0
    doc = clean_text(node.get("doc_id")).casefold()
    section = clean_text(node.get("section")).casefold()
    source = clean_text(node.get("source_ref")).casefold()
    content = clean_text(node.get("content")).casefold()
    score = 0.0
    for hint in hints:
        h = hint.casefold()
        if _alias_in_text(h, doc):
            score += 0.62
        if _alias_in_text(h, source) or _alias_in_text(h, section):
            score += 0.28
        if _alias_in_text(h, content):
            score += 0.12
    return min(1.0, score)


def source_rank_score(rank: int, weight: float, k: int = 35) -> float:
    return weight * (1.0 / (k + max(1, rank)))


def node_type_prior(route: dict[str, str], node: dict[str, Any]) -> float:
    node_id = clean_text(node.get("node_id"))
    node_type = clean_text(node.get("node_type")) or "text"
    structure_type = clean_text(node.get("structure_type"))
    route_name = clean_text(route.get("route"))
    prior = 1.0
    if node_id.startswith("AS_PROFILE_") or structure_type == "manual_profile":
        prior *= 0.42
    if node_type == "text":
        prior *= 1.12
    elif node_type == "title":
        prior *= 0.72
    elif node_type == "page":
        prior *= 0.45
    elif node_type in {"figure", "caption", "table"} and route_name.startswith("service"):
        prior *= 0.55
    elif node_type in {"figure", "caption", "table"}:
        prior *= 0.88
    return prior


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a product-aware union candidate pool from multiple candidate CSVs.")
    parser.add_argument("--questions", default="outputs/after_sales_kb/questions.csv")
    parser.add_argument("--nodes", default="outputs/after_sales_kb/nodes.qwen_full.jsonl")
    parser.add_argument("--routes", default="outputs/after_sales_kb/question_routes.csv")
    parser.add_argument("--outputs", default="outputs/after_sales_kb/candidates_union.csv")
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate CSV with optional source weight, for example old=outputs/.../candidates.csv:1.25",
    )
    args = parser.parse_args()

    candidate_specs = args.candidate or [
        "old=outputs/after_sales_kb/candidates.csv:1.35",
        "best=outputs/after_sales_kb/candidates_best.csv:0.88",
    ]
    questions = {row["question_id"]: row for row in read_csv(args.questions) if clean_text(row.get("question_id"))}
    routes = load_routes(args.routes)
    nodes_by_id = {clean_text(node.get("node_id")): node for node in read_jsonl(args.nodes) if clean_text(node.get("node_id"))}
    aggregate: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for spec in candidate_specs:
        label_part, _, rest = spec.partition("=")
        label = clean_text(label_part) or "source"
        path_part, _, weight_part = rest.rpartition(":")
        if not path_part:
            path_part = rest
            weight = 1.0
        else:
            weight = as_float(weight_part, 1.0)
        path = resolve_path(path_part)
        if not path.exists():
            print(f"Skipping missing candidate file: {path}")
            continue
        for row in read_csv(path):
            qid = clean_text(row.get("question_id"))
            node_id = clean_text(row.get("node_id"))
            if qid not in questions or node_id not in nodes_by_id:
                continue
            rank = int(as_float(row.get("rank"), 9999))
            node = nodes_by_id[node_id]
            route = routes.get(qid, {})
            product = clean_text(route.get("product"))
            product_score = node_product_score(product, node)
            base = source_rank_score(rank, weight)
            if product:
                base *= 0.65 + 0.75 * product_score
            base *= node_type_prior(route, node)
            # Preserve direct lexical relevance so old text hits are not washed out by visual routes.
            if node_embedding_text(node):
                base += 0.0005 * min(3000, len(node_retrieval_text(node)))
            bucket = aggregate[qid]
            current = bucket.get(node_id)
            if current is None:
                bucket[node_id] = {
                    "node_id": node_id,
                    "score": base,
                    "route_score": product_score,
                    "sources": [label],
                }
            else:
                current["score"] += base
                current["route_score"] = max(current["route_score"], product_score)
                if label not in current["sources"]:
                    current["sources"].append(label)

    rows: list[dict[str, Any]] = []
    for qid, question in questions.items():
        route = routes.get(qid, {})
        ranked = sorted(
            aggregate.get(qid, {}).values(),
            key=lambda item: (item["score"], item["route_score"], len(item["sources"])),
            reverse=True,
        )[: args.top_k]
        if not ranked:
            continue
        max_score = max(item["score"] for item in ranked) or 1.0
        for rank, item in enumerate(ranked, start=1):
            node = nodes_by_id[item["node_id"]]
            rows.append(
                {
                    "question_id": qid,
                    "doc_id": question.get("doc_id", ""),
                    "question": question.get("question", ""),
                    "rank": rank,
                    "node_id": item["node_id"],
                    "node_type": node.get("node_type", ""),
                    "page": node.get("page", ""),
                    "score": round(item["score"] / max_score, 6),
                    "retriever": "union_product",
                    "embedding_model": "",
                    "source_ref": node.get("source_ref", ""),
                    "content_preview": preview(node.get("content", "")),
                    "union_sources": ";".join(item["sources"]),
                    "route_product": route.get("product", ""),
                    "route_score": round(item["route_score"], 6),
                }
            )

    write_csv(args.outputs, rows, FIELDS)
    counts = defaultdict(int)
    for row in rows:
        counts[row["question_id"]] += 1
    print(f"Wrote {len(rows)} rows for {len(counts)} questions to {resolve_path(args.outputs)}")
    if counts:
        print(f"Per-question rows: min={min(counts.values())}, max={max(counts.values())}")


if __name__ == "__main__":
    main()
