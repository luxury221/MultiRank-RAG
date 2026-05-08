from __future__ import annotations

import argparse
import itertools
import re
from collections import defaultdict
from typing import Any

from pipeline_common import (
    DEFAULT_EDGES,
    DEFAULT_NODES,
    EDGE_FIELDS,
    LEGACY_EDGES,
    clean_text,
    copy_jsonl_alias,
    ensure_project_dirs,
    read_jsonl,
    resolve_path,
    write_jsonl,
)


REF_RE = re.compile(r"(图|圖|Fig\.?|Figure|表|Table)\s*([0-9A-Za-z\.\-]+)", re.I)


def add_edge(edges: list[dict[str, Any]], source: str, target: str, edge_type: str, weight: float = 1.0) -> None:
    if not source or not target or source == target:
        return
    edges.append(
        {
            "source_id": source,
            "target_id": target,
            "edge_type": edge_type,
            "weight": round(float(weight), 4),
        }
    )


def extract_refs(text: str) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = set()
    for match in REF_RE.finditer(text or ""):
        raw_type = match.group(1).lower()
        kind = "table" if raw_type.startswith("表") or raw_type.startswith("table") else "figure"
        refs.add((kind, match.group(2).strip(".-")))
    return refs


def build_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_doc_page: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_doc_kind_ref: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    page_nodes: dict[tuple[str, str], str] = {}

    for node in nodes:
        doc_id = clean_text(node.get("doc_id"))
        page = str(node.get("page", ""))
        node_type = clean_text(node.get("node_type"))
        by_doc_page[(doc_id, page)].append(node)
        if node_type == "page":
            page_nodes[(doc_id, page)] = node.get("node_id", "")
        for kind, ref_no in extract_refs(node.get("content", "") + " " + node.get("source_ref", "")):
            if node_type in {kind, "caption"}:
                by_doc_kind_ref[(doc_id, kind, ref_no)].append(node)

    edges: list[dict[str, Any]] = []
    for (doc_id, page), page_group in by_doc_page.items():
        page_id = page_nodes.get((doc_id, page), "")
        non_page = [node for node in page_group if node.get("node_type") != "page"]
        for node in non_page:
            add_edge(edges, page_id, node.get("node_id", ""), "belongs_to_page", 0.05)
        for left, right in itertools.combinations(non_page, 2):
            add_edge(edges, left.get("node_id", ""), right.get("node_id", ""), "same_page", 0.03)

        captions = [node for node in non_page if node.get("node_type") == "caption"]
        figures = [node for node in non_page if node.get("node_type") == "figure"]
        tables = [node for node in non_page if node.get("node_type") == "table"]
        for caption in captions:
            text = caption.get("content", "")
            if re.search(r"^(图|圖|Fig|Figure)", text, re.I):
                for figure in figures:
                    add_edge(edges, caption["node_id"], figure["node_id"], "figure_caption", 1.2)
            if re.search(r"^(表|Table)", text, re.I):
                for table in tables:
                    add_edge(edges, caption["node_id"], table["node_id"], "table_caption", 1.2)

    for node in nodes:
        node_id = node.get("node_id")
        doc_id = clean_text(node.get("doc_id"))
        node_type = clean_text(node.get("node_type"))
        if node_type not in {"text", "caption"}:
            continue
        for kind, ref_no in extract_refs(node.get("content", "")):
            for target in by_doc_kind_ref.get((doc_id, kind, ref_no), []):
                add_edge(edges, node_id, target.get("node_id", ""), f"text_ref_{kind}", 1.0)

    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        source = edge["source_id"]
        target = edge["target_id"]
        key_nodes = tuple(sorted([source, target]))
        key = (key_nodes[0], key_nodes[1], edge["edge_type"])
        if key in dedup:
            dedup[key]["weight"] = round(float(dedup[key]["weight"]) + float(edge["weight"]), 4)
        else:
            dedup[key] = edge
    return list(dedup.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build document graph edges from parsed evidence nodes.")
    parser.add_argument("--nodes", default=str(DEFAULT_NODES.relative_to(DEFAULT_NODES.parents[2])))
    parser.add_argument("--output", default=str(DEFAULT_EDGES.relative_to(DEFAULT_EDGES.parents[2])))
    args = parser.parse_args()

    ensure_project_dirs()
    nodes = read_jsonl(args.nodes)
    edges = build_edges(nodes)
    write_jsonl(args.output, edges)
    copy_jsonl_alias(args.output, LEGACY_EDGES)
    print(f"Wrote {len(edges)} edges to {resolve_path(args.output)}")
    if not nodes:
        print("No nodes were found. Run scripts/01_parse_pdf.py first.")


if __name__ == "__main__":
    main()
