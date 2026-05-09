from __future__ import annotations

import argparse
import itertools
import json
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


def parse_bbox(value: Any) -> list[float]:
    if not value:
        return []
    try:
        payload = json.loads(value) if isinstance(value, str) else value
        bbox = [float(item) for item in payload]
    except Exception:
        return []
    if len(bbox) != 4:
        return []
    if max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]) <= 0:
        return []
    return bbox


def horizontal_overlap(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    overlap = max(0.0, right - left)
    return overlap / max(1.0, min(a[2] - a[0], b[2] - b[0]))


def vertical_gap(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0


def layout_related(caption: dict[str, Any], target: dict[str, Any]) -> bool:
    caption_bbox = parse_bbox(caption.get("bbox"))
    target_bbox = parse_bbox(target.get("bbox"))
    if not caption_bbox or not target_bbox:
        return True
    try:
        page_height = float(caption.get("page_height") or target.get("page_height") or 800.0)
    except (TypeError, ValueError):
        page_height = 800.0
    return horizontal_overlap(caption_bbox, target_bbox) >= 0.03 and vertical_gap(caption_bbox, target_bbox) <= page_height * 0.45


def build_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_doc_page: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_doc_section: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_doc_kind_ref: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    page_nodes: dict[tuple[str, str], str] = {}

    for node in nodes:
        doc_id = clean_text(node.get("doc_id"))
        page = str(node.get("page", ""))
        node_type = clean_text(node.get("node_type"))
        section = clean_text(node.get("section"))
        by_doc_page[(doc_id, page)].append(node)
        if section and node_type != "page":
            by_doc_section[(doc_id, section)].append(node)
        if node_type == "page":
            page_nodes[(doc_id, page)] = node.get("node_id", "")
        for kind, ref_no in extract_refs(node.get("content", "") + " " + node.get("source_ref", "")):
            if node_type in {kind, "caption"}:
                by_doc_kind_ref[(doc_id, kind, ref_no)].append(node)

    edges: list[dict[str, Any]] = []
    for node in nodes:
        node_id = node.get("node_id", "")
        parent_id = clean_text(node.get("parent_chunk_id"))
        previous_id = clean_text(node.get("previous_node_id"))
        if parent_id:
            add_edge(edges, parent_id, node_id, "parent_section", 0.45)
        if previous_id:
            add_edge(edges, previous_id, node_id, "chunk_sequence", 0.08)

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
                related_figures = [figure for figure in figures if layout_related(caption, figure)] or figures
                for figure in related_figures:
                    add_edge(edges, caption["node_id"], figure["node_id"], "figure_caption", 1.2)
            if re.search(r"^(表|Table)", text, re.I):
                related_tables = [table for table in tables if layout_related(caption, table)] or tables
                for table in related_tables:
                    add_edge(edges, caption["node_id"], table["node_id"], "table_caption", 1.2)

    for (_doc_id, _section), section_group in by_doc_section.items():
        title_nodes = [node for node in section_group if node.get("node_type") == "title"]
        content_nodes = [node for node in section_group if node.get("node_type") != "title"]
        for title in title_nodes[:2]:
            for node in content_nodes[:30]:
                add_edge(edges, title.get("node_id", ""), node.get("node_id", ""), "section_title", 0.35)
        for left, right in zip(content_nodes, content_nodes[1:]):
            add_edge(edges, left.get("node_id", ""), right.get("node_id", ""), "same_section", 0.12)

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
