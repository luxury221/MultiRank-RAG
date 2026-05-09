from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline_common import DEFAULT_NODES, OUTPUT_DIR, clean_text, read_jsonl, resolve_path, write_csv


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def most_common(counter: Counter[str], default: str = "") -> str:
    return counter.most_common(1)[0][0] if counter else default


def parse_nodes(pdf_dir: Path, chunk_template: str, chunk_size: int) -> list[dict[str, Any]]:
    parse_pdf = load_script_module("parse_pdf_report", SCRIPTS_DIR / "01_parse_pdf.py")
    nodes: list[dict[str, Any]] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        nodes.extend(parse_pdf.pdf_to_nodes(pdf_path, chunk_size=chunk_size, chunk_template=chunk_template))
    return nodes


def edge_counts_for_nodes(nodes: list[dict[str, Any]]) -> tuple[Counter[str], set[str]]:
    build_graph = load_script_module("build_graph_report", SCRIPTS_DIR / "02_build_graph.py")
    edges = build_graph.build_edges(nodes)
    edge_counts = Counter(clean_text(edge.get("edge_type")) for edge in edges)
    connected: set[str] = set()
    for edge in edges:
        connected.add(clean_text(edge.get("source_id")))
        connected.add(clean_text(edge.get("target_id")))
    connected.discard("")
    return edge_counts, connected


def warn_for_doc(row: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if int(row["section_count"]) == 0:
        warnings.append("no_sections")
    if int(row["visual_nodes"]) == 0:
        warnings.append("no_visual_nodes")
    if float(row["isolated_rate"]) > 0.35:
        warnings.append("high_isolated_rate")
    if int(row["caption"]) > 0 and float(row["figure_caption_pair_rate"]) < 0.25 and float(row["table_caption_pair_rate"]) < 0.25:
        warnings.append("low_caption_pair_rate")
    if int(row["long_chunks"]) > max(2, int(row["evidence_nodes"]) * 0.08):
        warnings.append("many_long_chunks")
    return warnings


def build_report(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        nodes_by_doc[clean_text(node.get("doc_id")) or "unknown"].append(node)

    rows: list[dict[str, Any]] = []
    for doc_id, doc_nodes in sorted(nodes_by_doc.items()):
        type_counts = Counter(clean_text(node.get("node_type")) or "unknown" for node in doc_nodes)
        domain_counts = Counter(clean_text(node.get("paper_domain")) for node in doc_nodes if clean_text(node.get("paper_domain")))
        template_counts = Counter(clean_text(node.get("chunk_template")) for node in doc_nodes if clean_text(node.get("chunk_template")))
        structure_counts = Counter(
            clean_text(node.get("structure_type"))
            for node in doc_nodes
            if clean_text(node.get("structure_type")) not in {"", "page", "section_title", "caption"}
        )
        sections = sorted({clean_text(node.get("section")) for node in doc_nodes if clean_text(node.get("section"))})
        evidence_nodes = [node for node in doc_nodes if clean_text(node.get("node_type")) != "page"]
        content_lengths = [len(clean_text(node.get("content"))) for node in evidence_nodes]
        edge_counts, connected = edge_counts_for_nodes(doc_nodes)

        isolated = [
            node
            for node in evidence_nodes
            if clean_text(node.get("node_id")) and clean_text(node.get("node_id")) not in connected
        ]
        parented = [node for node in evidence_nodes if clean_text(node.get("parent_chunk_id"))]
        contexted = [
            node
            for node in evidence_nodes
            if clean_text(node.get("previous_node_id")) or clean_text(node.get("next_node_id"))
        ]
        explicit_ref_nodes = [node for node in evidence_nodes if clean_text(node.get("explicit_refs"))]
        bbox_nodes = [node for node in evidence_nodes if clean_text(node.get("bbox"))]
        layout_nodes = [node for node in evidence_nodes if clean_text(node.get("layout_parser"))]
        two_column_pages = {
            str(node.get("page", ""))
            for node in doc_nodes
            if str(node.get("layout_column_count", "")) == "2"
        }
        filtered_header_footer_blocks = 0
        for node in doc_nodes:
            if clean_text(node.get("node_type")) != "page":
                continue
            try:
                filtered_header_footer_blocks += int(float(node.get("filtered_header_footer_blocks") or 0))
            except (TypeError, ValueError):
                pass
        structured_tables = [
            node
            for node in evidence_nodes
            if clean_text(node.get("layout_parser")) == "pdfplumber" or clean_text(node.get("chunk_strategy")) == "structured_table"
        ]
        visual_nodes = type_counts["table"] + type_counts["figure"] + type_counts["caption"]
        avg_chars = statistics.mean(content_lengths) if content_lengths else 0.0
        median_chars = statistics.median(content_lengths) if content_lengths else 0.0
        long_chunks = sum(1 for length in content_lengths if length > 1200)
        short_chunks = sum(1 for length in content_lengths if length < 80)

        row: dict[str, Any] = {
            "doc_id": doc_id,
            "paper_domain": most_common(domain_counts, "unknown"),
            "chunk_template": most_common(template_counts, "unknown"),
            "requested_chunk_template": clean_text(doc_nodes[0].get("requested_chunk_template")) if doc_nodes else "",
            "auto_chunk_template": clean_text(doc_nodes[0].get("auto_chunk_template")) if doc_nodes else "",
            "auto_domain_confidence": clean_text(doc_nodes[0].get("auto_domain_confidence")) if doc_nodes else "",
            "domain_candidates": clean_text(doc_nodes[0].get("domain_candidates")) if doc_nodes else "",
            "total_nodes": len(doc_nodes),
            "evidence_nodes": len(evidence_nodes),
            "pages": type_counts["page"],
            "title": type_counts["title"],
            "text": type_counts["text"],
            "table": type_counts["table"],
            "figure": type_counts["figure"],
            "caption": type_counts["caption"],
            "equation": type_counts["equation"],
            "visual_nodes": visual_nodes,
            "section_count": len(sections),
            "section_list": ";".join(sections[:16]),
            "avg_chunk_chars": round(avg_chars, 2),
            "median_chunk_chars": round(float(median_chars), 2),
            "long_chunks": long_chunks,
            "short_chunks": short_chunks,
            "parented_chunks": len(parented),
            "parented_rate": round(len(parented) / max(1, len(evidence_nodes)), 4),
            "contexted_chunks": len(contexted),
            "contexted_rate": round(len(contexted) / max(1, len(evidence_nodes)), 4),
            "explicit_ref_nodes": len(explicit_ref_nodes),
            "bbox_nodes": len(bbox_nodes),
            "bbox_rate": round(len(bbox_nodes) / max(1, len(evidence_nodes)), 4),
            "layout_nodes": len(layout_nodes),
            "layout_node_rate": round(len(layout_nodes) / max(1, len(evidence_nodes)), 4),
            "two_column_pages": len(two_column_pages),
            "filtered_header_footer_blocks": filtered_header_footer_blocks,
            "structured_tables": len(structured_tables),
            "structure_blocks": sum(structure_counts.values()),
            "structure_types": ";".join(f"{name}:{count}" for name, count in structure_counts.most_common(10)),
            "section_title_edges": edge_counts["section_title"],
            "same_section_edges": edge_counts["same_section"],
            "parent_section_edges": edge_counts["parent_section"],
            "chunk_sequence_edges": edge_counts["chunk_sequence"],
            "table_caption_edges": edge_counts["table_caption"],
            "figure_caption_edges": edge_counts["figure_caption"],
            "table_caption_pair_rate": round(edge_counts["table_caption"] / max(1, type_counts["table"]), 4),
            "figure_caption_pair_rate": round(edge_counts["figure_caption"] / max(1, type_counts["figure"]), 4),
            "isolated_chunks": len(isolated),
            "isolated_rate": round(len(isolated) / max(1, len(evidence_nodes)), 4),
        }
        row["warnings"] = ";".join(warn_for_doc(row))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Report paper-aware chunking quality by document.")
    parser.add_argument("--nodes", default=str(DEFAULT_NODES.relative_to(ROOT)))
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--parse", action="store_true", help="Parse PDFs directly before reporting.")
    parser.add_argument(
        "--chunk-template",
        choices=["auto", "general", "ai", "math", "finance", "medical"],
        default="auto",
    )
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--output", default=str((OUTPUT_DIR / "metrics" / "chunk_quality.csv").relative_to(ROOT)))
    parser.add_argument("--json-output", default=str((OUTPUT_DIR / "metrics" / "chunk_quality.json").relative_to(ROOT)))
    args = parser.parse_args()

    if args.parse:
        nodes = parse_nodes(resolve_path(args.pdf_dir), args.chunk_template, args.chunk_size)
    else:
        nodes = read_jsonl(args.nodes)
        if not nodes:
            nodes = parse_nodes(resolve_path(args.pdf_dir), args.chunk_template, args.chunk_size)

    rows = build_report(nodes)
    write_csv(args.output, rows)
    json_path = resolve_path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} chunk quality rows to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
