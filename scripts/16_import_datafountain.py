from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from pipeline_common import (
    DEFAULT_EDGES,
    DEFAULT_NODES,
    DEFAULT_QUESTIONS,
    LEGACY_EDGES,
    LEGACY_NODES,
    QUESTION_FIELDS,
    clean_text,
    copy_jsonl_alias,
    ensure_project_dirs,
    normalize_doc_id,
    preview,
    resolve_path,
    write_csv,
    write_jsonl,
)


DEFAULT_ROOT = "DataFountain/KownledgeBase"
DEFAULT_OUTPUT_DIR = "outputs/datafountain_1165"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def is_zone_file(path: Path) -> bool:
    return "Zone.Identifier" in path.name


def discover_dataset_dirs(root: Path) -> tuple[Path, Path]:
    manual_dirs = [path for path in root.iterdir() if path.is_dir() and not is_zone_file(path)]
    if not manual_dirs:
        raise FileNotFoundError(f"No manual directory found under {root}")
    manual_dir = manual_dirs[0]
    image_dirs = [path for path in manual_dir.iterdir() if path.is_dir() and not is_zone_file(path)]
    if not image_dirs:
        raise FileNotFoundError(f"No image directory found under {manual_dir}")
    return manual_dir, image_dirs[0]


def image_index(image_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in image_dir.iterdir():
        if not path.is_file() or is_zone_file(path):
            continue
        if path.suffix.lower() in IMAGE_EXTS:
            indexed.setdefault(path.stem, path)
    return indexed


def read_questions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def normalize_question_text(value: Any) -> str:
    text = clean_text(value)
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip().strip(",").strip()
        line = line.strip('"').strip("'").strip()
        if line:
            parts.append(line)
    return clean_text(" ".join(parts))


def convert_questions(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for row in rows:
        raw_id = clean_text(row.get("id"))
        question = normalize_question_text(row.get("question"))
        if not raw_id or not question:
            continue
        converted.append(
            {
                "question_id": f"DF_{raw_id}",
                "doc_id": "",
                "question": question,
                "answer": "",
                "question_type": "DataFountain customer-service knowledge QA",
                "gold_node_ids": "",
                "gold_pages": "",
                "gold_modalities": "text;figure",
                "evidence_note": "Imported from DataFountain competition 1165 public questions.",
            }
        )
    return converted


def read_manual_payload(path: Path) -> tuple[str, list[str]]:
    raw = path.read_text(encoding="utf-8-sig")
    payloads: list[Any] = []

    def parse_payload(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)

    try:
        payloads = [parse_payload(raw)]
    except (json.JSONDecodeError, SyntaxError, ValueError):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payloads.append(parse_payload(line))
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue

    texts: list[str] = []
    image_ids: list[str] = []
    for payload in payloads:
        if isinstance(payload, list) and len(payload) >= 2:
            texts.append(clean_text(payload[0]))
            image_ids.extend(clean_text(item) for item in payload[1] if clean_text(item))
        elif isinstance(payload, str):
            texts.append(clean_text(payload))
    if texts:
        return "\n\n".join(texts), image_ids
    return clean_text(raw), []


def manual_files(manual_dir: Path, include_summary: bool) -> list[Path]:
    files: list[Path] = []
    for path in sorted(manual_dir.glob("*.txt")):
        if is_zone_file(path):
            continue
        if not include_summary and "汇总" in path.stem:
            continue
        files.append(path)
    return files


def add_edge(edges: list[dict[str, Any]], source: str, target: str, edge_type: str, weight: float) -> None:
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


def clean_heading(line: str) -> str:
    line = re.sub(r"^#+", "", clean_text(line)).strip()
    line = re.sub(r"\s+", " ", line)
    return line


def split_long_text(text: str, chunk_size: int) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    pieces: list[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            for start in range(0, len(paragraph), chunk_size):
                pieces.append(paragraph[start : start + chunk_size].strip())
            continue
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) > chunk_size and current:
            pieces.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(resolve_path(".")).as_posix()
    except ValueError:
        return path.as_posix()


class ManualBuilder:
    def __init__(
        self,
        path: Path,
        text: str,
        image_ids: list[str],
        images: dict[str, Path],
        chunk_size: int,
    ) -> None:
        self.path = path
        self.text = text
        self.image_ids = image_ids
        self.images = images
        self.chunk_size = chunk_size
        self.doc_id = normalize_doc_id(path.stem)
        self.doc_title = path.stem
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.section = self.doc_title
        self.section_title_id = ""
        self.previous_node_id = ""
        self.position = 0
        self.node_counter = 0
        self.pic_index = 0
        self.image_linked = 0
        self.image_missing = 0

    def next_id(self, prefix: str) -> str:
        self.node_counter += 1
        return f"{prefix}_{self.doc_id}_{self.node_counter}"

    def next_position(self) -> int:
        self.position += 1
        return self.position

    def add_sequence_edges(self, node: dict[str, Any]) -> None:
        if self.previous_node_id:
            node["previous_node_id"] = self.previous_node_id
            add_edge(self.edges, self.previous_node_id, node["node_id"], "chunk_sequence", 0.08)
        if self.section_title_id and node["node_id"] != self.section_title_id:
            node["parent_chunk_id"] = self.section_title_id
            add_edge(self.edges, self.section_title_id, node["node_id"], "section_title", 0.35)
            add_edge(self.edges, self.section_title_id, node["node_id"], "parent_section", 0.45)
        self.previous_node_id = clean_text(node.get("node_id"))

    def base_node(self, node_type: str, content: str, source_ref: str) -> dict[str, Any]:
        return {
            "node_id": self.next_id({"title": "H", "figure": "F"}.get(node_type, "T")),
            "doc_id": self.doc_id,
            "page": self.next_position(),
            "node_type": node_type,
            "content": clean_text(content),
            "source_ref": source_ref,
            "section": self.section,
            "paper_domain": "datafountain_customer_service",
            "chunk_template": "manual_qa",
            "requested_chunk_template": "manual_qa",
            "auto_chunk_template": "manual_qa",
            "chunk_level": "section",
            "chunk_strategy": "manual_heading_pic",
            "structure_type": "manual",
        }

    def add_title(self, heading: str) -> None:
        heading = clean_heading(heading)
        if not heading:
            return
        self.section = heading
        node = self.base_node("title", heading, f"{self.doc_title} / {heading}")
        node["structure_type"] = "section_title"
        self.section_title_id = node["node_id"]
        self.add_sequence_edges(node)
        self.nodes.append(node)

    def add_text(self, text: str) -> None:
        for chunk in split_long_text(text, self.chunk_size):
            node = self.base_node(
                "text",
                chunk,
                f"{self.doc_title} / {self.section} / segment {self.position + 1}",
            )
            self.add_sequence_edges(node)
            self.nodes.append(node)

    def add_figure(self) -> None:
        self.pic_index += 1
        image_id = self.image_ids[self.pic_index - 1] if self.pic_index <= len(self.image_ids) else ""
        image_path = self.images.get(image_id)
        content = (
            f"Manual illustration {self.pic_index}. "
            f"Document: {self.doc_title}. Section: {self.section}. Image id: {image_id}."
        )
        node = self.base_node(
            "figure",
            content,
            f"{self.doc_title} / {self.section} / illustration {self.pic_index} / {image_id}",
        )
        node["structure_type"] = "manual_illustration"
        node["layout_role"] = "illustration"
        node["visual_title"] = f"{self.doc_title} illustration {self.pic_index}"
        node["visual_type"] = "product manual illustration"
        node["visual_summary"] = f"Illustration linked to section '{self.section}' in {self.doc_title}."
        node["image_id"] = image_id
        if image_path:
            rel = rel_path(image_path)
            node["crop_image_path"] = rel
            node["page_image_path"] = rel
            self.image_linked += 1
        else:
            node["missing_image_id"] = image_id
            self.image_missing += 1
        self.add_sequence_edges(node)
        if self.previous_node_id and self.previous_node_id != node["node_id"]:
            add_edge(self.edges, self.previous_node_id, node["node_id"], "text_ref_figure", 1.0)
        self.nodes.append(node)

    def enrich_figure_context(self) -> None:
        for index, node in enumerate(self.nodes):
            if node.get("node_type") != "figure":
                continue
            prev_text = ""
            next_text = ""
            prev_id = ""
            next_id = ""
            for left in reversed(self.nodes[:index]):
                if left.get("node_type") in {"text", "title"}:
                    prev_text = preview(left.get("content", ""), 220)
                    prev_id = clean_text(left.get("node_id"))
                    break
            for right in self.nodes[index + 1 :]:
                if right.get("node_type") in {"text", "title"}:
                    next_text = preview(right.get("content", ""), 220)
                    next_id = clean_text(right.get("node_id"))
                    break
            node["previous_chunk_preview"] = prev_text
            node["next_chunk_preview"] = next_text
            if prev_id:
                add_edge(self.edges, prev_id, node["node_id"], "text_ref_figure", 1.0)
            if next_id:
                add_edge(self.edges, next_id, node["node_id"], "text_ref_figure", 0.75)
            context = clean_text(
                f"{node.get('content', '')}\n"
                f"Nearby text before: {prev_text}\n"
                f"Nearby text after: {next_text}"
            )
            node["content"] = context

    def build(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                self.add_text("\n".join(buffer))
                buffer.clear()

        for part in re.split(r"(<PIC>)", self.text):
            if part == "<PIC>":
                flush()
                self.add_figure()
                continue
            for line in part.splitlines():
                line = clean_text(line)
                if not line:
                    continue
                if line.startswith("#"):
                    flush()
                    self.add_title(line)
                else:
                    buffer.append(line)
                    if sum(len(item) for item in buffer) >= self.chunk_size:
                        flush()
        flush()
        self.enrich_figure_context()
        return (
            self.nodes,
            dedupe_edges(self.edges),
            {"linked": self.image_linked, "missing": self.image_missing},
        )


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        source = clean_text(edge.get("source_id"))
        target = clean_text(edge.get("target_id"))
        edge_type = clean_text(edge.get("edge_type")) or "related"
        if not source or not target or source == target:
            continue
        key_nodes = tuple(sorted([source, target]))
        key = (key_nodes[0], key_nodes[1], edge_type)
        weight = float(edge.get("weight") or 1.0)
        if key in dedup:
            dedup[key]["weight"] = round(float(dedup[key]["weight"]) + weight, 4)
        else:
            dedup[key] = {
                "source_id": source,
                "target_id": target,
                "edge_type": edge_type,
                "weight": round(weight, 4),
            }
    return list(dedup.values())


def import_manuals(
    manual_dir: Path,
    image_dir: Path,
    chunk_size: int,
    include_summary: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    images = image_index(image_dir)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    summary = {
        "manuals": 0,
        "images_available": len(images),
        "images_linked": 0,
        "images_missing": 0,
    }
    for path in manual_files(manual_dir, include_summary):
        text, image_ids = read_manual_payload(path)
        builder = ManualBuilder(path, text, image_ids, images, chunk_size)
        doc_nodes, doc_edges, stats = builder.build()
        nodes.extend(doc_nodes)
        edges.extend(doc_edges)
        summary["manuals"] += 1
        summary["images_linked"] += stats["linked"]
        summary["images_missing"] += stats["missing"]
    return nodes, dedupe_edges(edges), summary


def activate_outputs(output_dir: Path) -> None:
    questions = output_dir / "questions.csv"
    nodes = output_dir / "nodes.jsonl"
    edges = output_dir / "edges.jsonl"
    if questions.exists():
        shutil.copy2(questions, DEFAULT_QUESTIONS)
    if nodes.exists():
        shutil.copy2(nodes, DEFAULT_NODES)
        copy_jsonl_alias(DEFAULT_NODES, LEGACY_NODES)
    if edges.exists():
        shutil.copy2(edges, DEFAULT_EDGES)
        copy_jsonl_alias(DEFAULT_EDGES, LEGACY_EDGES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import DataFountain competition 1165 manuals into the evidence-node format."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--include-summary", action="store_true")
    parser.add_argument("--activate", action="store_true", help="Also copy outputs into data/questions.csv and outputs/parsed.")
    args = parser.parse_args()

    ensure_project_dirs()
    root = resolve_path(args.root)
    output_dir = resolve_path(args.output_dir)
    manual_dir, image_dir = discover_dataset_dirs(root)

    nodes, edges, manual_summary = import_manuals(
        manual_dir=manual_dir,
        image_dir=image_dir,
        chunk_size=args.chunk_size,
        include_summary=args.include_summary,
    )
    question_rows = convert_questions(read_questions(root / "question_public.csv"))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "nodes.jsonl", nodes)
    write_jsonl(output_dir / "edges.jsonl", edges)
    write_csv(output_dir / "questions.csv", question_rows, QUESTION_FIELDS)

    summary = {
        **manual_summary,
        "questions": len(question_rows),
        "nodes": len(nodes),
        "edges": len(edges),
        "manual_dir": rel_path(manual_dir),
        "image_dir": rel_path(image_dir),
        "output_dir": rel_path(output_dir),
        "activated": bool(args.activate),
    }
    (output_dir / "import_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.activate:
        activate_outputs(output_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
