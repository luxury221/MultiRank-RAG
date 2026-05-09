from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline_common import DEFAULT_NODES, clean_text, normalize_doc_id, read_jsonl, resolve_path, write_csv


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

COLORS = {
    "title": (37, 99, 235),
    "text": (71, 85, 105),
    "table": (22, 163, 74),
    "figure": (147, 51, 234),
    "caption": (8, 145, 178),
    "equation": (220, 38, 38),
    "page": (100, 116, 139),
}


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def safe_stem(value: str) -> str:
    import re

    value = clean_text(value)
    value = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", value)
    return value.strip("_") or "doc"


def rel_output(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_pdf_dir(pdf_dir: Path, chunk_template: str, chunk_size: int) -> list[dict[str, Any]]:
    parse_pdf = load_script_module("parse_pdf_visualize", SCRIPTS_DIR / "01_parse_pdf.py")
    nodes: list[dict[str, Any]] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        nodes.extend(parse_pdf.pdf_to_nodes(pdf_path, chunk_size=chunk_size, chunk_template=chunk_template))
    return nodes


def find_pdf_by_doc_id(pdf_dir: Path) -> dict[str, Path]:
    return {normalize_doc_id(path.name): path for path in sorted(pdf_dir.glob("*.pdf"))}


def group_nodes(nodes: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for node in nodes:
        bbox = parse_bbox(node.get("bbox"))
        if not bbox:
            continue
        doc_id = clean_text(node.get("doc_id"))
        try:
            page = int(float(node.get("page") or 0))
        except (TypeError, ValueError):
            continue
        if doc_id and page > 0:
            grouped[doc_id][page].append(node)
    return grouped


def draw_label(draw: Any, xy: tuple[int, int], label: str, color: tuple[int, int, int]) -> None:
    x, y = xy
    label = label[:48]
    try:
        bbox = draw.textbbox((x, y), label)
        pad = 3
        draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=color)
        draw.text((x, y), label, fill=(255, 255, 255))
    except Exception:
        draw.text((x, y), label, fill=color)


def render_doc(
    pdf_path: Path,
    pages: dict[int, list[dict[str, Any]]],
    output_root: Path,
    dpi: int,
    max_pages: int,
    max_boxes: int,
) -> list[dict[str, Any]]:
    import fitz
    from PIL import Image, ImageDraw

    rows: list[dict[str, Any]] = []
    doc_id = normalize_doc_id(pdf_path.name)
    out_dir = output_root / safe_stem(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(str(pdf_path)) as doc:
        page_numbers = sorted(page for page in pages if 1 <= page <= len(doc))
        if max_pages > 0:
            page_numbers = page_numbers[:max_pages]
        for page_no in page_numbers:
            page = doc[page_no - 1]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(image, "RGBA")
            page_nodes = pages[page_no][:max_boxes]
            for node in page_nodes:
                bbox = parse_bbox(node.get("bbox"))
                if not bbox:
                    continue
                color = COLORS.get(clean_text(node.get("node_type")), (100, 116, 139))
                scaled = tuple(int(round(value * zoom)) for value in bbox)
                draw.rectangle(scaled, outline=(*color, 230), width=3)
                label = f"{node.get('node_type', '')}:{node.get('node_id', '')}"
                draw_label(draw, (scaled[0] + 3, max(0, scaled[1] - 16)), label, color)

            out_path = out_dir / f"page_{page_no:03d}.png"
            image.save(out_path)
            rows.append(
                {
                    "doc_id": doc_id,
                    "pdf": pdf_path.name,
                    "page": page_no,
                    "box_count": len(page_nodes),
                    "image_path": rel_output(out_path),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PDF pages with parsed layout bboxes overlaid.")
    parser.add_argument("--nodes", default=str(DEFAULT_NODES.relative_to(ROOT)))
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--parse", action="store_true", help="Parse PDFs first instead of reading --nodes.")
    parser.add_argument(
        "--chunk-template",
        choices=["auto", "general", "ai", "math", "finance", "medical"],
        default="auto",
    )
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--output-dir", default="outputs/layout_debug")
    parser.add_argument("--report", default="outputs/metrics/layout_bbox_debug.csv")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--max-pages-per-doc", type=int, default=4)
    parser.add_argument("--max-boxes-per-page", type=int, default=90)
    args = parser.parse_args()

    pdf_dir = resolve_path(args.pdf_dir)
    nodes = parse_pdf_dir(pdf_dir, args.chunk_template, args.chunk_size) if args.parse else read_jsonl(args.nodes)
    if not nodes:
        nodes = parse_pdf_dir(pdf_dir, args.chunk_template, args.chunk_size)

    pdf_by_doc = find_pdf_by_doc_id(pdf_dir)
    grouped = group_nodes(nodes)
    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for doc_id, pages in sorted(grouped.items()):
        pdf_path = pdf_by_doc.get(doc_id)
        if not pdf_path:
            continue
        rows.extend(
            render_doc(
                pdf_path,
                pages,
                output_root,
                args.dpi,
                args.max_pages_per_doc,
                args.max_boxes_per_page,
            )
        )

    write_csv(args.report, rows)
    print(f"Wrote {len(rows)} layout debug images to {output_root}")


if __name__ == "__main__":
    main()
