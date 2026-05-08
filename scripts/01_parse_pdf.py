from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pipeline_common import (
    DEFAULT_NODES,
    LEGACY_NODES,
    NODE_FIELDS,
    clean_text,
    copy_jsonl_alias,
    empty_questions_template,
    ensure_project_dirs,
    make_node_id,
    normalize_doc_id,
    read_csv,
    read_jsonl,
    resolve_path,
    write_jsonl,
)


CAPTION_RE = re.compile(r"^\s*((图|圖|Fig\.?|Figure|表|Table)\s*[\w\-\.]+[:：]?.*)", re.I)
TABLE_HINT_RE = re.compile(r"(\d+\s+\d+\s+\d+)|(\|.+\|)|(指标|方法|准确率|召回率|F1|Accuracy|Recall)")


def extract_pages_with_python(pdf_path: Path) -> list[str]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return [clean_text(page.extract_text() or "") for page in reader.pages]
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        return [clean_text(page.extract_text() or "") for page in reader.pages]
    except Exception:
        pass

    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        return [clean_text(page.get_text("text")) for page in doc]
    except Exception:
        pass

    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            return [clean_text(page.extract_text() or "") for page in pdf.pages]
    except Exception:
        pass

    return []


def extract_pages_with_pdftotext(pdf_path: Path) -> list[str]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.txt"
            subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), str(out)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            text = out.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    pages = [clean_text(page) for page in text.split("\f")]
    return [page for page in pages if page]


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    pages = extract_pages_with_python(pdf_path)
    if pages:
        return pages
    return extract_pages_with_pdftotext(pdf_path)


def split_blocks(text: str, chunk_size: int = 900) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) <= 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        blocks = []
        current: list[str] = []
        for line in lines:
            current.append(line)
            if len(" ".join(current)) >= chunk_size or CAPTION_RE.search(line):
                blocks.append("\n".join(current))
                current = []
        if current:
            blocks.append("\n".join(current))
    chunks: list[str] = []
    for block in blocks:
        if len(block) <= chunk_size:
            chunks.append(block)
            continue
        sentences = re.split(r"(?<=[。！？.!?])\s+", block)
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > chunk_size and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current.strip())
    return chunks


def node_type_for_block(block: str) -> str:
    if CAPTION_RE.search(block):
        return "caption"
    if TABLE_HINT_RE.search(block) and len(block.splitlines()) >= 2:
        return "table"
    return "text"


def content_list_to_nodes(content_list_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(content_list_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("content_list") or payload.get("items") or []
        source_name = payload.get("file_path") or payload.get("doc_id") or content_list_path.stem
    else:
        items = payload
        source_name = content_list_path.stem
    doc_id = normalize_doc_id(str(source_name))
    counters: dict[tuple[str, int], int] = {}
    nodes: list[dict[str, Any]] = []
    page_seen: set[int] = set()
    for item in items:
        page = int(item.get("page_idx", item.get("page", 0))) + 1
        if page not in page_seen:
            page_seen.add(page)
            nodes.append(
                {
                    "node_id": make_node_id("page", doc_id, page),
                    "doc_id": doc_id,
                    "page": page,
                    "node_type": "page",
                    "content": f"Page {page} of {doc_id}",
                    "source_ref": f"page {page}",
                }
            )
        raw_type = str(item.get("type", "text")).lower()
        node_type = {
            "image": "figure",
            "figure": "figure",
            "table": "table",
            "equation": "equation",
            "text": "text",
        }.get(raw_type, "text")
        parts = [
            item.get("text"),
            item.get("content"),
            item.get("table_body"),
            item.get("table_data"),
            " ".join(item.get("image_caption") or []),
            " ".join(item.get("table_caption") or []),
            item.get("latex"),
        ]
        content = clean_text("\n".join(str(part) for part in parts if part))
        if not content:
            continue
        counters[(node_type, page)] = counters.get((node_type, page), 0) + 1
        nodes.append(
            {
                "node_id": make_node_id(node_type, doc_id, page, counters[(node_type, page)]),
                "doc_id": doc_id,
                "page": page,
                "node_type": node_type,
                "content": content,
                "source_ref": f"page {page} {node_type}",
            }
        )
    return nodes


def pdf_to_nodes(pdf_path: Path, chunk_size: int = 900) -> list[dict[str, Any]]:
    doc_id = normalize_doc_id(pdf_path.name)
    pages = extract_pdf_pages(pdf_path)
    if not pages:
        pages = [
            (
                f"{pdf_path.name} could not be text-extracted in the current environment. "
                "Install pypdf/pdfplumber or export RAG-Anything content_list JSON, then rerun parsing."
            )
        ]

    nodes: list[dict[str, Any]] = []
    counters: dict[tuple[str, int], int] = {}
    for page_index, page_text in enumerate(pages, start=1):
        nodes.append(
            {
                "node_id": make_node_id("page", doc_id, page_index),
                "doc_id": doc_id,
                "page": page_index,
                "node_type": "page",
                "content": f"Page {page_index} of {doc_id}",
                "source_ref": f"{pdf_path.name} page {page_index}",
            }
        )
        blocks = split_blocks(page_text, chunk_size=chunk_size)
        if not blocks:
            blocks = [f"No extractable text on page {page_index} of {pdf_path.name}."]
        for block in blocks:
            node_type = node_type_for_block(block)
            counters[(node_type, page_index)] = counters.get((node_type, page_index), 0) + 1
            nodes.append(
                {
                    "node_id": make_node_id(node_type, doc_id, page_index, counters[(node_type, page_index)]),
                    "doc_id": doc_id,
                    "page": page_index,
                    "node_type": node_type,
                    "content": block,
                    "source_ref": f"{pdf_path.name} page {page_index}",
                }
            )
            caption_match = CAPTION_RE.search(block)
            if caption_match and re.search(r"^(图|圖|Fig|Figure)", caption_match.group(1), re.I):
                counters[("figure", page_index)] = counters.get(("figure", page_index), 0) + 1
                nodes.append(
                    {
                        "node_id": make_node_id("figure", doc_id, page_index, counters[("figure", page_index)]),
                        "doc_id": doc_id,
                        "page": page_index,
                        "node_type": "figure",
                        "content": f"Figure node inferred from caption: {block}",
                        "source_ref": f"{pdf_path.name} page {page_index} figure/caption",
                    }
                )
            if caption_match and re.search(r"^(表|Table)", caption_match.group(1), re.I):
                counters[("table", page_index)] = counters.get(("table", page_index), 0) + 1
                nodes.append(
                    {
                        "node_id": make_node_id("table", doc_id, page_index, counters[("table", page_index)]),
                        "doc_id": doc_id,
                        "page": page_index,
                        "node_type": "table",
                        "content": f"Table node inferred from caption: {block}",
                        "source_ref": f"{pdf_path.name} page {page_index} table/caption",
                    }
                )
    return nodes


def load_manual_nodes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    rows = read_csv(path)
    nodes: list[dict[str, Any]] = []
    counters: dict[tuple[str, int], int] = {}
    for row in rows:
        doc_id = normalize_doc_id(row.get("doc_id") or row.get("document") or "manual")
        page = int(row.get("page") or 1)
        node_type = (row.get("node_type") or "text").strip()
        counters[(node_type, page)] = counters.get((node_type, page), 0) + 1
        row = {field: row.get(field, "") for field in NODE_FIELDS}
        row["doc_id"] = doc_id
        row["page"] = page
        row["node_type"] = node_type
        row["node_id"] = row.get("node_id") or make_node_id(node_type, doc_id, page, counters[(node_type, page)])
        nodes.append(row)
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDFs or RAG-Anything content_list into evidence nodes.")
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--content-list", default="", help="Optional RAG-Anything content_list JSON.")
    parser.add_argument("--manual-nodes", default="data/manual_nodes.csv")
    parser.add_argument("--output", default=str(DEFAULT_NODES.relative_to(DEFAULT_NODES.parents[2])))
    parser.add_argument("--chunk-size", type=int, default=900)
    args = parser.parse_args()

    ensure_project_dirs()
    nodes: list[dict[str, Any]] = []

    if args.content_list:
        nodes.extend(content_list_to_nodes(resolve_path(args.content_list)))

    pdf_dir = resolve_path(args.pdf_dir)
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        nodes.extend(pdf_to_nodes(pdf_path, chunk_size=args.chunk_size))

    manual_path = resolve_path(args.manual_nodes)
    nodes.extend(load_manual_nodes(manual_path))

    write_jsonl(args.output, nodes)
    copy_jsonl_alias(args.output, LEGACY_NODES)
    if not resolve_path("data/questions.csv").exists():
        empty_questions_template()
    print(f"Wrote {len(nodes)} nodes to {resolve_path(args.output)}")
    if not nodes:
        print("No PDFs or manual nodes found. Put PDFs in data/pdfs/ or add data/manual_nodes.csv.")


if __name__ == "__main__":
    main()
