from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SCRIPTS_DIR = ROOT / "scripts"

DEFAULT_NODES = OUTPUT_DIR / "parsed" / "nodes.jsonl"
DEFAULT_EDGES = OUTPUT_DIR / "parsed" / "edges.jsonl"
LEGACY_NODES = OUTPUT_DIR / "nodes.jsonl"
LEGACY_EDGES = OUTPUT_DIR / "edges.jsonl"
DEFAULT_QUESTIONS = DATA_DIR / "questions.csv"
DEFAULT_CANDIDATES = OUTPUT_DIR / "rankings" / "candidates.csv"
DEFAULT_RANKINGS = OUTPUT_DIR / "rankings" / "reranked.csv"
DEFAULT_SUMMARY = OUTPUT_DIR / "metrics" / "summary_metrics.csv"

NODE_FIELDS = [
    "node_id",
    "doc_id",
    "page",
    "node_type",
    "content",
    "source_ref",
]

EDGE_FIELDS = ["source_id", "target_id", "edge_type", "weight"]

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


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR / "pdfs",
        DATA_DIR / "sample",
        OUTPUT_DIR / "parsed",
        OUTPUT_DIR / "embeddings",
        OUTPUT_DIR / "rankings",
        OUTPUT_DIR / "metrics",
        OUTPUT_DIR / "evidence_cards",
        OUTPUT_DIR / "graphrag",
        OUTPUT_DIR / "cases",
        ROOT / "docs",
        ROOT / "demo",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = resolve_path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    file_path = resolve_path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    rows = list(rows)
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[\ud800-\udfff]", "", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_doc_id(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"\s+", "_", stem.strip())
    stem = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", stem)
    return stem.strip("_") or "doc"


def preview(text: Any, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", clean_text(text))
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def split_multi(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[;；,，|]\s*", text)
    return [part.strip() for part in parts if part.strip()]


def split_ints(value: Any) -> set[int]:
    ints: set[int] = set()
    for item in split_multi(value):
        match = re.search(r"\d+", item)
        if match:
            ints.add(int(match.group(0)))
    return ints


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def node_type_prefix(node_type: str) -> str:
    return {
        "text": "T",
        "table": "TB",
        "figure": "F",
        "page": "P",
        "caption": "C",
        "title": "H",
        "equation": "E",
    }.get(node_type, "N")


def make_node_id(node_type: str, doc_id: str, page: int, index: int | None = None) -> str:
    prefix = node_type_prefix(node_type)
    if node_type == "page":
        return f"{prefix}_{doc_id}_{page}"
    return f"{prefix}_{doc_id}_{page}_{index or 1}"


def copy_jsonl_alias(source: str | Path, alias: str | Path) -> None:
    rows = read_jsonl(source)
    write_jsonl(alias, rows)


def empty_questions_template(path: str | Path = DEFAULT_QUESTIONS) -> None:
    write_csv(path, [], QUESTION_FIELDS)


__all__ = [
    "ROOT",
    "DATA_DIR",
    "OUTPUT_DIR",
    "SCRIPTS_DIR",
    "DEFAULT_NODES",
    "DEFAULT_EDGES",
    "LEGACY_NODES",
    "LEGACY_EDGES",
    "DEFAULT_QUESTIONS",
    "DEFAULT_CANDIDATES",
    "DEFAULT_RANKINGS",
    "DEFAULT_SUMMARY",
    "NODE_FIELDS",
    "EDGE_FIELDS",
    "QUESTION_FIELDS",
    "ensure_project_dirs",
    "resolve_path",
    "read_jsonl",
    "write_jsonl",
    "read_csv",
    "write_csv",
    "clean_text",
    "normalize_doc_id",
    "preview",
    "split_multi",
    "split_ints",
    "as_float",
    "node_type_prefix",
    "make_node_id",
    "copy_jsonl_alias",
    "empty_questions_template",
]
