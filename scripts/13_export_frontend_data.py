#!/usr/bin/env python3
"""Export the latest RAG outputs for the React evidence UI."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
WEB_PUBLIC = ROOT / "web" / "public"
DETAILS_DIR = WEB_PUBLIC / "app-data" / "details"
RANKINGS_DIR = WEB_PUBLIC / "app-data" / "rankings"

QUESTIONS_CSV = ROOT / "data" / "questions.csv"
PDF_DIR = ROOT / "data" / "pdfs"
CHAIN_STEPS_CSV = ROOT / "outputs" / "evidence_chains" / "chain_steps.csv"
CARDS_MANIFEST_CSV = ROOT / "outputs" / "evidence_cards" / "cards_manifest.csv"
CARDS_QUALITY_CSV = ROOT / "outputs" / "evidence_cards" / "cards_quality_report.csv"
SUMMARY_METRICS_CSV = ROOT / "outputs" / "metrics" / "summary_metrics.csv"
RERANKED_CSV = ROOT / "outputs" / "rankings" / "reranked.csv"
NODES_JSONL = ROOT / "outputs" / "parsed" / "nodes.jsonl"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except ValueError:
        return default


def as_int(value: str | None, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except ValueError:
        return default


def split_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def normalize_rel(path_text: str | None) -> str:
    if not path_text:
        return ""
    return path_text.replace("\\", "/").strip().lstrip("./")


def public_url(path_text: str | None, copied: set[str]) -> str:
    rel = normalize_rel(path_text)
    if not rel:
        return ""

    src = ROOT / rel
    if not src.exists() or not src.is_file():
        return ""

    dst = WEB_PUBLIC / rel
    if rel not in copied:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.add(rel)
    return "/" + rel


def trim_text(value: str | None, max_len: int = 520) -> str:
    if not value:
        return ""
    clean = " ".join(value.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "…"


def clean_generated_assets() -> None:
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    for rel in [
        "app-data",
        "outputs/evidence_cards",
        "outputs/visual/crops",
        "outputs/visual/pages",
    ]:
        target = WEB_PUBLIC / rel
        if target.exists():
            shutil.rmtree(target)


def detail_url_for(question_id: str) -> str:
    return f"/app-data/details/{quote(question_id, safe='')}.json"


def ranking_url_for(question_id: str) -> str:
    return f"/app-data/rankings/{quote(question_id, safe='')}.json"


def build_pdf_index(nodes: list[dict[str, Any]], questions: list[dict[str, str]]) -> list[dict[str, Any]]:
    pages_by_doc: dict[str, int] = defaultdict(int)
    nodes_by_doc: Counter[str] = Counter()
    modalities_by_doc: dict[str, Counter[str]] = defaultdict(Counter)

    for node in nodes:
        doc_id = str(node.get("doc_id", ""))
        if not doc_id:
            continue
        page = int(node.get("page") or 0)
        pages_by_doc[doc_id] = max(pages_by_doc[doc_id], page)
        node_type = str(node.get("node_type", "unknown"))
        nodes_by_doc[doc_id] += 1
        modalities_by_doc[doc_id][node_type] += 1

    question_count = Counter(row["doc_id"] for row in questions if row.get("doc_id"))

    pdfs: list[dict[str, Any]] = []
    known_doc_ids = {path.stem for path in PDF_DIR.glob("*.pdf")}
    known_doc_ids.update(question_count.keys())
    known_doc_ids.update(pages_by_doc.keys())

    for doc_id in sorted(known_doc_ids):
        pdf_path = PDF_DIR / f"{doc_id}.pdf"
        pdfs.append(
            {
                "doc_id": doc_id,
                "file_name": pdf_path.name if pdf_path.exists() else f"{doc_id}.pdf",
                "pages": pages_by_doc.get(doc_id, 0),
                "question_count": question_count.get(doc_id, 0),
                "node_count": nodes_by_doc.get(doc_id, 0),
                "modalities": dict(sorted(modalities_by_doc.get(doc_id, Counter()).items())),
            }
        )
    return pdfs


def main() -> None:
    clean_generated_assets()
    copied_assets: set[str] = set()

    questions = read_csv(QUESTIONS_CSV)
    chain_rows = read_csv(CHAIN_STEPS_CSV)
    manifest_rows = read_csv(CARDS_MANIFEST_CSV)
    quality_rows = read_csv(CARDS_QUALITY_CSV)
    metrics_rows = read_csv(SUMMARY_METRICS_CSV)
    ranking_rows = read_csv(RERANKED_CSV)
    nodes = read_jsonl(NODES_JSONL)

    card_by_qid = {row.get("question_id", ""): row for row in manifest_rows}
    quality_by_qid = {row.get("question_id", ""): row for row in quality_rows}

    chains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chain_rows:
        qid = row.get("question_id", "")
        crop_url = public_url(row.get("crop_image_path"), copied_assets)
        page_url = public_url(row.get("page_image_path"), copied_assets)
        chains[qid].append(
            {
                "chain_step": as_int(row.get("chain_step")),
                "role": row.get("role", ""),
                "node_id": row.get("node_id", ""),
                "node_type": row.get("node_type", ""),
                "page": as_int(row.get("page")),
                "relation": row.get("relation", ""),
                "score": as_float(row.get("score")),
                "visual_score": as_float(row.get("visual_score")),
                "source_ref": row.get("source_ref", ""),
                "crop_url": crop_url,
                "page_url": page_url,
                "visual_summary": trim_text(row.get("visual_summary"), 420),
                "visual_caption": trim_text(row.get("visual_caption"), 420),
                "reason": trim_text(row.get("reason"), 320),
                "content_preview": trim_text(row.get("content_preview"), 520),
            }
        )

    for steps in chains.values():
        steps.sort(key=lambda item: item["chain_step"])

    rankings: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in ranking_rows:
        rank = as_int(row.get("rank"))
        if rank > 8:
            continue
        qid = row.get("question_id", "")
        method = row.get("method", "")
        rankings[qid][method].append(
            {
                "rank": rank,
                "node_id": row.get("node_id", ""),
                "node_type": row.get("node_type", ""),
                "page": as_int(row.get("page")),
                "score": as_float(row.get("score")),
                "sim_score": as_float(row.get("sim_score")),
                "bridge_score": as_float(row.get("bridge_score")),
                "ref_score": as_float(row.get("ref_score")),
                "visual_score": as_float(row.get("visual_score")),
                "has_visual_crop": as_int(row.get("has_visual_crop")),
                "has_visual_caption": as_int(row.get("has_visual_caption")),
                "source_ref": row.get("source_ref", ""),
                "content_preview": trim_text(row.get("content_preview"), 360),
                "crop_url": public_url(row.get("crop_image_path"), copied_assets),
            }
        )

    normalized_rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for qid, methods in rankings.items():
        normalized_rankings[qid] = {}
        for method, items in methods.items():
            normalized_rankings[qid][method] = sorted(items, key=lambda item: item["rank"])

    exported_questions: list[dict[str, Any]] = []
    for row in questions:
        qid = row.get("question_id", "")
        card = card_by_qid.get(qid, {})
        quality = quality_by_qid.get(qid, {})
        card_url = public_url(card.get("card_path") or quality.get("card_path"), copied_assets)
        chain_count = len(chains.get(qid, []))
        num_steps = as_int(card.get("num_steps"), chain_count) or chain_count
        exported_questions.append(
            {
                "question_id": qid,
                "doc_id": row.get("doc_id", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "question_type": row.get("question_type", ""),
                "gold_node_ids": split_values(row.get("gold_node_ids")),
                "gold_pages": split_values(row.get("gold_pages")),
                "gold_modalities": split_values(row.get("gold_modalities")),
                "evidence_note": row.get("evidence_note", ""),
                "card_url": card_url,
                "detail_url": detail_url_for(qid),
                "ranking_url": ranking_url_for(qid),
                "num_steps": num_steps,
                "quality_status": quality.get("status", ""),
                "quality_issues": split_values(quality.get("issues")),
                "visual_required": as_int(quality.get("visual_required")),
                "visual_node_steps": as_int(quality.get("visual_node_steps")),
                "crop_steps": as_int(quality.get("crop_steps")),
                "existing_crop_steps": as_int(quality.get("existing_crop_steps")),
                "qwen_caption_steps": as_int(quality.get("qwen_caption_steps")),
                "source_pages": split_values(quality.get("source_pages")),
            }
        )

    metrics: list[dict[str, Any]] = []
    for row in metrics_rows:
        metrics.append(
            {
                key: (as_float(value) if key != "method" else value)
                for key, value in row.items()
            }
        )

    pdfs = build_pdf_index(nodes, questions)
    parsed_pdf_count = len([pdf for pdf in pdfs if pdf.get("pages") or pdf.get("node_count")])

    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    RANKINGS_DIR.mkdir(parents=True, exist_ok=True)
    for question in exported_questions:
        qid = question["question_id"]
        detail_path = WEB_PUBLIC / question["detail_url"].lstrip("/")
        ranking_path = WEB_PUBLIC / question["ranking_url"].lstrip("/")
        detail_payload = {
            "question_id": qid,
            "steps": chains.get(qid, []),
            "rankings": {},
        }
        ranking_payload = {
            "question_id": qid,
            "rankings": normalized_rankings.get(qid, {}),
        }
        with detail_path.open("w", encoding="utf-8") as f:
            json.dump(detail_payload, f, ensure_ascii=False, separators=(",", ":"))
        with ranking_path.open("w", encoding="utf-8") as f:
            json.dump(ranking_payload, f, ensure_ascii=False, separators=(",", ":"))

    app_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "num_pdfs": parsed_pdf_count,
            "num_questions": len(exported_questions),
            "num_chain_steps": sum(len(steps) for steps in chains.values()),
            "num_cards": len([q for q in exported_questions if q.get("card_url")]),
            "quality_pass": len([q for q in exported_questions if q.get("quality_status") == "pass"]),
            "quality_warn": len([q for q in exported_questions if q.get("quality_status") == "warn"]),
            "quality_fail": len([q for q in exported_questions if q.get("quality_status") == "fail"]),
        },
        "pdfs": pdfs,
        "questions": exported_questions,
        "chains": {},
        "rankings": {},
        "metrics": metrics,
    }

    output_path = WEB_PUBLIC / "app-data.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(app_data, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(exported_questions)} questions to {output_path}")
    print(f"Exported {len(exported_questions)} question details to {DETAILS_DIR}")
    print(f"Exported {len(exported_questions)} ranking details to {RANKINGS_DIR}")
    print(f"Copied {len(copied_assets)} visual assets into {WEB_PUBLIC}")


if __name__ == "__main__":
    main()
