from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from pipeline_common import clean_text, read_csv, resolve_path, write_csv


VISUAL_NODE_TYPES = {"table", "figure", "caption"}
EXPECTED_CARD_SIZES = {(1920, 1080)}
VISUAL_QUERY_TERMS = (
    "table",
    "figure",
    "fig.",
    "chart",
    "plot",
    "\u8868",
    "\u8868\u683c",
    "\u56fe",
    "\u56fe\u8868",
    "\u56fe\u6587",
    "\u8de8\u6a21\u6001",
)

REPORT_FIELDS = [
    "question_id",
    "doc_id",
    "question_type",
    "visual_required",
    "card_path",
    "card_exists",
    "card_width",
    "card_height",
    "card_size_kb",
    "card_nonblank",
    "has_question",
    "has_answer",
    "chain_steps",
    "visual_node_steps",
    "crop_steps",
    "existing_crop_steps",
    "qwen_caption_steps",
    "source_pages",
    "status",
    "issues",
]


def group_steps(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[clean_text(row.get("question_id"))].append(row)
    return grouped


def question_requires_visual(question: dict[str, str]) -> bool:
    blob = clean_text(
        f"{question.get('question_type', '')} {question.get('question', '')} {question.get('gold_modalities', '')}"
    ).lower()
    return any(term in blob for term in VISUAL_QUERY_TERMS)


def card_stats(path: Path) -> tuple[int, int, int, int]:
    if not path.exists():
        return 0, 0, 0, 0
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return 0, 0, int(path.stat().st_size / 1024), 0
    stat = ImageStat.Stat(image.resize((80, 80)))
    nonblank = int(sum(stat.stddev) > 3.0)
    return image.width, image.height, int(path.stat().st_size / 1024), nonblank


def existing_crop_count(steps: list[dict[str, str]]) -> int:
    total = 0
    for step in steps:
        crop = clean_text(step.get("crop_image_path"))
        if crop and resolve_path(crop).exists():
            total += 1
    return total


def build_report_row(question: dict[str, str], steps: list[dict[str, str]], card_dir: Path) -> dict[str, Any]:
    qid = clean_text(question.get("question_id"))
    card_path = card_dir / f"{qid}_evidence_card.png"
    width, height, size_kb, nonblank = card_stats(card_path)
    visual_required = question_requires_visual(question)
    visual_node_steps = sum(1 for step in steps if clean_text(step.get("node_type")) in VISUAL_NODE_TYPES)
    crop_steps = sum(1 for step in steps if clean_text(step.get("crop_image_path")))
    existing_crops = existing_crop_count(steps)
    qwen_caption_steps = sum(
        1
        for step in steps
        if clean_text(step.get("visual_caption")) or "VLM caption" in clean_text(step.get("visual_summary"))
    )
    pages = sorted(
        {clean_text(step.get("page")) for step in steps if clean_text(step.get("page"))},
        key=lambda item: int(float(item)) if item.replace(".", "", 1).isdigit() else 9999,
    )

    issues: list[str] = []
    if not card_path.exists():
        issues.append("missing_card")
    if (width, height) not in EXPECTED_CARD_SIZES:
        issues.append("unexpected_card_size")
    if not nonblank:
        issues.append("blank_or_unreadable_card")
    if not clean_text(question.get("question")):
        issues.append("missing_question")
    if not clean_text(question.get("answer")):
        issues.append("missing_answer")
    if len(steps) < 3:
        issues.append("too_few_chain_steps")
    if existing_crops < 1:
        issues.append("missing_existing_crop")
    if visual_required and visual_node_steps < 1:
        issues.append("visual_required_without_visual_node")
    if visual_required and qwen_caption_steps < 1:
        issues.append("visual_required_without_qwen_caption")

    hard_fail = {
        "missing_card",
        "unexpected_card_size",
        "blank_or_unreadable_card",
        "missing_question",
        "missing_answer",
        "too_few_chain_steps",
        "missing_existing_crop",
    }
    if any(issue in hard_fail for issue in issues):
        status = "fail"
    elif issues:
        status = "warn"
    else:
        status = "pass"

    return {
        "question_id": qid,
        "doc_id": question.get("doc_id", ""),
        "question_type": question.get("question_type", ""),
        "visual_required": int(visual_required),
        "card_path": str(card_path.relative_to(resolve_path("."))) if card_path.exists() else str(card_path),
        "card_exists": int(card_path.exists()),
        "card_width": width,
        "card_height": height,
        "card_size_kb": size_kb,
        "card_nonblank": nonblank,
        "has_question": int(bool(clean_text(question.get("question")))),
        "has_answer": int(bool(clean_text(question.get("answer")))),
        "chain_steps": len(steps),
        "visual_node_steps": visual_node_steps,
        "crop_steps": crop_steps,
        "existing_crop_steps": existing_crops,
        "qwen_caption_steps": qwen_caption_steps,
        "source_pages": ";".join(pages),
        "status": status,
        "issues": ";".join(issues),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check evidence-chain PNG cards and write a quality report.")
    parser.add_argument("--questions", default="data/questions.csv")
    parser.add_argument("--chain-steps", default="outputs/evidence_chains/chain_steps.csv")
    parser.add_argument("--card-dir", default="outputs/evidence_cards")
    parser.add_argument("--output", default="outputs/evidence_cards/cards_quality_report.csv")
    args = parser.parse_args()

    questions = [row for row in read_csv(args.questions) if clean_text(row.get("question_id"))]
    steps_by_qid = group_steps(read_csv(args.chain_steps))
    card_dir = resolve_path(args.card_dir)
    rows = [
        build_report_row(question, steps_by_qid.get(clean_text(question.get("question_id")), []), card_dir)
        for question in questions
    ]
    write_csv(args.output, rows, REPORT_FIELDS)

    counts = Counter(row["status"] for row in rows)
    print(f"Wrote card quality report to {resolve_path(args.output)}")
    print(f"Cards checked: {len(rows)} | pass={counts.get('pass', 0)} warn={counts.get('warn', 0)} fail={counts.get('fail', 0)}")


if __name__ == "__main__":
    main()
