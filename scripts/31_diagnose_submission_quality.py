from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

from pipeline_common import clean_text, read_csv, read_jsonl, resolve_path


def load_generator_module() -> Any:
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "21_generate_competition_submission_llm.py"
    spec = importlib.util.spec_from_file_location("competition_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = load_generator_module()


def read_submission(path: str | Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "profile", "flags", "question", "ret_preview", "pic_ids"]
    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def submission_id(question_id: str) -> str:
    return GEN.submission_id(question_id)


def load_questions(path: str | Path) -> dict[str, str]:
    questions: dict[str, str] = {}
    for row in read_csv(path):
        sid = submission_id(clean_text(row.get("question_id")) or clean_text(row.get("id")))
        question = clean_text(row.get("question"))
        if sid and question:
            questions[sid] = question
    return questions


def image_id_from_node(node: dict[str, Any]) -> str:
    image_id = clean_text(node.get("image_id"))
    if image_id:
        return image_id
    for field in ("image_path", "crop_image_path", "source_ref", "content"):
        value = clean_text(node.get(field))
        if not value:
            continue
        stem = Path(value).stem
        if re.fullmatch(r"[A-Za-z0-9_\-]+", stem):
            return stem
        match = re.search(r"\b(?:Image id|image_id)\s*[:=]\s*([A-Za-z0-9_\-]+)", value)
        if match:
            return match.group(1)
    return ""


def load_valid_image_ids(path: str | Path) -> set[str]:
    ids: set[str] = set()
    file_path = resolve_path(path)
    if not file_path.exists():
        return ids
    for node in read_jsonl(file_path):
        image_id = image_id_from_node(node)
        if image_id:
            ids.add(image_id)
    return ids


def parse_pic_ids(text: str) -> list[str]:
    match = re.search(r";\s*(\[[^\]]*\])\s*$", clean_text(text))
    if not match:
        return []
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return re.findall(r"[A-Za-z0-9_\-]+", match.group(1))
    ids: list[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            image_id = clean_text(str(item))
            if image_id and image_id not in ids:
                ids.append(image_id)
    return ids


def has_bad_pic_suffix(text: str) -> bool:
    text = clean_text(text)
    if "<PIC>" not in text:
        return False
    return not bool(re.search(r"<PIC>\s*;?\s*\[[^\]]+\]\s*$", text))


def is_dangling_answer(text: str) -> bool:
    stripped = re.sub(r"\s*<PIC>\s*;?\s*\[[^\]]+\]\s*$", "", clean_text(text)).strip()
    return bool(re.search(r"(?:^|\s)\d+\.\s*$", stripped) or re.search(r"[:;,\uff1a\uff1b\uff0c]\s*$", stripped))


def diagnose_row(row: dict[str, str], question: str, valid_image_ids: set[str]) -> dict[str, Any]:
    sid = clean_text(row.get("id"))
    ret = clean_text(row.get("ret"))
    pic_ids = parse_pic_ids(ret)
    flags: list[str] = []
    profile = GEN.answer_profile(question)
    service = GEN.is_service_question(question)
    manual_visual = GEN.is_manual_visual_question(question)
    invalid = [image_id for image_id in pic_ids if valid_image_ids and image_id not in valid_image_ids]

    if not ret:
        flags.append("empty_answer")
    if len(ret) < 45:
        flags.append("very_short")
    if len(ret) > 760:
        flags.append("too_long")
    if GEN.is_uncertain_answer(ret):
        flags.append("uncertain_or_refusal")
    if has_bad_pic_suffix(ret):
        flags.append("bad_pic_suffix")
    if invalid:
        flags.append("invalid_pic_id:" + "|".join(invalid))
    if service and pic_ids:
        flags.append("service_with_pic")
    if manual_visual and not pic_ids:
        flags.append("manual_visual_without_pic")
    if is_dangling_answer(ret):
        flags.append("possibly_truncated")

    return {
        "id": sid,
        "profile": profile,
        "flags": "|".join(flags),
        "question": question,
        "ret_preview": ret[:180],
        "pic_ids": json.dumps(pic_ids, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose DataFountain submission quality risks without changing the submission.")
    parser.add_argument("--submission", default="outputs/datafountain_submit_final.csv")
    parser.add_argument("--questions", default="outputs/after_sales_kb/questions.csv")
    parser.add_argument("--nodes", default="outputs/after_sales_kb/nodes.qwen_full.jsonl")
    parser.add_argument("--report", default="outputs/after_sales_kb/submission_quality_report.json")
    parser.add_argument("--flagged", default="outputs/after_sales_kb/submission_quality_flags.csv")
    args = parser.parse_args()

    rows = read_submission(args.submission)
    questions = load_questions(args.questions)
    valid_image_ids = load_valid_image_ids(args.nodes)
    diagnostics = [
        diagnose_row(row, questions.get(clean_text(row.get("id")), ""), valid_image_ids)
        for row in rows
    ]
    flagged = [row for row in diagnostics if row["flags"]]

    flag_counts: dict[str, int] = {}
    profile_counts: dict[str, int] = {}
    for item in diagnostics:
        profile = clean_text(item.get("profile")) or "unknown"
        profile_counts[profile] = profile_counts.get(profile, 0) + 1
        for flag in clean_text(item.get("flags")).split("|"):
            if not flag:
                continue
            key = flag.split(":", 1)[0]
            flag_counts[key] = flag_counts.get(key, 0) + 1

    report = {
        "submission": str(resolve_path(args.submission)),
        "rows": len(rows),
        "flagged_rows": len(flagged),
        "pic_rows": sum("<PIC>" in clean_text(row.get("ret")) for row in rows),
        "profile_counts": profile_counts,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))),
        "flagged_examples": flagged[:30],
    }

    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.flagged, flagged)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
