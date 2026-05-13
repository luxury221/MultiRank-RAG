from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_answer_tools() -> Any:
    target = SCRIPTS_DIR / "21_generate_competition_submission_llm.py"
    spec = importlib.util.spec_from_file_location("answer_tools", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOLS = _load_answer_tools()


def read_submission(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row["ret"] for row in csv.DictReader(f)}


def pic_suffix(text: str) -> str:
    match = re.search(r"\s*<PIC>\s*;\s*(\[[^\]]+\])\s*$", TOOLS.clean_submission_text(text))
    return match.group(0).strip() if match else ""


def has_actionable_detail(text: str) -> bool:
    text = TOOLS.clean_submission_text(text)
    if not text:
        return False
    if TOOLS.is_uncertain_answer(text):
        return False
    if len(text) < 80:
        return False
    detail_markers = (
        "步骤",
        "首先",
        "准备",
        "确认",
        "按下",
        "打开",
        "关闭",
        "安装",
        "拆卸",
        "清洁",
        "更换",
        "Step",
        "First",
        "Press",
        "Open",
        "Close",
        "Install",
        "Remove",
        "Clean",
        "Replace",
    )
    return any(marker.casefold() in text.casefold() for marker in detail_markers)


def choose_answer(baseline: str, candidate: str) -> tuple[str, str]:
    base = TOOLS.clean_submission_text(baseline)
    cand = TOOLS.clean_submission_text(candidate)
    base_uncertain = TOOLS.is_uncertain_answer(base)
    cand_uncertain = TOOLS.is_uncertain_answer(cand)
    base_pic = pic_suffix(base)
    cand_pic = pic_suffix(cand)

    if base_uncertain and not cand_uncertain and has_actionable_detail(cand):
        chosen = cand
        reason = "candidate_replaces_uncertain_baseline"
    elif cand_uncertain and not base_uncertain:
        chosen = base
        reason = "baseline_keeps_concrete_answer"
    else:
        chosen = base
        reason = "baseline_default"

    # The older submission empirically preserved useful image grounding better.
    # If we keep a candidate answer but it dropped a valid old image suffix, restore it.
    if base_pic and "<PIC>" not in chosen and not TOOLS.is_service_question(chosen):
        chosen = TOOLS.clean_submission_text(f"{chosen} {base_pic}")
        reason += "+restore_pic"
    elif base_pic and cand_pic and len(base_pic) > len(cand_pic) and chosen == cand:
        chosen = TOOLS.clean_submission_text(f"{TOOLS.strip_pic_suffix(chosen)} {base_pic}")
        reason += "+prefer_richer_pic"

    return TOOLS.clean_submission_text(chosen), reason


def write_submission(path: str | Path, rows: list[dict[str, str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: int(row["id"]) if row["id"].isdigit() else row["id"])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative calibrated submission from two CSVs.")
    parser.add_argument("--baseline", default="outputs/datafountain_submit.csv")
    parser.add_argument("--candidate", default="outputs/datafountain_submit_best.csv")
    parser.add_argument("--output", default="outputs/datafountain_submit_calibrated.csv")
    parser.add_argument("--report", default="outputs/after_sales_kb/submission_calibrated_report.json")
    args = parser.parse_args()

    baseline = read_submission(ROOT / args.baseline)
    candidate = read_submission(ROOT / args.candidate)
    rows: list[dict[str, str]] = []
    reasons: dict[str, int] = {}
    changed: list[dict[str, str]] = []
    for qid in sorted(baseline, key=lambda item: int(item) if item.isdigit() else item):
        chosen, reason = choose_answer(baseline[qid], candidate.get(qid, ""))
        rows.append({"id": qid, "ret": chosen})
        reasons[reason] = reasons.get(reason, 0) + 1
        if chosen != TOOLS.clean_submission_text(baseline[qid]):
            changed.append({"id": qid, "reason": reason})

    write_submission(ROOT / args.output, rows)
    report = {
        "rows": len(rows),
        "changed": len(changed),
        "reasons": reasons,
        "changed_ids": changed,
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
