from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from pipeline_common import clean_text, preview, read_csv, resolve_path


DEFAULT_QUESTIONS = "outputs/after_sales_kb/questions.csv"
DEFAULT_CHAINS = "outputs/after_sales_kb/evidence_chains/chains.jsonl"
DEFAULT_OUTPUT = "outputs/after_sales_kb/submission.csv"


def load_fast_module():
    path = Path(__file__).with_name("17_generate_datafountain_submission.py")
    spec = importlib.util.spec_from_file_location("datafountain_fast_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load fast submission module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAST = load_fast_module()


def submission_id(question_id: str) -> str:
    question_id = clean_text(question_id)
    match = re.search(r"(\d+)$", question_id)
    return match.group(1) if match else question_id


def normalize_ret(text: str, limit: int = 520) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ["。", "；", ";", ".", "！", "!", "？", "?"]:
        pos = cut.rfind(sep)
        if pos >= limit * 0.55:
            return cut[: pos + 1]
    return cut.rstrip("，,、；;：: ") + "。"


def load_chains(path: str | Path) -> dict[str, dict[str, Any]]:
    file_path = resolve_path(path)
    chains: dict[str, dict[str, Any]] = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = clean_text(row.get("question_id"))
            if qid:
                chains[qid] = row
    return chains


def step_to_node(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": step.get("node_id", ""),
        "doc_id": step.get("doc_id") or source_doc(step),
        "node_type": step.get("node_type", ""),
        "page": step.get("page", ""),
        "content": step.get("content_preview", ""),
        "source_ref": step.get("source_ref", ""),
        "visual_summary": step.get("visual_summary", ""),
    }


def source_doc(step: dict[str, Any]) -> str:
    source_ref = clean_text(step.get("source_ref"))
    return source_ref.split("/", 1)[0].strip() if source_ref else ""


def useful_steps(chain: dict[str, Any]) -> list[dict[str, Any]]:
    steps = list(chain.get("steps") or [])
    useful: list[dict[str, Any]] = []
    for step in steps:
        text = clean_text(step.get("content_preview"))
        if not text:
            continue
        node_type = clean_text(step.get("node_type"))
        if node_type == "figure" and text.startswith("Manual illustration"):
            continue
        useful.append(step)
    return useful or steps[:1]


def policy_from_chain(question: str, kind: str, steps: list[dict[str, Any]]) -> str:
    english = FAST.is_english(question)
    main = steps[0] if steps else {}
    text = clean_text(main.get("content_preview"))
    node_id = clean_text(main.get("node_id"))
    if node_id.startswith("AS_POLICY") and text:
        if english:
            return normalize_ret(text)
        return normalize_ret(f"您好，{text}")
    return FAST.policy_answer(kind, question)


def manual_from_chain(question: str, steps: list[dict[str, Any]]) -> str:
    nodes = [step_to_node(step) for step in steps]
    if not nodes:
        return FAST.manual_answer(question, [], 0.0)
    ret = FAST.manual_answer(question, nodes, 1.0)
    return normalize_ret(ret)


def answer_for_question(question_row: dict[str, str], chain: dict[str, Any] | None) -> str:
    question = FAST.normalize_question(question_row.get("question", ""))
    kind = FAST.question_kind(question)
    if not chain:
        return normalize_ret(FAST.policy_answer(kind, question) if kind != "manual" else FAST.manual_answer(question, [], 0.0))

    steps = useful_steps(chain)
    first_node_id = clean_text(steps[0].get("node_id")) if steps else ""
    first_source = clean_text(steps[0].get("source_ref")) if steps else ""
    if first_node_id.startswith("AS_POLICY") or first_source.startswith("售后通用政策") or kind != "manual":
        return normalize_ret(policy_from_chain(question, kind, steps))
    return manual_from_chain(question, steps)


def generate_submission(
    questions: list[dict[str, str]],
    chains: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for question_row in questions:
        qid = clean_text(question_row.get("question_id"))
        ret = answer_for_question(question_row, chains.get(qid))
        rows.append({"id": submission_id(qid), "ret": ret})
    rows.sort(key=lambda row: int(row["id"]) if row["id"].isdigit() else row["id"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DataFountain id/ret submission from after-sales G4 evidence chains.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--chains", default=DEFAULT_CHAINS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    questions = read_csv(args.questions)
    chains = load_chains(args.chains)
    if not questions:
        raise SystemExit(f"No questions found: {resolve_path(args.questions)}")
    if not chains:
        raise SystemExit(f"No chains found: {resolve_path(args.chains)}")

    rows = generate_submission(questions, chains)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"First row: {json.dumps(rows[0], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
