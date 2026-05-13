from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from ark_clients import ArkChatClient
from pipeline_common import clean_text, read_csv, resolve_path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOLS = _load_module("answer_tools", SCRIPTS_DIR / "21_generate_competition_submission_llm.py")
JUDGE = _load_module("judge_tools", SCRIPTS_DIR / "26_judge_submissions.py")


GENERIC_UNCERTAIN_TERMS = (
    "not covered in the provided material",
    "not covered in the provided content",
    "specific operation",
    "specific details",
    "cannot find the cause",
    "contact an authorized",
    "contact a qualified professional",
    "refer to the official manual",
)


def submission_id(question_id: str) -> str:
    match = re.search(r"(\d+)$", clean_text(question_id))
    return match.group(1) if match else clean_text(question_id)


def read_submission(path: str | Path) -> dict[str, str]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row["ret"] for row in csv.DictReader(f)}


def write_submission(path: str | Path, rows: list[dict[str, str]]) -> None:
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: int(row["id"]) if row["id"].isdigit() else row["id"])
    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(rows)


def parse_submission_spec(spec: str) -> tuple[str, str]:
    label, sep, path = spec.partition("=")
    if not sep:
        file_path = Path(spec)
        return file_path.stem, spec
    return clean_text(label) or Path(path).stem, path


def pic_ids(text: str) -> list[str]:
    match = re.search(r"<PIC>\s*;\s*(\[[^\]]+\])\s*$", text)
    if not match:
        return []
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    ids: list[str] = []
    for item in value if isinstance(value, list) else []:
        image_id = clean_text(item)
        if image_id and image_id not in ids:
            ids.append(image_id)
    return ids[:3]


def pic_suffix(text: str) -> str:
    ids = pic_ids(text)
    return f"<PIC> ;{json.dumps(ids, ensure_ascii=False)}" if ids else ""


def extract_image_ids(*texts: str) -> list[str]:
    image_ids: list[str] = []
    for text in texts:
        for image_id in pic_ids(text):
            if image_id not in image_ids:
                image_ids.append(image_id)
        for image_id in re.findall(r"\b(?:Image id|image_id)\s*[:=]\s*([A-Za-z0-9_\-]+)", text):
            if image_id not in image_ids:
                image_ids.append(image_id)
        for image_id in re.findall(r"\b(?:Manual|Camera|jetski|snowmobile|oven|pump)_[A-Za-z0-9_\-]+", text):
            if image_id not in image_ids:
                image_ids.append(image_id)
    return image_ids[:8]


def is_generic_uncertain(text: str) -> bool:
    blob = clean_text(text).casefold()
    return TOOLS.is_uncertain_answer(text) or any(term in blob for term in GENERIC_UNCERTAIN_TERMS)


def strip_pic(text: str) -> str:
    return re.sub(r"\s*<PIC>\s*;\s*\[[^\]]+\]\s*$", "", clean_text(text)).strip()


def looks_cut_by_limit(text: str) -> bool:
    text = TOOLS.clean_submission_text(text)
    if len(text) >= 755:
        return True
    tail = strip_pic(text)
    if re.search(r"\b[a-z]{1,4}\.$", tail, flags=re.I):
        return True
    if re.search(r"(?:,|;|:|\band|\bor|\bto|\bwith|\bfor|\bthe|\ba|\ban|\bof|\bin|\bon|\bat|\bby|\bfrom|\bthen|\bafter|\bbefore)$", tail, flags=re.I):
        return True
    return False


def is_risky(qid: str, question: str, base: str, baseline: str) -> bool:
    if JUDGE.looks_truncated_answer(base):
        return True
    if looks_cut_by_limit(base):
        return True
    if is_generic_uncertain(base) and not is_generic_uncertain(baseline):
        return True
    if not TOOLS.is_service_question(question) and "<PIC>" in baseline and "<PIC>" not in base:
        return True
    return False


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    cached: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = clean_text(row.get("id"))
            ret = clean_text(row.get("ret"))
            if qid and ret:
                cached[qid] = ret
    return cached


def append_cache(path: Path, qid: str, ret: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"id": qid, "ret": ret}, ensure_ascii=False) + "\n")


def clean_refined_answer(question: str, answer: str, fallback: str, baseline: str) -> str:
    answer = TOOLS.clean_submission_text(answer)
    if TOOLS.is_service_question(question):
        answer = TOOLS.clean_submission_text(TOOLS.strip_pic_suffix(answer))
    elif "<PIC>" not in answer and "<PIC>" in baseline:
        suffix = pic_suffix(baseline)
        if suffix:
            answer = TOOLS.clean_submission_text(f"{answer} {suffix}")
    if not answer or is_generic_uncertain(answer) and not is_generic_uncertain(baseline):
        return TOOLS.clean_submission_text(baseline)
    if not JUDGE.valid_pic_suffix(answer):
        answer = TOOLS.clean_submission_text(TOOLS.strip_pic_suffix(answer))
    if (JUDGE.looks_truncated_answer(answer) or looks_cut_by_limit(answer)) and not (
        JUDGE.looks_truncated_answer(fallback) or looks_cut_by_limit(fallback)
    ):
        return TOOLS.clean_submission_text(fallback)
    return answer


def refine_one(
    client: ArkChatClient,
    qid: str,
    question: str,
    evidence: list[str],
    candidates: dict[str, str],
    baseline_label: str,
) -> str:
    allowed_images = extract_image_ids(*(list(candidates.values()) + evidence))
    candidate_text = "\n\n".join(
        f"[{label}]\n{TOOLS.clean_submission_text(answer)[:900]}"
        for label, answer in candidates.items()
        if clean_text(answer)
    )
    evidence_text = "\n\n".join(evidence[:10]) if evidence else "No retrieved evidence was provided."
    image_rule = (
        f"If an image is useful, append exactly one suffix using at most 3 ids from this list: {json.dumps(allowed_images, ensure_ascii=False)}."
        if allowed_images and not TOOLS.is_service_question(question)
        else "Do not append a <PIC> suffix."
    )
    system_prompt = (
        "You write final benchmark submission answers. Use the evidence as source of truth. "
        "Keep the answer concise, complete, and directly useful. Return only the final answer text."
    )
    user_prompt = f"""Question id: {qid}
Question:
{question}

Retrieved evidence:
{evidence_text}

Candidate answers:
{candidate_text}

Requirements:
- Use the same language as the question.
- Prefer factual content supported by retrieved evidence or the strongest candidate.
- Keep the final answer under 620 characters, including any <PIC> suffix.
- Do not end with an incomplete numbered step such as "3." or "4. <PIC>".
- Do not say the material has no information if a candidate gives supported concrete details.
- {image_rule}
- For policy/service questions, do not include pictures.
- Return only the final answer, no JSON and no markdown.
"""
    fallback = candidates.get("base") or candidates.get(baseline_label) or next(iter(candidates.values()))
    baseline = candidates.get(baseline_label, fallback)
    last = fallback
    for attempt in range(2):
        raw = client.complete(system_prompt, user_prompt, temperature=0.05, max_tokens=420)
        last = clean_refined_answer(question, raw, fallback, baseline)
        if not JUDGE.looks_truncated_answer(last) and not looks_cut_by_limit(last) and not (
            is_generic_uncertain(last) and not is_generic_uncertain(baseline)
        ):
            return last
        user_prompt += f"\n\nPrevious draft was still risky, too long, or truncated. Rewrite it as a complete answer under 560 characters including the image suffix:\n{last}"
        time.sleep(0.5 * (attempt + 1))
    return clean_refined_answer(question, last, fallback, baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine only high-risk submission answers with Ark and retrieved evidence.")
    parser.add_argument("--questions", default="outputs/after_sales_kb/questions.csv")
    parser.add_argument("--base", default="outputs/datafountain_submit_ensemble_ark_evidence.csv")
    parser.add_argument("--baseline", default="calibrated=outputs/datafountain_submit_calibrated.csv")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--evidence-candidates", default="outputs/after_sales_kb/candidates_union.csv")
    parser.add_argument("--evidence-top-k", type=int, default=10)
    parser.add_argument("--ids", default="", help="Comma-separated ids. If empty, risky ids are detected automatically.")
    parser.add_argument("--output", default="outputs/datafountain_submit_ensemble_ark_evidence_refined.csv")
    parser.add_argument("--cache", default="outputs/after_sales_kb/submission_refine_ark_cache.jsonl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    questions = {
        submission_id(row["question_id"]): clean_text(row.get("question"))
        for row in read_csv(args.questions)
        if clean_text(row.get("question_id")) and clean_text(row.get("question"))
    }
    base = read_submission(args.base)
    baseline_label, baseline_path = parse_submission_spec(args.baseline)
    baseline = read_submission(baseline_path)
    candidate_specs = [("base", args.base), (baseline_label, baseline_path)]
    candidate_specs.extend(parse_submission_spec(spec) for spec in args.candidate)
    submissions = {label: read_submission(path) for label, path in candidate_specs}
    evidence_by_qid = JUDGE.load_evidence_candidates(args.evidence_candidates, max(0, args.evidence_top_k))
    cache_path = resolve_path(args.cache)
    cache = load_cache(cache_path) if args.resume else {}

    if args.ids:
        refine_ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    else:
        refine_ids = [
            qid
            for qid in sorted(base, key=lambda item: int(item) if item.isdigit() else item)
            if qid in questions and qid in baseline and is_risky(qid, questions[qid], base[qid], baseline[qid])
        ]

    client = ArkChatClient()
    refined = dict(base)
    print(f"Refining {len(refine_ids)} ids: {','.join(refine_ids)}", flush=True)
    for index, qid in enumerate(refine_ids, start=1):
        if qid in cache:
            refined[qid] = cache[qid]
            print(f"Refined {index}/{len(refine_ids)} id={qid} cache", flush=True)
            continue
        question = questions[qid]
        candidates = {
            label: submission[qid]
            for label, submission in submissions.items()
            if qid in submission and clean_text(submission[qid])
        }
        ret = refine_one(client, qid, question, evidence_by_qid.get(qid, []), candidates, baseline_label)
        refined[qid] = ret
        append_cache(cache_path, qid, ret)
        print(f"Refined {index}/{len(refine_ids)} id={qid} len={len(ret)} pic={int('<PIC>' in ret)}", flush=True)

    rows = [{"id": qid, "ret": TOOLS.clean_submission_text(refined[qid])} for qid in refined]
    write_submission(args.output, rows)
    print(f"Wrote {len(rows)} rows to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
