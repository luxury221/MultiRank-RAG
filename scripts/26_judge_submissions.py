from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import importlib.util
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ark_clients import ArkChatClient, get_env
from pipeline_common import clean_text, preview, read_csv, resolve_path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def _load_answer_tools() -> Any:
    target = SCRIPTS_DIR / "21_generate_competition_submission_llm.py"
    spec = importlib.util.spec_from_file_location("answer_tools", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOLS = _load_answer_tools()
_THREAD = threading.local()


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


def load_evidence_candidates(path: str | Path, top_k: int) -> dict[str, list[str]]:
    if not path:
        return {}
    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    grouped: dict[str, list[str]] = {}
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = submission_id(row.get("question_id", ""))
            if not qid:
                continue
            bucket = grouped.setdefault(qid, [])
            if len(bucket) >= top_k:
                continue
            node_type = clean_text(row.get("node_type"))
            source_ref = clean_text(row.get("source_ref"))[:220]
            content = clean_text(row.get("content_preview"))[:650]
            image_hint = ""
            if node_type == "figure":
                image_match = re.search(r"\b(?:Image id|image_id)\s*[:=]\s*([A-Za-z0-9_\-]+)", content)
                if image_match:
                    image_hint = f" image_id={image_match.group(1)}"
            bucket.append(
                f"[E{len(bucket) + 1}] type={node_type or '?'} rank={clean_text(row.get('rank'))}"
                f"{image_hint} source={source_ref}\n{content}"
            )
    return grouped


def valid_pic_suffix(text: str) -> bool:
    if "<PIC>" not in text:
        return True
    match = re.search(r"<PIC>\s*;\s*(\[[^\]]+\])\s*$", text)
    if not match:
        return False
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return False
    return isinstance(value, list) and 1 <= len(value) <= 3 and all(isinstance(item, str) and item for item in value)


def has_pic(text: str) -> bool:
    return "<PIC>" in text


def looks_truncated_answer(text: str) -> bool:
    text = TOOLS.clean_submission_text(text)
    if re.search(r"\b\d+\.\s*(?:<PIC>|$)", text):
        return True
    tail = re.sub(r"\s*<PIC>\s*;\s*\[[^\]]+\]\s*$", "", text).strip()
    if not tail:
        return False
    if re.search(r"\b(?:and|or|to|with|for|the|a|an|of|in|on|at|by|from|then|after|before)$", tail, flags=re.I):
        return True
    if re.search(r"(?:，|、|:|：|;|；)$", tail):
        return True
    return False


def actionable_score(text: str) -> float:
    text = clean_text(text).casefold()
    markers = (
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
        "注意",
        "step",
        "first",
        "press",
        "open",
        "close",
        "install",
        "remove",
        "clean",
        "replace",
        "note",
    )
    hits = sum(1 for marker in markers if marker.casefold() in text)
    return min(2.0, 0.35 * hits)


def heuristic_score(question: str, answer: str) -> float:
    answer = TOOLS.clean_submission_text(answer)
    score = 4.0
    length = len(answer)
    if length < 80:
        score -= 1.2
    elif length > 520:
        score -= 0.25
    else:
        score += 0.45
    if TOOLS.is_uncertain_answer(answer):
        score -= 2.4
    if looks_truncated_answer(answer):
        score -= 1.6
    if not valid_pic_suffix(answer):
        score -= 1.5
    if has_pic(answer) and TOOLS.is_service_question(question):
        score -= 1.0
    if has_pic(answer) and TOOLS.is_manual_visual_question(question):
        score += 0.35
    score += actionable_score(answer)
    q_terms = set(re.findall(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", question.casefold()))
    if q_terms:
        coverage = sum(1 for term in q_terms if term in answer.casefold()) / max(1, len(q_terms))
        score += min(1.2, 1.8 * coverage)
    return round(max(0.0, min(10.0, score)), 3)


class DashScopeChatClient:
    def __init__(self, model: str = "qwen-max", timeout_seconds: int = 120) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for DashScope-compatible chat calls.") from exc
        api_key = get_env("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured.")
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=timeout_seconds,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 420,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


def chat_client(provider: str, model: str) -> Any:
    cache_key = f"client_{provider}_{model}"
    client = getattr(_THREAD, cache_key, None)
    if client is not None:
        return client
    if provider == "dashscope":
        client = DashScopeChatClient(model=model or "qwen-max")
    elif provider == "ark":
        client = ArkChatClient(model=model or None)
    else:
        try:
            client = ArkChatClient(model=model or None)
        except Exception:
            client = DashScopeChatClient(model=model or "qwen-max")
    setattr(_THREAD, cache_key, client)
    return client


def extract_json(text: str) -> dict[str, Any]:
    text = clean_text(text).strip("` \n")
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    return json.loads(text)


def judge_with_llm(
    qid: str,
    question: str,
    candidates: dict[str, str],
    evidence_blocks: list[str],
    baseline_label: str,
    judge_provider: str,
    judge_model: str,
) -> dict[str, Any]:
    labels = list(candidates)
    heuristic = {label: heuristic_score(question, answer) for label, answer in candidates.items()}
    candidate_blocks = []
    for label, answer in candidates.items():
        answer = TOOLS.clean_submission_text(answer)
        candidate_blocks.append(
            f"[{label}]\n"
            f"heuristic={heuristic[label]} pic={int(has_pic(answer))} "
            f"uncertain={int(TOOLS.is_uncertain_answer(answer))} truncated={int(looks_truncated_answer(answer))}\n"
            f"{answer[:1100]}"
        )
    evidence_text = "\n\n".join(evidence_blocks[:10]) if evidence_blocks else "No retrieved evidence was provided."
    system_prompt = (
        "You are a strict answer selector for a multimodal customer-service QA benchmark. "
        "Choose the candidate that best answers the question. Penalize wrong product/domain, refusal-like answers, "
        "missing operation steps, invalid image suffixes, and vague customer-service boilerplate. "
        "Use the retrieved evidence as the source of truth; penalize fluent answers that conflict with it. "
        f"When two answers are close, prefer {baseline_label} because it is the validated baseline. "
        "Return only JSON."
    )
    user_prompt = f"""Question id: {qid}
Question:
{question}

Retrieved evidence snippets:
{evidence_text}

Candidates:

{chr(10).join(candidate_blocks)}

Scoring rubric:
- 0-10 relevance and correctness to the exact question
- completeness of sub-questions
- concrete actionable steps or product details
- no unsupported exact promise/price/deadline
- consistency with the retrieved evidence and the exact product/domain
- valid <PIC> ;["id"] suffix when useful for manual/visual questions; no PIC for pure service policy questions
- avoid "no relevant information" if another candidate gives a reasonable answer
- strongly penalize answers that end with an incomplete numbered step such as "3. <PIC>" or are visibly cut off

Return JSON:
{{"best_label":"...", "scores":{{"{labels[0]}":0}}, "reason":"short"}}
"""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = chat_client(judge_provider, judge_model).complete(
                system_prompt,
                user_prompt,
                temperature=0.0,
                max_tokens=320,
            )
            result = extract_json(raw)
            best = clean_text(result.get("best_label"))
            scores = result.get("scores") if isinstance(result.get("scores"), dict) else {}
            cleaned_scores = {
                label: float(scores.get(label, heuristic.get(label, 0.0)))
                for label in labels
            }
            if best not in candidates:
                best = max(cleaned_scores, key=cleaned_scores.get)
            return {
                "id": qid,
                "best_label": best,
                "scores": cleaned_scores,
                "heuristic_scores": heuristic,
                "reason": clean_text(result.get("reason")),
            }
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    best = max(heuristic, key=heuristic.get)
    return {
        "id": qid,
        "best_label": best,
        "scores": heuristic,
        "heuristic_scores": heuristic,
        "reason": f"LLM judge failed; heuristic fallback: {last_error}",
    }


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
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
            if qid:
                cache[qid] = row
    return cache


def append_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an LLM judge to select the best answer per question.")
    parser.add_argument("--questions", default="outputs/after_sales_kb/questions.csv")
    parser.add_argument("--submission", action="append", required=True, help="label=path")
    parser.add_argument("--output", default="outputs/datafountain_submit_ensemble.csv")
    parser.add_argument("--report", default="outputs/after_sales_kb/submission_ensemble_judge.jsonl")
    parser.add_argument("--cache", default="outputs/after_sales_kb/submission_ensemble_judge_cache.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--min-margin", type=float, default=0.35)
    parser.add_argument("--judge-provider", choices=["auto", "ark", "dashscope"], default="auto")
    parser.add_argument("--judge-model", default="", help="For DashScope use qwen-max/qwen-plus; for Ark use endpoint id.")
    parser.add_argument("--evidence-candidates", default="", help="Optional retrieved candidate CSV used as grounding evidence.")
    parser.add_argument("--evidence-top-k", type=int, default=8)
    args = parser.parse_args()

    specs = [parse_submission_spec(spec) for spec in args.submission]
    baseline_label = specs[0][0]
    submissions = {label: read_submission(path) for label, path in specs}
    questions = {
        submission_id(row["question_id"]): row
        for row in read_csv(args.questions)
        if clean_text(row.get("question_id")) and clean_text(row.get("question"))
    }
    evidence_by_qid = load_evidence_candidates(args.evidence_candidates, max(0, args.evidence_top_k))
    cache_path = resolve_path(args.cache)
    cached = load_cache(cache_path) if args.resume else {}

    rows: list[dict[str, str]] = []
    report_rows: list[dict[str, Any]] = []
    pending: list[tuple[str, str, dict[str, str], list[str]]] = []
    for qid in sorted(questions, key=lambda item: int(item) if item.isdigit() else item):
        question = clean_text(questions[qid].get("question"))
        candidates = {
            label: TOOLS.clean_submission_text(rows_by_id[qid])
            for label, rows_by_id in submissions.items()
            if qid in rows_by_id and clean_text(rows_by_id[qid])
        }
        if not candidates:
            continue
        if qid in cached:
            result = cached[qid]
            report_rows.append(result)
            selected_label = clean_text(result.get("selected_label") or result.get("best_label")) or baseline_label
            rows.append({"id": qid, "ret": candidates.get(selected_label, candidates.get(baseline_label, next(iter(candidates.values()))))})
            continue
        pending.append((qid, question, candidates, evidence_by_qid.get(qid, [])))

    def process(item: tuple[str, str, dict[str, str], list[str]]) -> dict[str, Any]:
        qid, question, candidates, evidence_blocks = item
        if args.no_llm:
            scores = {label: heuristic_score(question, answer) for label, answer in candidates.items()}
            best_label = max(scores, key=scores.get)
            result = {"id": qid, "best_label": best_label, "scores": scores, "heuristic_scores": scores, "reason": "heuristic"}
        else:
            result = judge_with_llm(
                qid,
                question,
                candidates,
                evidence_blocks,
                baseline_label,
                args.judge_provider,
                args.judge_model,
            )
        scores = {label: float(score) for label, score in result.get("scores", {}).items()}
        best_label = clean_text(result.get("best_label")) if clean_text(result.get("best_label")) in candidates else baseline_label
        baseline_score = scores.get(baseline_label, heuristic_score(question, candidates.get(baseline_label, "")))
        best_score = scores.get(best_label, heuristic_score(question, candidates.get(best_label, "")))
        selected_label = best_label if best_label == baseline_label or best_score >= baseline_score + args.min_margin else baseline_label
        # Hard guardrails beat judge enthusiasm.
        if TOOLS.is_uncertain_answer(candidates[selected_label]) and baseline_label in candidates and not TOOLS.is_uncertain_answer(candidates[baseline_label]):
            selected_label = baseline_label
        if not valid_pic_suffix(candidates[selected_label]) and baseline_label in candidates and valid_pic_suffix(candidates[baseline_label]):
            selected_label = baseline_label
        if looks_truncated_answer(candidates[selected_label]):
            selected_score = scores.get(selected_label, heuristic_score(question, candidates[selected_label]))
            alternatives = [
                (
                    scores.get(label, heuristic_score(question, answer)),
                    label,
                )
                for label, answer in candidates.items()
                if not looks_truncated_answer(answer) and not TOOLS.is_uncertain_answer(answer) and valid_pic_suffix(answer)
            ]
            if alternatives:
                alt_score, alt_label = max(alternatives, key=lambda item: item[0])
                if alt_score >= selected_score - 0.5:
                    selected_label = alt_label
        result["selected_label"] = selected_label
        result["selected_ret"] = candidates[selected_label]
        return result

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(process, item) for item in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                append_cache(cache_path, result)
                report_rows.append(result)
                rows.append({"id": result["id"], "ret": result["selected_ret"]})
                if index == 1 or index % 20 == 0 or index == len(futures):
                    print(
                        f"Judged {index}/{len(futures)} pending; id={result['id']} "
                        f"best={result.get('best_label')} selected={result.get('selected_label')} "
                        f"{preview(result.get('reason', ''), 80)}",
                        flush=True,
                    )

    by_id = {row["id"]: row for row in rows}
    rows = [by_id[qid] for qid in sorted(by_id, key=lambda item: int(item) if item.isdigit() else item)]
    write_submission(args.output, rows)
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in sorted(report_rows, key=lambda item: int(item["id"]) if str(item["id"]).isdigit() else str(item["id"])):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in report_rows:
        label = clean_text(row.get("selected_label"))
        counts[label] = counts.get(label, 0) + 1
    print(f"Wrote {len(rows)} rows to {resolve_path(args.output)}")
    print(json.dumps({"selected_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
