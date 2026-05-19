from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from ark_clients import create_chat_client
from pipeline_common import clean_text, preview, read_jsonl, resolve_path, write_csv, write_jsonl
from rerank_lib import self_correct_answer


PROMPT_VERSION = "chain-answer-v1"

ANSWER_FIELDS = [
    "question_id",
    "doc_id",
    "question_type",
    "question",
    "answer",
    "provider",
    "model",
    "status",
    "latency_ms",
    "self_correction_status",
    "self_correction_removed_sentences",
    "self_correction_notes",
    "evidence_count",
    "visual_evidence_count",
    "evidence_node_ids",
    "evidence_pages",
    "evidence_modalities",
    "citations",
    "chain_summary",
]

SYSTEM_PROMPT = """你是一个严谨的多模态 RAG 答案生成器。
你的任务是只根据给定证据链回答问题，不能使用证据链之外的知识。
回答必须结论先行、结构清楚、信息合并自然，并体现文本、表格、图片或图注之间的互补关系。
如果证据包含图片、表格、图注或视觉摘要，请在对应结论附近保留 <PIC:证据节点ID> 标记。
如果证据不足，请明确说明“根据当前证据无法确认”，不要编造。
答案语言应与用户问题保持一致。"""


def looks_chinese(text: str) -> bool:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return cjk >= max(2, latin // 3)


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        item = clean_text(item)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def trim_sentence_tail(text: str) -> str:
    return re.sub(r"[\s。；;,.，、]+$", "", clean_text(text))


def step_is_visual(step: dict[str, Any]) -> bool:
    node_type = clean_text(step.get("node_type")).lower()
    return bool(
        node_type in {"table", "figure", "caption"}
        or clean_text(step.get("crop_image_path"))
        or clean_text(step.get("page_image_path"))
        or clean_text(step.get("visual_summary"))
        or clean_text(step.get("visual_caption"))
    )


def evidence_content(step: dict[str, Any], limit: int = 380) -> str:
    parts = [
        clean_text(step.get("content_preview")),
        clean_text(step.get("visual_summary")),
        clean_text(step.get("visual_caption")),
    ]
    return preview(" ".join(part for part in parts if part), limit)


def citation_for_step(index: int, step: dict[str, Any]) -> str:
    node_id = clean_text(step.get("node_id"))
    node_type = clean_text(step.get("node_type"))
    page = clean_text(step.get("page"))
    page_part = f"p.{page}" if page else "p.?"
    return f"[E{index}] {node_id} ({node_type}, {page_part})"


def visual_marker(step: dict[str, Any]) -> str:
    if not step_is_visual(step):
        return ""
    node_id = clean_text(step.get("node_id"))
    return f"<PIC:{node_id}>" if node_id else "<PIC>"


def build_prompt(chain: dict[str, Any], max_steps: int) -> str:
    question = clean_text(chain.get("question"))
    summary = clean_text(chain.get("summary"))
    steps = list(chain.get("steps") or [])[:max_steps]
    lines = [
        f"问题：{question}",
        "",
        f"证据链摘要：{summary}",
        "",
        "证据链：",
    ]
    for index, step in enumerate(steps, 1):
        marker = visual_marker(step)
        content = evidence_content(step, limit=520)
        lines.extend(
            [
                (
                    f"[E{index}] node_id={clean_text(step.get('node_id'))}; "
                    f"type={clean_text(step.get('node_type'))}; page={clean_text(step.get('page'))}; "
                    f"role={clean_text(step.get('role'))}; relation={clean_text(step.get('relation'))}"
                ),
                f"reason: {clean_text(step.get('reason'))}",
                f"content: {content}",
            ]
        )
        if marker:
            lines.append(f"visual_marker: {marker}")
        lines.append("")
    lines.extend(
        [
            "请生成最终答案，要求：",
            "1. 直接回答问题，不复述无关背景。",
            "2. 合并多个证据，不要简单罗列证据原文。",
            "3. 关键判断后用 [E1] 这样的证据编号标注来源。",
            "4. 如果使用视觉证据，在相关句子附近保留 <PIC:node_id>。",
            "5. 不要输出 JSON，不要输出思考过程。",
        ]
    )
    return "\n".join(lines)


def fallback_answer(chain: dict[str, Any], max_steps: int) -> str:
    question = clean_text(chain.get("question"))
    steps = list(chain.get("steps") or [])[:max_steps]
    evidence = [
        (index, step, evidence_content(step, limit=240))
        for index, step in enumerate(steps, 1)
        if evidence_content(step, limit=240)
    ]
    if not evidence:
        return "根据当前证据链，暂时无法确认该问题的答案。" if looks_chinese(question) else (
            "The current evidence chain is insufficient to answer the question."
        )

    main_items = unique_keep_order([item[2] for item in evidence[:3]])
    visual_items = [(index, step) for index, step, _ in evidence if step_is_visual(step)]
    citations = " ".join(f"[E{index}]" for index, _, _ in evidence[:3])
    visual = ""
    if visual_items:
        visual = " " + " ".join(visual_marker(step) for _, step in visual_items[:2] if visual_marker(step))

    if looks_chinese(question):
        answer = "根据证据链，" + "；".join(trim_sentence_tail(item) for item in main_items if trim_sentence_tail(item))
        return clean_text(f"{answer}{visual}。证据来源：{citations}")
    answer = "Based on the evidence chain, " + "; ".join(main_items)
    return clean_text(f"{answer}{visual}. Sources: {citations}")


def cache_key(provider: str, model: str, chain: dict[str, Any], max_steps: int) -> str:
    steps = []
    for step in list(chain.get("steps") or [])[:max_steps]:
        steps.append(
            {
                "node_id": clean_text(step.get("node_id")),
                "content": evidence_content(step, limit=700),
                "visual": clean_text(step.get("visual_summary")) or clean_text(step.get("visual_caption")),
            }
        )
    raw = json.dumps(
        {
            "version": PROMPT_VERSION,
            "provider": provider,
            "model": model,
            "question": clean_text(chain.get("question")),
            "summary": clean_text(chain.get("summary")),
            "steps": steps,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    file_path = resolve_path(path)
    if not file_path.exists():
        return {}
    items: dict[str, dict[str, Any]] = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = clean_text(row.get("cache_key"))
            if key:
                items[key] = row
    return items


def append_cache(path: str | Path, row: dict[str, Any]) -> None:
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evidence_metadata(chain: dict[str, Any], max_steps: int) -> dict[str, str]:
    steps = list(chain.get("steps") or [])[:max_steps]
    node_ids = unique_keep_order([clean_text(step.get("node_id")) for step in steps])
    pages = unique_keep_order([clean_text(step.get("page")) for step in steps])
    modalities = unique_keep_order([clean_text(step.get("node_type")) for step in steps])
    citations = [citation_for_step(index, step) for index, step in enumerate(steps, 1)]
    return {
        "evidence_node_ids": ";".join(node_ids),
        "evidence_pages": ";".join(pages),
        "evidence_modalities": ";".join(modalities),
        "citations": " | ".join(citations),
    }


def generate_one(
    chain: dict[str, Any],
    client: Any,
    provider: str,
    model: str,
    max_steps: int,
    max_tokens: int,
    temperature: float,
    cache: dict[str, dict[str, Any]],
    cache_path: str,
) -> dict[str, Any]:
    steps = list(chain.get("steps") or [])[:max_steps]
    metadata = evidence_metadata(chain, max_steps)
    key = cache_key(provider, model, chain, max_steps)
    start = time.perf_counter()
    status = "ok"

    if key in cache:
        answer = clean_text(cache[key].get("answer"))
        status = clean_text(cache[key].get("status")) or "cached"
    elif provider == "none":
        answer = fallback_answer(chain, max_steps)
        status = "fallback"
    else:
        prompt = build_prompt(chain, max_steps)
        try:
            answer = clean_text(
                client.complete(
                    SYSTEM_PROMPT,
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            if not answer:
                answer = fallback_answer(chain, max_steps)
                status = "empty_fallback"
        except Exception as exc:  # pragma: no cover - depends on external model service.
            answer = fallback_answer(chain, max_steps)
            status = f"error_fallback: {type(exc).__name__}: {str(exc)[:160]}"
        append_cache(cache_path, {"cache_key": key, "answer": answer, "status": status})

    correction = self_correct_answer(chain, answer, steps)
    answer = clean_text(correction.get("answer")) or answer
    correction_status = clean_text(correction.get("status"))
    if correction_status and correction_status not in {"verified", "disabled"}:
        status = f"{status};answer_self_correction={correction_status}"

    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    visual_count = sum(1 for step in steps if step_is_visual(step))
    return {
        "question_id": clean_text(chain.get("question_id")),
        "doc_id": clean_text(chain.get("doc_id")),
        "question_type": clean_text(chain.get("question_type")),
        "question": clean_text(chain.get("question")),
        "answer": answer,
        "provider": provider,
        "model": model,
        "status": status,
        "latency_ms": latency_ms,
        "self_correction_status": correction_status,
        "self_correction_removed_sentences": int(correction.get("removed_sentences") or 0),
        "self_correction_notes": clean_text(correction.get("notes")),
        "evidence_count": len(steps),
        "visual_evidence_count": visual_count,
        "chain_summary": clean_text(chain.get("summary")),
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final answers from V4 evidence chains.")
    parser.add_argument("--chains", default="outputs/evidence_chains/chains.jsonl")
    parser.add_argument("--output-csv", default="outputs/evidence_chains/answers.csv")
    parser.add_argument("--output-jsonl", default="outputs/evidence_chains/answers.jsonl")
    parser.add_argument("--cache", default="outputs/evidence_chains/answer_cache.jsonl")
    parser.add_argument(
        "--provider",
        default="none",
        choices=["none", "ark", "doubao", "volcengine", "xinference", "openai_compatible", "openai-compatible", "local_openai", "local-server"],
        help="none uses deterministic extractive fallback; other values call an OpenAI-compatible chat API.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    chains = read_jsonl(args.chains)
    if args.limit > 0:
        chains = chains[: args.limit]
    provider = clean_text(args.provider).lower()
    model = clean_text(args.model)
    client = None if provider == "none" else create_chat_client(provider, model=model or None)
    if client is not None and not model:
        model = clean_text(getattr(client, "model", ""))

    cache = load_cache(args.cache)
    rows = [
        generate_one(
            chain,
            client=client,
            provider=provider,
            model=model,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            cache=cache,
            cache_path=args.cache,
        )
        for chain in chains
        if clean_text(chain.get("question"))
    ]

    write_csv(args.output_csv, rows, ANSWER_FIELDS)
    write_jsonl(args.output_jsonl, rows)
    print(f"Wrote {len(rows)} chain-grounded answers to {resolve_path(args.output_csv)}")
    print(f"Wrote answer JSONL to {resolve_path(args.output_jsonl)}")


if __name__ == "__main__":
    main()
