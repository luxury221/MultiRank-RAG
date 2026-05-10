from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from ark_clients import ArkChatClient, ArkError
from pipeline_common import clean_text, preview, read_csv, read_jsonl, resolve_path


DEFAULT_QUESTIONS = "outputs/after_sales_kb/questions.csv"
DEFAULT_NODES = "outputs/after_sales_kb/nodes.jsonl"
DEFAULT_RANKINGS = "outputs/after_sales_kb/reranked.csv"
DEFAULT_OUTPUT = "outputs/after_sales_kb/submission_llm_rubric.csv"
DEFAULT_CACHE = "outputs/after_sales_kb/submission_llm_rubric_cache.jsonl"

SERVICE_TERMS = (
    "退货",
    "换货",
    "退款",
    "无理由",
    "运费",
    "发票",
    "开票",
    "物流",
    "快递",
    "投诉",
    "售后",
    "保修",
    "维修",
    "包装",
    "破损",
    "订单",
    "补发",
    "假货",
    "二手",
    "赔偿",
    "质保",
)


def submission_id(question_id: str) -> str:
    match = re.search(r"(\d+)$", clean_text(question_id))
    return match.group(1) if match else clean_text(question_id)


def is_english(text: str) -> bool:
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return letters > max(20, cjk * 2)


def is_service_question(question: str) -> bool:
    return any(term in question for term in SERVICE_TERMS)


def image_id_from_node(node: dict[str, Any]) -> str:
    image_id = clean_text(node.get("image_id"))
    if image_id:
        return image_id
    text = f"{node.get('source_ref', '')} {node.get('content', '')}"
    match = re.search(r"Image id:\s*([A-Za-z0-9_\-]+)", text)
    if match:
        return match.group(1)
    source_ref = clean_text(node.get("source_ref"))
    if "/" in source_ref:
        tail = source_ref.rsplit("/", 1)[-1].strip()
        if re.fullmatch(r"[A-Za-z0-9_\-]+", tail):
            return tail
    return ""


def clean_submission_text(text: str, limit: int = 760) -> str:
    text = clean_text(text).replace("```", "")
    text = re.sub(r"^\s*(答案|ret|输出)\s*[:：]\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    pic_part = ""
    pic_match = re.search(r"\s*<PIC>\s*;?\s*\[[^\]]+\]\s*$", text)
    if pic_match:
        pic_part = pic_match.group(0).strip()
        text = text[: pic_match.start()].strip()
    cut = text[: max(80, limit - len(pic_part) - 2)]
    for sep in ["。", "；", ";", ".", "！", "!", "？", "?"]:
        pos = cut.rfind(sep)
        if pos >= len(cut) * 0.55:
            cut = cut[: pos + 1]
            break
    else:
        cut = cut.rstrip("，,、；;：: ") + ("." if is_english(cut) else "。")
    return f"{cut} {pic_part}".strip() if pic_part else cut


def focused_content(node: dict[str, Any], question: str, limit: int = 900) -> str:
    content = clean_text(
        " ".join(
            [
                node.get("section", ""),
                node.get("previous_chunk_preview", ""),
                node.get("content", ""),
                node.get("next_chunk_preview", ""),
                node.get("visual_summary", ""),
            ]
        )
    )
    if len(content) <= limit:
        return content

    terms = [
        term
        for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", question)
        if term not in {"请问", "如何", "什么", "哪些", "怎么", "可以", "需要", "商品", "使用"}
    ]
    best = -1
    lowered = content.casefold()
    for term in terms:
        pos = lowered.find(term.casefold())
        if pos >= 0:
            best = pos
            break
    if best < 0:
        return preview(content, limit)
    start = max(0, best - limit // 3)
    end = min(len(content), start + limit)
    return content[start:end].strip(" ，,；;。")


def load_rankings(path: str | Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(path):
        if row.get("method") != "G4":
            continue
        qid = clean_text(row.get("question_id"))
        if qid:
            groups.setdefault(qid, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row.get("rank") or 9999))
    return groups


def select_evidence(
    question: str,
    ranking_rows: list[dict[str, str]],
    nodes_by_id: dict[str, dict[str, Any]],
    max_items: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    service_question = is_service_question(question)
    selected: list[dict[str, Any]] = []
    images: list[str] = []
    seen: set[str] = set()

    for row in ranking_rows[:18]:
        node = nodes_by_id.get(clean_text(row.get("node_id")), {})
        if not node:
            continue
        node_id = clean_text(node.get("node_id"))
        if node_id in seen:
            continue
        node_type = clean_text(node.get("node_type"))
        source_ref = clean_text(node.get("source_ref"))
        if node_id.startswith("AS_PROFILE_"):
            continue
        if not service_question and source_ref.startswith("售后通用政策"):
            continue
        if service_question and node_type == "figure" and len(selected) < 2:
            continue

        seen.add(node_id)
        selected.append({"row": row, "node": node})
        image_id = image_id_from_node(node)
        if image_id and image_id not in images and node_type == "figure":
            images.append(image_id)
        if len(selected) >= max_items:
            break

    if not selected:
        for row in ranking_rows[:max_items]:
            node = nodes_by_id.get(clean_text(row.get("node_id")), {})
            if node:
                selected.append({"row": row, "node": node})

    for row in ranking_rows[:12]:
        node = nodes_by_id.get(clean_text(row.get("node_id")), {})
        image_id = image_id_from_node(node)
        if image_id and image_id not in images and clean_text(node.get("node_type")) == "figure":
            images.append(image_id)
        if len(images) >= 3:
            break
    return selected, images[:3]


def format_evidence(question: str, evidence: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(evidence, start=1):
        row = item["row"]
        node = item["node"]
        image_id = image_id_from_node(node)
        image_part = f" | image_id: {image_id}" if image_id else ""
        blocks.append(
            "\n".join(
                [
                    f"[{index}] node_id: {node.get('node_id')} | doc: {node.get('doc_id')} | type: {node.get('node_type')} | score: {row.get('score')}{image_part}",
                    f"source: {node.get('source_ref', '')}",
                    f"text: {focused_content(node, question)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, evidence: list[dict[str, Any]], images: list[str]) -> tuple[str, str]:
    english = is_english(question)
    service_question = is_service_question(question)
    language_rule = "Answer in English." if english else "请用中文回答。"
    image_rule = (
        f"可用图片 id: {json.dumps(images, ensure_ascii=False)}。如果图片能帮助理解步骤、部件、表格或位置，"
        "请在答案末尾追加 `<PIC> ;[\"图片id1\", \"图片id2\"]`，最多 3 张。"
        if images
        else "没有可用图片时不要写 <PIC>。"
    )
    system_prompt = (
        "你是比赛中的高质量多模态客服智能体答案生成器。评分标准重视：直接回应问题、结构清晰、"
        "答案完整有深度、图文互补。不要写空泛套话，不要只说“以平台为准/联系售后”。"
        "只能依据证据和常识客服流程回答；证据不足时也要给出可执行处理步骤。最终只输出 ret 字段内容。"
    )
    style_rule = (
        "这是售后服务问题：请明确责任判断、处理步骤、凭证要求、时效或费用口径；少用模糊话。"
        if service_question
        else "这是商品手册/操作说明问题：请按步骤、部件或注意事项回答，优先提取手册里的具体操作。"
    )
    user_prompt = f"""问题：
{question}

证据：
{format_evidence(question, evidence)}

生成要求：
1. {language_rule}
2. {style_rule}
3. 回答要像客服最终回复，不要说“根据证据1/2”，不要输出 Markdown。
4. 如果问题问“如何做”，请用步骤；如果问“是什么/有哪些”，请列出关键点。
5. {image_rule}
6. 中文控制在 120-350 字，英文控制在 80-220 words。"""
    return system_prompt, user_prompt


def fallback_answer(question: str, evidence: list[dict[str, Any]], images: list[str]) -> str:
    parts: list[str] = []
    for item in evidence[:3]:
        node = item["node"]
        text = focused_content(node, question, 260)
        if text and not text.startswith("Manual illustration"):
            parts.append(text)
    answer = " ".join(parts)
    if not answer:
        answer = "您好，已收到您的问题。建议您提供商品型号、订单信息和具体情况，我们会结合手册或售后规则进一步处理。"
    if images and not is_service_question(question):
        answer += f" <PIC> ;{json.dumps(images, ensure_ascii=False)}"
    return clean_submission_text(answer)


def load_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
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
            ret = clean_text(row.get("ret"))
            if qid and ret:
                cache[qid] = ret
    return cache


def append_cache(path: Path, qid: str, ret: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"id": qid, "ret": ret}, ensure_ascii=False) + "\n")


def write_submission(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["id"]) if row["id"].isdigit() else row["id"])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(rows)


_thread_state = threading.local()


def thread_chat_client() -> ArkChatClient:
    chat = getattr(_thread_state, "chat", None)
    if chat is None:
        chat = ArkChatClient()
        _thread_state.chat = chat
    return chat


def generate_answer(
    index: int,
    total: int,
    question_row: dict[str, str],
    nodes_by_id: dict[str, dict[str, Any]],
    rankings: dict[str, list[dict[str, str]]],
    max_evidence: int,
    use_llm: bool,
) -> tuple[dict[str, str], str]:
    qid = submission_id(question_row.get("question_id", ""))
    question = clean_text(question_row.get("question"))
    ranking_key = clean_text(question_row.get("question_id"))
    evidence, images = select_evidence(question, rankings.get(ranking_key, []), nodes_by_id, max_evidence)
    try:
        if use_llm:
            system_prompt, user_prompt = build_prompt(question, evidence, images)
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    ret = thread_chat_client().complete(system_prompt, user_prompt, temperature=0.15, max_tokens=620)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
            else:
                raise last_error or ArkError("LLM request failed.")
        else:
            ret = fallback_answer(question, evidence, images)
    except Exception as exc:
        ret = fallback_answer(question, evidence, images)
        return {"id": qid, "ret": clean_submission_text(ret)}, (
            f"[{index}/{total}] id={qid} LLM failed, fallback used: {exc}"
        )

    ret = clean_submission_text(ret)
    if images and not is_service_question(question) and "<PIC>" not in ret:
        ret = clean_submission_text(f"{ret} <PIC> ;{json.dumps(images, ensure_ascii=False)}")
    return {"id": qid, "ret": ret}, f"[{index}/{total}] id={qid}: {preview(ret, 100)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a rubric-aligned DataFountain submission with Ark LLM.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--rankings", default=DEFAULT_RANKINGS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default="", help="Comma-separated submission ids to generate, for example: 1,94,222")
    parser.add_argument("--max-evidence", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent LLM calls.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    questions = [row for row in read_csv(args.questions) if clean_text(row.get("question"))]
    if args.ids:
        wanted_ids = {item.strip() for item in args.ids.split(",") if item.strip()}
        questions = [row for row in questions if submission_id(row.get("question_id", "")) in wanted_ids]
    if args.limit:
        questions = questions[: args.limit]
    nodes_by_id = {clean_text(node.get("node_id")): node for node in read_jsonl(args.nodes) if clean_text(node.get("node_id"))}
    rankings = load_rankings(args.rankings)
    output = resolve_path(args.output)
    cache_path = resolve_path(args.cache)
    cached = load_cache(cache_path) if args.resume else {}
    rows: list[dict[str, str]] = []
    pending: list[tuple[int, dict[str, str]]] = []
    for index, question_row in enumerate(questions, start=1):
        qid = submission_id(question_row.get("question_id", ""))
        if qid in cached:
            rows.append({"id": qid, "ret": cached[qid]})
            continue
        pending.append((index, question_row))

    if rows:
        write_submission(output, rows)
        print(f"Loaded {len(rows)} cached rows. Pending {len(pending)} rows.")

    use_llm = not args.no_llm
    workers = max(1, args.workers)
    if workers == 1:
        for index, question_row in pending:
            row, message = generate_answer(
                index,
                len(questions),
                question_row,
                nodes_by_id,
                rankings,
                args.max_evidence,
                use_llm,
            )
            rows.append(row)
            append_cache(cache_path, row["id"], row["ret"])
            if len(rows) % 5 == 0 or len(rows) == 1:
                write_submission(output, rows)
            print(message, flush=True)
            time.sleep(0.05)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    generate_answer,
                    index,
                    len(questions),
                    question_row,
                    nodes_by_id,
                    rankings,
                    args.max_evidence,
                    use_llm,
                )
                for index, question_row in pending
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                row, message = future.result()
                rows.append(row)
                append_cache(cache_path, row["id"], row["ret"])
                if len(rows) % 5 == 0 or done == len(futures):
                    write_submission(output, rows)
                print(message, flush=True)

    write_submission(output, rows)
    print(f"Wrote {len(rows)} rows to {output}")
    if rows:
        print(json.dumps(rows[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
