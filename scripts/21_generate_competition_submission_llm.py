from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ark_clients import ArkChatClient, ArkError
from pipeline_common import clean_text, preview, read_csv, read_jsonl, resolve_path


DEFAULT_QUESTIONS = "outputs/after_sales_kb/questions.csv"
DEFAULT_NODES = "outputs/after_sales_kb/nodes.jsonl"
DEFAULT_RANKINGS = "outputs/after_sales_kb/reranked.csv"
DEFAULT_VISUAL_INDEX = "outputs/visual_index/images.jsonl"
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
    "return",
    "refund",
    "exchange",
    "warranty",
    "repair",
    "complaint",
)

STRONG_SERVICE_TERMS = (
    "\u9000\u8d27",
    "\u6362\u8d27",
    "\u9000\u6b3e",
    "\u65e0\u7406\u7531",
    "\u8fd0\u8d39",
    "\u53d1\u7968",
    "\u5f00\u7968",
    "\u7269\u6d41",
    "\u5feb\u9012",
    "\u6295\u8bc9",
    "\u552e\u540e",
    "\u4fdd\u4fee",
    "\u7ef4\u4fee",
    "\u7834\u635f",
    "\u6c61\u6e0d",
    "\u62c6\u5c01",
    "\u5c11\u53d1",
    "\u9519\u53d1",
    "\u8ba2\u5355",
    "\u8865\u53d1",
    "\u5047\u8d27",
    "\u4e8c\u624b",
    "\u8d54\u507f",
    "\u8d28\u4fdd",
    "return",
    "refund",
    "exchange",
    "warranty",
    "complaint",
    "shipping",
    "delivery",
    "freight",
    "invoice",
    "order",
)

WEAK_SERVICE_TERMS = ("\u5305\u88c5", "repair")

MANUAL_IMAGE_EXTRA_TERMS = (
    "\u5305\u88c5\u76d2",
    "\u914d\u4ef6",
    "\u6e05\u5355",
    "\u5305\u542b",
    "\u5185\u5bb9\u7269",
    "\u5b89\u5168",
    "\u6ce8\u610f",
    "\u8b66\u544a",
    "included",
    "package",
    "box",
    "accessory",
    "safe",
    "safety",
    "warning",
    "attention",
    "finger",
)

MANUAL_IMAGE_TERMS = (
    "如何",
    "怎么",
    "步骤",
    "安装",
    "拆卸",
    "拆除",
    "更换",
    "清洁",
    "连接",
    "设置",
    "调节",
    "调整",
    "打开",
    "关闭",
    "添加",
    "使用",
    "按钮",
    "接口",
    "指示灯",
    "部件",
    "位置",
    "图",
    "图片",
    "install",
    "remove",
    "replace",
    "clean",
    "connect",
    "adjust",
    "set up",
    "setup",
    "open",
    "close",
    "steps",
    "how",
    "what should i do",
    "turn on",
    "turn off",
    "reset",
    "charge",
    "flush",
    "inspect",
    "check",
    "maintain",
    "maintenance",
    "move forward",
    "sail",
    "cleaning",
    "button",
    "interface",
    "indicator",
    "led",
    "part",
    "position",
    "diagram",
)

UNCERTAIN_ANSWER_TERMS = (
    "没有相关信息",
    "未包含",
    "未提及",
    "未记录",
    "暂未",
    "无法回答",
    "no relevant information",
    "does not include",
    "does not contain",
    "not include specific",
    "not contain complete",
    "provided reference material does not",
    "currently provided reference",
)

QUESTION_STOP_TERMS = {
    "\u8bf7\u95ee",
    "\u5982\u4f55",
    "\u600e\u4e48",
    "\u4ec0\u4e48",
    "\u54ea\u4e9b",
    "\u53ef\u4ee5",
    "\u9700\u8981",
    "\u662f\u5426",
    "\u5546\u54c1",
    "\u4ea7\u54c1",
    "\u6211\u7684",
    "\u4f60\u4eec",
    "\u4e00\u4e0b",
    "\u8fd9\u4e2a",
    "\u90a3\u4e2a",
    "\u539f\u56e0",
    "\u600e\u4e48\u529e",
    "what",
    "which",
    "please",
    "need",
    "product",
    "item",
}

ANSWER_PROFILE_TERMS = {
    "shipping": (
        "\u7269\u6d41",
        "\u5feb\u9012",
        "\u8fd0\u8d39",
        "\u53d1\u8d27",
        "\u63fd\u6536",
        "\u7b7e\u6536",
        "\u9001\u8fbe",
        "\u4e61\u9547",
        "shipping",
        "delivery",
        "freight",
    ),
    "invoice": ("\u53d1\u7968", "\u5f00\u7968", "\u7968\u636e", "invoice", "receipt"),
    "return_refund": (
        "\u9000\u8d27",
        "\u6362\u8d27",
        "\u9000\u6b3e",
        "\u65e0\u7406\u7531",
        "\u9000\u6362",
        "return",
        "refund",
        "exchange",
    ),
    "warranty_repair": (
        "\u4fdd\u4fee",
        "\u7ef4\u4fee",
        "\u8d28\u4fdd",
        "\u6545\u969c",
        "\u635f\u574f",
        "warranty",
        "repair",
        "fault",
        "broken",
    ),
    "operation": (
        "\u4f7f\u7528",
        "\u64cd\u4f5c",
        "\u8bbe\u7f6e",
        "\u5b89\u88c5",
        "\u62c6\u5378",
        "\u66f4\u6362",
        "\u6e05\u6d01",
        "\u8fde\u63a5",
        "use",
        "operate",
        "set",
        "install",
        "replace",
        "clean",
        "connect",
    ),
    "visual_part": (
        "\u6309\u94ae",
        "\u6307\u793a\u706f",
        "\u90e8\u4ef6",
        "\u4f4d\u7f6e",
        "\u5c3a\u5bf8",
        "\u8868\u5e26",
        "\u63a5\u53e3",
        "\u5305\u88c5\u76d2",
        "\u914d\u4ef6",
        "\u6e05\u5355",
        "button",
        "indicator",
        "led",
        "part",
        "position",
        "size",
        "strap",
        "package",
        "box",
        "accessory",
    ),
}

ANSWER_PROFILE_INSTRUCTIONS = {
    "shipping": "Use this structure: direct logistics conclusion, likely reason, handling steps, fee/time caveat.",
    "invoice": "Use this structure: whether invoice is supported, required order information, issuing or correction steps.",
    "return_refund": "Use this structure: whether return/refund/exchange is supported, eligibility, proof required, next steps.",
    "warranty_repair": "Use this structure: responsibility judgment, warranty/repair handling, required proof, escalation condition.",
    "operation": "Use this structure: numbered operation steps, buttons/parts involved, safety or reset notes.",
    "visual_part": "Use this structure: identify the part/status first, then explain the visual cue and the action to take.",
    "general": "Use a concise customer-service answer with conclusion first, then concrete steps and caveats.",
}


def submission_id(question_id: str) -> str:
    match = re.search(r"(\d+)$", clean_text(question_id))
    return match.group(1) if match else clean_text(question_id)


def is_english(text: str) -> bool:
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return letters > max(20, cjk * 2)


def term_matches(text: str, term: str) -> bool:
    blob = clean_text(text).casefold()
    needle = clean_text(term).casefold()
    if not needle:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 _-]*", needle):
        pattern = r"\b" + re.escape(needle).replace(r"\ ", r"\s+") + r"s?\b"
        return bool(re.search(pattern, blob))
    return needle in blob


def is_service_question(question: str) -> bool:
    blob = clean_text(question).casefold()
    for term in STRONG_SERVICE_TERMS:
        if term == "order":
            if re.search(r"\b(my|purchase|sales|order\s+number)\s+order\b|\border\s+(number|id|status|refund)\b", blob):
                return True
            continue
        if term_matches(blob, term):
            return True
    return False


def is_manual_visual_question(question: str) -> bool:
    blob = clean_text(question).casefold()
    if is_service_question(question):
        return False
    return any(term.casefold() in blob for term in (*MANUAL_IMAGE_TERMS, *MANUAL_IMAGE_EXTRA_TERMS))


def is_uncertain_answer(text: str) -> bool:
    blob = clean_text(text).casefold()
    return any(term.casefold() in blob for term in UNCERTAIN_ANSWER_TERMS)


def _contains_any(text: str, terms: tuple[str, ...] | list[str] | set[str]) -> bool:
    return any(term_matches(text, term) for term in terms if clean_text(term))


def important_question_terms(text: str) -> list[str]:
    blob = clean_text(text).casefold()
    terms: list[str] = []
    for word in re.findall(r"[a-z][a-z0-9_-]{2,}", blob):
        if word not in QUESTION_STOP_TERMS and word not in terms:
            terms.append(word)
    for seq in re.findall(r"[\u4e00-\u9fff]{2,}", blob):
        if seq not in QUESTION_STOP_TERMS and len(seq) <= 10 and seq not in terms:
            terms.append(seq)
        window = seq[:32]
        for size in (4, 3, 2):
            if len(window) < size:
                continue
            for index in range(0, len(window) - size + 1):
                gram = window[index : index + size]
                if gram in QUESTION_STOP_TERMS:
                    continue
                if any(stop in gram for stop in QUESTION_STOP_TERMS if len(stop) >= 2 and len(gram) <= len(stop) + 1):
                    continue
                if gram not in terms:
                    terms.append(gram)
    return terms[:48]


def answer_profile(question: str) -> str:
    blob = clean_text(question).casefold()
    if _contains_any(blob, ANSWER_PROFILE_TERMS["shipping"]):
        return "shipping"
    if _contains_any(blob, ANSWER_PROFILE_TERMS["invoice"]):
        return "invoice"
    if _contains_any(blob, ANSWER_PROFILE_TERMS["return_refund"]):
        return "return_refund"
    if is_service_question(question) and _contains_any(blob, ANSWER_PROFILE_TERMS["warranty_repair"]):
        return "warranty_repair"
    if _contains_any(blob, ANSWER_PROFILE_TERMS["visual_part"]):
        return "visual_part"
    if _contains_any(blob, ANSWER_PROFILE_TERMS["operation"]) or is_manual_visual_question(question):
        return "operation"
    if _contains_any(blob, ANSWER_PROFILE_TERMS["warranty_repair"]):
        return "warranty_repair"
    return "general"


def answer_profile_instruction(question: str) -> str:
    profile = answer_profile(question)
    return ANSWER_PROFILE_INSTRUCTIONS.get(profile, ANSWER_PROFILE_INSTRUCTIONS["general"])


def node_has_visual_caption(node: dict[str, Any]) -> bool:
    return bool(clean_text(node.get("visual_caption")) or clean_text(node.get("qa_evidence")))


def image_node_is_useful(node: dict[str, Any], question: str) -> bool:
    if clean_text(node.get("node_type")) != "figure":
        return False
    if not is_manual_visual_question(question):
        return False
    if node_has_visual_caption(node):
        return True
    return bool(clean_text(node.get("source_ref")) or clean_text(node.get("content")))


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


def normalize_pic_suffix(text: str) -> str:
    text = clean_text(text)
    pic_match = re.search(r"\s*<PIC>\s*(.*?)\s*(?:</PIC>)?\s*$", text)
    if not pic_match:
        return text

    raw_ids = pic_match.group(1).strip().lstrip(";").strip()
    base_text = text[: pic_match.start()].strip()
    if not raw_ids:
        return base_text

    if raw_ids.startswith("["):
        try:
            parsed = json.loads(raw_ids)
        except json.JSONDecodeError:
            parsed = re.findall(r"[A-Za-z0-9_\-]+", raw_ids)
    else:
        parsed = re.findall(r"[A-Za-z0-9_\-]+", raw_ids)
    image_ids = []
    for item in parsed if isinstance(parsed, list) else []:
        image_id = clean_text(item)
        if image_id and image_id not in image_ids:
            image_ids.append(image_id)
        if len(image_ids) >= 3:
            break
    if not image_ids:
        return base_text
    return f"{base_text} <PIC> ;{json.dumps(image_ids, ensure_ascii=False)}".strip()


def clean_submission_text(text: str, limit: int = 760) -> str:
    text = clean_text(text).replace("```", "").strip(" {}")
    text = re.sub(r"\s*#{1,6}\s*", " ", text)
    text = re.sub(r"\s*Evidence note:\s*[^<。.]*(?:[。.]|$)", " ", text, flags=re.I)
    text = re.sub(r"^\s*(?:Answer\s+profile|Profile\s+rule|Template)\s*:\s*[A-Za-z_ -]+\.?\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*(?:Profile|Answer\s+style)\s*:\s*[A-Za-z_ -]+\.?\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*(?:Use\s+this\s+structure|Evidence\s+note)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*:\s*", "", text)
    text = re.sub(r"^\s*(答案|输出)\s*[:：]?\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*ret\b\s*[:：]?\s*", "", text, flags=re.I)
    text = normalize_pic_suffix(text)
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
        if sep == "." and pos > 0 and cut[pos - 1].isdigit():
            continue
        if pos >= len(cut) * 0.55:
            cut = cut[: pos + 1]
            break
    else:
        cut = cut.rstrip("，,、；;：: ") + ("." if is_english(cut) else "。")
    return f"{cut} {pic_part}".strip() if pic_part else cut


def strip_pic_suffix(text: str) -> str:
    return clean_text(
        re.sub(
            r"\s*<PIC>\s*(?:;?\s*\[[^\]]+\]|;?\s*[A-Za-z0-9_\-]+|[A-Za-z0-9_\-]+\s*</PIC>)\s*$",
            "",
            text,
        )
    )


def safe_print(message: Any, flush: bool = True) -> None:
    text = str(message)
    encoding = sys.stdout.encoding or "utf-8"
    try:
        print(text, flush=flush)
    except UnicodeEncodeError:
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=flush)


def focused_content(node: dict[str, Any], question: str, limit: int = 900) -> str:
    content = clean_text(
        " ".join(
            [
                node.get("section", ""),
                node.get("previous_chunk_preview", ""),
                node.get("content", ""),
                node.get("next_chunk_preview", ""),
                node.get("visual_title", ""),
                node.get("key_objects", ""),
                node.get("ocr_text", ""),
                node.get("data_or_trends", ""),
                node.get("qa_evidence", ""),
                node.get("visual_caption", ""),
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


def load_rankings(path: str | Path, method: str = "G4") -> dict[str, list[dict[str, str]]]:
    wanted_method = clean_text(method).upper() or "G4"
    groups: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(path):
        if clean_text(row.get("method")).upper() != wanted_method:
            continue
        qid = clean_text(row.get("question_id"))
        if qid:
            groups.setdefault(qid, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row.get("rank") or 9999))
    return groups


def load_visual_index(path: str | Path) -> list[dict[str, Any]]:
    file_path = resolve_path(path)
    if not file_path.exists():
        return []
    return read_jsonl(file_path)


def relevance_score(question: str, text: str) -> float:
    question = clean_text(question).casefold()
    text = clean_text(text).casefold()
    if not question or not text:
        return 0.0
    q_terms = important_question_terms(question)
    if not q_terms:
        return 0.0
    total_weight = 0.0
    hit_weight = 0.0
    for term in q_terms:
        weight = 1.0 + min(1.0, max(0, len(term) - 2) * 0.2)
        total_weight += weight
        if term in text:
            hit_weight += weight
    coverage = hit_weight / max(1.0, total_weight)
    density = hit_weight / max(18.0, len(text) / 80.0)
    return min(1.0, 0.88 * coverage + 0.35 * density)


def visual_cue_bonus(question: str, text: str) -> float:
    bonus = 0.0
    for profile in ("operation", "visual_part", "warranty_repair"):
        terms = ANSWER_PROFILE_TERMS[profile]
        if _contains_any(question, terms) and _contains_any(text, terms):
            bonus += 0.08
    return min(0.2, bonus)


def min_visual_image_score(question: str) -> float:
    if _contains_any(question, ANSWER_PROFILE_TERMS["visual_part"]):
        return 0.16
    if _contains_any(question, ANSWER_PROFILE_TERMS["operation"]):
        return 0.13
    return 0.1


def visual_candidate_text(node: dict[str, Any], visual_row: dict[str, Any] | None = None) -> str:
    visual_row = visual_row or {}
    return clean_text(
        " ".join(
            [
                node.get("doc_id", ""),
                node.get("section", ""),
                node.get("source_ref", ""),
                node.get("content", ""),
                node.get("visual_title", ""),
                node.get("key_objects", ""),
                node.get("ocr_text", ""),
                node.get("data_or_trends", ""),
                node.get("qa_evidence", ""),
                node.get("visual_caption", ""),
                node.get("visual_summary", ""),
                visual_row.get("visual_title", ""),
                visual_row.get("key_objects", ""),
                visual_row.get("ocr_text", ""),
                visual_row.get("qa_evidence", ""),
                visual_row.get("visual_caption", ""),
            ]
        )
    )


def rerank_images_second_stage(
    question: str,
    ranking_rows: list[dict[str, str]],
    nodes_by_id: dict[str, dict[str, Any]],
    visual_index: list[dict[str, Any]],
    max_images: int = 3,
) -> list[str]:
    if not is_manual_visual_question(question):
        return []
    visual_by_node = {clean_text(row.get("node_id")): row for row in visual_index if clean_text(row.get("node_id"))}
    candidate_nodes: dict[str, tuple[dict[str, Any], dict[str, Any], float]] = {}
    for rank_pos, row in enumerate(ranking_rows[:30], start=1):
        node = nodes_by_id.get(clean_text(row.get("node_id")), {})
        image_id = image_id_from_node(node)
        if not image_id or not image_node_is_useful(node, question):
            continue
        rank_bonus = max(0.0, (31 - rank_pos) / 30.0)
        score = 0.12 * rank_bonus
        try:
            ranking_score = float(row.get("score") or 0.0)
        except (TypeError, ValueError):
            ranking_score = 0.0
        score += 0.18 * max(0.0, ranking_score)
        candidate_nodes[image_id] = (node, visual_by_node.get(clean_text(node.get("node_id")), {}), score)

    ranking_node_ids = {clean_text(row.get("node_id")) for row in ranking_rows[:30]}
    for visual_row in visual_index:
        node_id = clean_text(visual_row.get("node_id"))
        if node_id not in ranking_node_ids:
            continue
        node = nodes_by_id.get(node_id, {})
        image_id = clean_text(visual_row.get("image_id")) or image_id_from_node(node)
        if not image_id or not image_node_is_useful(node, question):
            continue
        candidate_nodes.setdefault(image_id, (node, visual_row, 0.1))

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for image_id, (node, visual_row, base_score) in candidate_nodes.items():
        text = visual_candidate_text(node, visual_row)
        score = base_score + 0.68 * relevance_score(question, text) + visual_cue_bonus(question, text)
        if node_has_visual_caption(node) or clean_text(visual_row.get("qa_evidence")):
            score += 0.16
        if clean_text(node.get("ocr_text")) or clean_text(visual_row.get("ocr_text")):
            score += 0.04
        scored.append((score, image_id, node))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[str] = []
    uncaptioned = 0
    min_score = min_visual_image_score(question)
    for score, image_id, node in scored:
        if score < min_score:
            continue
        if not node_has_visual_caption(node):
            if uncaptioned >= 1:
                continue
            uncaptioned += 1
        selected.append(image_id)
        if len(selected) >= max_images:
            break
    if not selected and scored and scored[0][0] >= min_score * 0.65:
        selected.append(scored[0][1])
    return selected


def _source_topic(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s*/\s*(?:segment|illustration)\s+\d+.*$", "", text, flags=re.I)
    text = re.sub(r"\s*/\s*[A-Za-z0-9_-]+\s*$", "", text)
    return text.strip()


_VISUAL_INDEX_BY_DOC_CACHE: dict[int, dict[str, list[dict[str, Any]]]] = {}


def visual_index_by_doc(visual_index: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cache_key = id(visual_index)
    cached = _VISUAL_INDEX_BY_DOC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in visual_index:
        doc_id = clean_text(row.get("doc_id"))
        if doc_id:
            by_doc.setdefault(doc_id, []).append(row)
    _VISUAL_INDEX_BY_DOC_CACHE[cache_key] = by_doc
    return by_doc


def relevance_score_from_terms(q_terms: list[str], text: str) -> float:
    text = clean_text(text).casefold()
    if not q_terms or not text:
        return 0.0
    total_weight = 0.0
    hit_weight = 0.0
    for term in q_terms:
        weight = 1.0 + min(1.0, max(0, len(term) - 2) * 0.2)
        total_weight += weight
        if term in text:
            hit_weight += weight
    coverage = hit_weight / max(1.0, total_weight)
    density = hit_weight / max(18.0, len(text) / 80.0)
    return min(1.0, 0.88 * coverage + 0.35 * density)


def global_visual_fallback_images(
    question: str,
    evidence: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    visual_index: list[dict[str, Any]],
    max_images: int = 3,
) -> list[str]:
    if not is_manual_visual_question(question) or not evidence or not visual_index:
        return []

    evidence_doc_ids = {
        clean_text(item["node"].get("doc_id"))
        for item in evidence
        if clean_text(item["node"].get("doc_id"))
    }
    evidence_topics = {
        _source_topic(clean_text(item["node"].get("source_ref")) or clean_text(item["node"].get("section")))
        for item in evidence
    }
    evidence_topics = {topic for topic in evidence_topics if topic}
    q_terms = important_question_terms(question)
    min_threshold = max(0.08, min_visual_image_score(question) * 0.55)

    scored: list[tuple[float, str, dict[str, Any]]] = []
    candidate_rows: list[dict[str, Any]] = []
    by_doc = visual_index_by_doc(visual_index)
    for doc_id in evidence_doc_ids:
        candidate_rows.extend(by_doc.get(doc_id, []))
    if not candidate_rows:
        candidate_rows = visual_index

    for visual_row in candidate_rows:
        image_id = clean_text(visual_row.get("image_id"))
        if not image_id:
            continue
        node = nodes_by_id.get(clean_text(visual_row.get("node_id")), {}) or {"node_type": "figure", **visual_row}
        doc_id = clean_text(visual_row.get("doc_id")) or clean_text(node.get("doc_id"))
        if evidence_doc_ids and doc_id and doc_id not in evidence_doc_ids:
            continue

        source = clean_text(visual_row.get("source_ref")) or clean_text(node.get("source_ref"))
        topic = _source_topic(source)
        same_topic = any(topic and (topic == ev_topic or topic.startswith(ev_topic) or ev_topic.startswith(topic)) for ev_topic in evidence_topics)
        text = visual_candidate_text(node, visual_row)
        score = 0.72 * relevance_score_from_terms(q_terms, text)
        if doc_id in evidence_doc_ids:
            score += 0.12
        if same_topic:
            score += 0.28
        if clean_text(visual_row.get("qa_evidence")) or clean_text(visual_row.get("visual_caption")):
            score += 0.12
        if clean_text(visual_row.get("ocr_text")):
            score += 0.04
        if score >= min_threshold:
            scored.append((score, image_id, node))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    for _score, image_id, _node in scored:
        if image_id not in selected:
            selected.append(image_id)
        if len(selected) >= max_images:
            break
    return selected


def select_evidence(
    question: str,
    ranking_rows: list[dict[str, str]],
    nodes_by_id: dict[str, dict[str, Any]],
    max_items: int,
    visual_index: list[dict[str, Any]] | None = None,
    visual_second_stage: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    service_question = is_service_question(question)
    allow_images = is_manual_visual_question(question)
    selected: list[dict[str, Any]] = []
    images: list[str] = []
    uncaptioned_images = 0
    seen: set[str] = set()

    def row_priority(row: dict[str, str]) -> tuple[float, int]:
        node = nodes_by_id.get(clean_text(row.get("node_id")), {})
        node_id = clean_text(node.get("node_id"))
        node_type = clean_text(node.get("node_type"))
        structure_type = clean_text(node.get("structure_type"))
        rank = int(row.get("rank") or 9999)
        penalty = 0.0
        if node_id.startswith("AS_PROFILE_") or structure_type == "manual_profile":
            penalty += 20.0
        if node_type == "title":
            penalty += 4.0
        elif node_type == "page":
            penalty += 8.0
        elif node_type in {"figure", "caption"} and not node_has_visual_caption(node):
            penalty += 3.0
        elif node_type == "text":
            penalty -= 1.5
        if service_question and node_type in {"figure", "caption"}:
            penalty += 6.0
        return penalty + rank, rank

    evidence_rows = sorted(ranking_rows[:30], key=row_priority)

    for row in evidence_rows:
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
        if allow_images and image_id and image_id not in images and image_node_is_useful(node, question):
            if not node_has_visual_caption(node) and uncaptioned_images >= 1:
                pass
            else:
                if not node_has_visual_caption(node):
                    uncaptioned_images += 1
                images.append(image_id)
        if len(selected) >= max_items:
            break

    if not selected:
        for row in ranking_rows[:max_items]:
            node = nodes_by_id.get(clean_text(row.get("node_id")), {})
            if node:
                selected.append({"row": row, "node": node})

    if allow_images and visual_second_stage:
        reranked_images = rerank_images_second_stage(
            question,
            ranking_rows,
            nodes_by_id,
            visual_index or [],
            max_images=3,
        )
        if reranked_images:
            images = reranked_images

    if allow_images and not images:
        fallback_candidates: list[tuple[float, str, dict[str, Any]]] = []
        for row in ranking_rows[:12]:
            node = nodes_by_id.get(clean_text(row.get("node_id")), {})
            image_id = image_id_from_node(node)
            if image_id and image_id not in images and image_node_is_useful(node, question):
                text = visual_candidate_text(node, {})
                try:
                    base = float(row.get("score") or 0.0)
                except (TypeError, ValueError):
                    base = 0.0
                score = 0.15 * max(0.0, base) + 0.75 * relevance_score(question, text) + visual_cue_bonus(question, text)
                if node_has_visual_caption(node):
                    score += 0.12
                fallback_candidates.append((score, image_id, node))
        fallback_candidates.sort(key=lambda item: item[0], reverse=True)
        min_score = min_visual_image_score(question) * 0.7
        for score, image_id, node in fallback_candidates:
            if score < min_score and images:
                continue
            if not node_has_visual_caption(node) and uncaptioned_images >= 1:
                continue
            if not node_has_visual_caption(node):
                uncaptioned_images += 1
            images.append(image_id)
            if len(images) >= 3:
                break
    if allow_images and len(images) < 2:
        for image_id in global_visual_fallback_images(
            question,
            selected,
            nodes_by_id,
            visual_index or [],
            max_images=3,
        ):
            if image_id not in images:
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


def evidence_quality_hint(question: str, evidence: list[dict[str, Any]], images: list[str]) -> str:
    if not evidence:
        return "Evidence note: no retrieved evidence was selected; give a conservative answer and do not invent image ids."
    node_types = [clean_text(item["node"].get("node_type")) for item in evidence]
    has_visual_node = any(node_type in {"figure", "table", "caption"} for node_type in node_types)
    has_visual_text = any(
        clean_text(item["node"].get(field))
        for item in evidence
        for field in ("visual_caption", "qa_evidence", "ocr_text", "key_objects", "visual_summary")
    )
    notes: list[str] = []
    if is_manual_visual_question(question):
        if images:
            notes.append("Use the selected image ids only when they support the operation, part, label, table, or safety cue.")
        elif has_visual_node or has_visual_text:
            notes.append("Visual evidence exists in the text fields, but no valid image id was selected; answer from the visual text and do not append PIC.")
        else:
            notes.append("No reliable image was selected; answer with concrete manual steps from the text evidence and do not append PIC.")
    if answer_profile(question) in {"visual_part", "operation"}:
        notes.append("For parts, indicators, buttons, sizes, setup, or safety questions, prefer concrete labels and steps over saying the manual does not include the information.")
    if not notes:
        notes.append("Use only the strongest retrieved evidence and keep the answer direct.")
    return "Evidence note: " + " ".join(notes)


def build_prompt(question: str, evidence: list[dict[str, Any]], images: list[str]) -> tuple[str, str]:
    english = is_english(question)
    service_question = is_service_question(question)
    language_rule = "Answer in English." if english else "请用中文回答。"
    if service_question:
        image_rule = "这是售后政策、维修、投诉或交易规则类问题，不要追加 <PIC>；只给处理流程和责任口径。"
    elif images:
        image_rule = (
            f"可用图片 id: {json.dumps(images, ensure_ascii=False)}。如果图片能帮助理解步骤、部件、表格或位置，"
            "请在答案末尾追加 `<PIC> ;[\"图片id1\", \"图片id2\"]`，最多 3 张；不要编造不在列表中的图片 id。"
        )
    else:
        image_rule = "没有可用图片时不要写 <PIC>。"
    system_prompt = (
        "你是比赛中的高质量多模态客服智能体答案生成器。评分标准重视：直接回应问题、结构清晰、"
        "答案完整有深度、图文互补。不要写空泛套话，不要只说“以平台为准/联系售后”。"
        "只能依据证据和常识客服流程回答；证据不足时也要优先根据最相关证据和产品常识给出可执行步骤，"
        "不要把答案写成“资料未提供/无法回答”。不要复制内部标签，例如 Answer profile、Profile rule、Template。"
        "最终只输出 ret 字段内容。"
    )
    style_rule = (
        "这是售后服务问题：请明确责任判断、处理步骤、凭证要求、时效或费用口径；少用模糊话。"
        if service_question
        else "这是商品手册/操作说明问题：请按步骤、部件或注意事项回答，优先提取手册里的具体操作。"
    )
    profile_rule = answer_profile_instruction(question)
    evidence_note = evidence_quality_hint(question, evidence, images)
    user_prompt = f"""问题：
{question}

证据：
{format_evidence(question, evidence)}

生成要求：
1. {language_rule}
2. {style_rule}
Answer style guidance, do not copy this line: {profile_rule}
Evidence quality guidance, do not copy this line: {evidence_note}
3. 回答要像客服最终回复，不要说“根据证据1/2”，不要输出 Markdown。
4. 如果问题问“如何做”，请用步骤；如果问“是什么/有哪些”，请列出关键点。
5. 不要输出“资料中没有相关信息/does not include”等拒答式话术；如果证据不完整，请给出保守但可执行的通用步骤并说明注意安全。
6. {image_rule}
7. 中文控制在 120-350 字，英文控制在 80-220 words。"""
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
    if images and is_manual_visual_question(question):
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


def self_check_answer(
    question: str,
    evidence: list[dict[str, Any]],
    images: list[str],
    ret: str,
) -> str:
    if not ret:
        return ret
    service_question = is_service_question(question)
    profile_rule = answer_profile_instruction(question)
    evidence_note = evidence_quality_hint(question, evidence, images)
    image_rule = (
        "This is a policy/service question: remove any <PIC> suffix."
        if service_question
        else f"Only keep a <PIC> suffix if it is useful and the image ids are in this list: {json.dumps(images, ensure_ascii=False)}."
    )
    system_prompt = (
        "You are a strict DataFountain submission answer reviewer. "
        "Revise the answer only when it misses a sub-question, invents unsupported details, has an invalid <PIC> suffix, "
        "or is too vague. Do not turn a concrete draft into a refusal such as 'the reference does not include this'. "
        "Remove any leaked internal labels such as Answer profile, Profile rule, Template, or Evidence note. "
        "Return only the final ret text, with no Markdown and no explanation."
    )
    user_prompt = f"""Question:
{question}

Evidence:
{format_evidence(question, evidence)[:2800]}

Draft answer:
{ret}

Checklist:
Answer style guidance, do not copy: {profile_rule}
Evidence guidance, do not copy: {evidence_note}
1. Directly answer every sub-question.
2. Keep concrete handling steps, required proof, fees/timing caveats, or manual operations when supported.
3. Do not invent exact prices, deadlines, promises, or image ids.
4. Do not answer with "no relevant information" when the draft already provides a reasonable procedure.
5. {image_rule}
6. Chinese: 120-350 characters when possible. English: 80-220 words.
"""
    revised = thread_chat_client().complete(system_prompt, user_prompt, temperature=0.05, max_tokens=520)
    revised = clean_submission_text(revised or ret)
    if is_uncertain_answer(revised) and not is_uncertain_answer(ret):
        return clean_submission_text(ret)
    return revised


def generate_answer(
    index: int,
    total: int,
    question_row: dict[str, str],
    nodes_by_id: dict[str, dict[str, Any]],
    rankings: dict[str, list[dict[str, str]]],
    max_evidence: int,
    use_llm: bool,
    use_self_check: bool,
    visual_index: list[dict[str, Any]],
    visual_second_stage: bool,
) -> tuple[dict[str, str], str]:
    qid = submission_id(question_row.get("question_id", ""))
    question = clean_text(question_row.get("question"))
    ranking_key = clean_text(question_row.get("question_id"))
    evidence, images = select_evidence(
        question,
        rankings.get(ranking_key, []),
        nodes_by_id,
        max_evidence,
        visual_index=visual_index,
        visual_second_stage=visual_second_stage,
    )
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
    if use_llm and use_self_check:
        draft_ret = ret
        try:
            ret = self_check_answer(question, evidence, images, ret)
        except Exception as exc:
            safe_print(f"[{index}/{total}] id={qid} self-check skipped: {exc}")
            ret = draft_ret
        if is_uncertain_answer(ret) and not is_uncertain_answer(draft_ret):
            ret = draft_ret
    if is_service_question(question):
        ret = clean_submission_text(strip_pic_suffix(ret))
    if images and is_manual_visual_question(question) and "<PIC>" not in ret:
        ret = clean_submission_text(f"{ret} <PIC> ;{json.dumps(images, ensure_ascii=False)}")
    return {"id": qid, "ret": ret}, f"[{index}/{total}] id={qid}: {preview(ret, 100)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a rubric-aligned DataFountain submission with Ark LLM.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--rankings", default=DEFAULT_RANKINGS)
    parser.add_argument("--ranking-method", default="G4", help="Rerank method to use from --rankings, e.g. G0/G1/G2/G3/G4.")
    parser.add_argument("--visual-index", default=DEFAULT_VISUAL_INDEX)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default="", help="Comma-separated submission ids to generate, for example: 1,94,222")
    parser.add_argument("--max-evidence", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent LLM calls.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--skip-visual-second-stage", action="store_true")
    args = parser.parse_args()

    questions = [row for row in read_csv(args.questions) if clean_text(row.get("question"))]
    if args.ids:
        wanted_ids = {item.strip() for item in args.ids.split(",") if item.strip()}
        questions = [row for row in questions if submission_id(row.get("question_id", "")) in wanted_ids]
    if args.limit:
        questions = questions[: args.limit]
    nodes_by_id = {clean_text(node.get("node_id")): node for node in read_jsonl(args.nodes) if clean_text(node.get("node_id"))}
    rankings = load_rankings(args.rankings, args.ranking_method)
    visual_index = load_visual_index(args.visual_index)
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
        safe_print(f"Loaded {len(rows)} cached rows. Pending {len(pending)} rows.")

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
                not args.skip_self_check,
                visual_index,
                not args.skip_visual_second_stage,
            )
            rows.append(row)
            append_cache(cache_path, row["id"], row["ret"])
            if len(rows) % 5 == 0 or len(rows) == 1:
                write_submission(output, rows)
            safe_print(message)
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
                    not args.skip_self_check,
                    visual_index,
                    not args.skip_visual_second_stage,
                )
                for index, question_row in pending
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                row, message = future.result()
                rows.append(row)
                append_cache(cache_path, row["id"], row["ret"])
                if len(rows) % 5 == 0 or done == len(futures):
                    write_submission(output, rows)
                safe_print(message)

    write_submission(output, rows)
    safe_print(f"Wrote {len(rows)} rows to {output}")
    if rows:
        safe_print(json.dumps(rows[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
