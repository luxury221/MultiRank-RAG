from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from pipeline_common import clean_text, preview, read_csv, read_jsonl, resolve_path


DEFAULT_QUESTIONS = "outputs/datafountain_1165/questions.csv"
DEFAULT_NODES = "outputs/datafountain_1165/nodes.jsonl"
DEFAULT_OUTPUT = "outputs/datafountain_1165/submission_fast.csv"


POLICY_PATTERNS = {
    "return_refund": (
        "退换",
        "退货",
        "换货",
        "退款",
        "无理由",
        "运费",
        "return",
        "refund",
        "exchange",
    ),
    "invoice": ("发票", "开票", "invoice"),
    "shipping_damage": ("包装破损", "破损", "物流", "快递", "运输", "shipping", "package"),
    "complaint_repair": ("售后", "维修", "送修", "客服", "投诉", "没修好", "repair", "warranty", "service"),
}


def is_english(text: str) -> bool:
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return letters > max(20, cjk * 2)


def submission_id(question_id: str) -> str:
    question_id = clean_text(question_id)
    if question_id.startswith("DF_"):
        return question_id[3:]
    return question_id


def normalize_question(value: Any) -> str:
    text = clean_text(value)
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip().strip(",").strip('"').strip("'").strip()
        if line:
            parts.append(line)
    return clean_text(" ".join(parts))


def question_kind(question: str) -> str:
    lowered = question.casefold()
    for kind, patterns in POLICY_PATTERNS.items():
        if any(pattern.casefold() in lowered for pattern in patterns):
            return kind
    return "manual"


def node_text(node: dict[str, Any]) -> str:
    fields = [
        node.get("doc_id", ""),
        node.get("section", ""),
        node.get("node_type", ""),
        node.get("visual_summary", ""),
        node.get("previous_chunk_preview", ""),
        node.get("content", ""),
        node.get("next_chunk_preview", ""),
    ]
    return clean_text(" ".join(str(field) for field in fields if field))


def answer_source_text(node: dict[str, Any]) -> str:
    node_type = clean_text(node.get("node_type"))
    if node_type == "figure":
        text = clean_text(
            " ".join(
                [
                    node.get("previous_chunk_preview", ""),
                    node.get("next_chunk_preview", ""),
                ]
            )
        )
        if not text:
            text = clean_text(node.get("content", ""))
    else:
        text = clean_text(node.get("content", ""))
    text = re.sub(r"Manual illustration \d+\..*?Nearby text before:", "相关图示说明：", text)
    text = re.sub(r"Illustration linked to section '.*?' in .*?\.", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def short_sentence(text: str, limit: int = 240) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ["。", "；", ";", ".", "！", "!", "？", "?"]:
        pos = cut.rfind(sep)
        if pos >= limit * 0.45:
            return cut[: pos + 1]
    return cut.rstrip("，,、；;：:") + "。"


def policy_answer(kind: str, question: str) -> str:
    english = is_english(question)
    if english:
        if kind == "return_refund":
            return (
                "Hello, return, exchange, shipping-fee and refund rules should follow the order page and platform "
                "after-sales policy. Please keep the order information and product condition photos; if the product "
                "has a quality issue, contact customer service for verification and further handling."
            )
        if kind == "invoice":
            return (
                "Hello, invoice availability and invoice type depend on the order page and store policy. Please provide "
                "the required invoice title and tax information in the order or contact customer service for confirmation."
            )
        if kind == "shipping_damage":
            return (
                "Hello, if the package is damaged, please keep photos of the outer package, waybill and product condition "
                "first, then contact customer service. If the product is affected, we will verify it according to the "
                "after-sales policy."
            )
        return (
            "Hello, we are sorry for the inconvenience. Please provide your order number, repair record or product model "
            "so that customer service can verify the status and give you a follow-up solution."
        )

    if kind == "return_refund":
        return (
            "您好，是否支持7天无理由退换货、运费由谁承担以及退款到账时间，需以商品页面和平台售后政策为准。"
            "如果是商品质量问题或错发漏发，建议您保留订单、商品照片和包装信息后联系售后核实；符合条件后通常会按原支付渠道处理退款或换货。"
        )
    if kind == "invoice":
        return (
            "您好，是否支持开发票、发票类型以及开具时间，需要以订单页面和店铺开票规则为准。"
            "如页面支持开票，请按要求填写发票抬头、税号等信息；如果无法确认，建议提供订单号联系售后客服核实。"
        )
    if kind == "shipping_damage":
        return (
            "您好，如果收到商品时发现包装破损，请先拍照留存外包装、运单号和商品状态，并尽快联系售后客服。"
            "若商品本身受损或影响正常使用，可根据平台售后规则申请退换货或进一步处理。"
        )
    return (
        "您好，非常抱歉给您带来不便。请您提供订单号、商品型号、送修记录或故障照片，售后会进一步核实维修进度和处理方案。"
        "如果属于保修范围内的问题，会按保修政策协助维修或更换；若为人为损坏或超出保修范围，费用需以检测结果为准。"
    )


def manual_answer(question: str, top_nodes: list[dict[str, Any]], top_score: float) -> str:
    english = is_english(question)
    useful_nodes = [node for node in top_nodes if answer_source_text(node)]
    if not useful_nodes or top_score < 0.035:
        if english:
            return (
                "Hello, the current information is not enough to give a fully confirmed answer. Please provide the product "
                "model, fault description or related image, and customer service will verify it according to the manual."
            )
        return (
            "您好，目前问题信息还不够明确，建议您补充商品型号、具体故障现象或相关图片，售后会结合产品手册和实际情况为您核实处理。"
        )

    first = useful_nodes[0]
    first_doc = clean_text(first.get("doc_id"))
    second = next(
        (
            node
            for node in useful_nodes[1:]
            if clean_text(node.get("doc_id")) == first_doc
            and answer_source_text(node) != answer_source_text(first)
        ),
        None,
    )
    doc = clean_text(first.get("doc_id")) or "相关手册"
    snippet1 = short_sentence(answer_source_text(first), 280 if not english else 320)
    snippet2 = short_sentence(answer_source_text(second), 180 if not english else 240) if second else ""

    if english:
        answer = f"According to {doc}, {snippet1}"
        if snippet2 and snippet2 not in snippet1:
            answer += f" In addition, {snippet2}"
        answer += " If the problem remains after following the manual, please contact after-sales support with the product model and issue details."
        return clean_text(answer)

    answer = f"您好，查询{doc}中的相关说明：{snippet1}"
    if snippet2 and snippet2 not in snippet1:
        answer += f" 另外，{snippet2}"
    answer += " 如果按手册操作后仍无法解决，建议您提供商品型号、故障现象或图片，联系售后进一步核实。"
    return clean_text(answer)


def build_fast_ranker(nodes: list[dict[str, Any]]):
    texts = [node_text(node) for node in nodes]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel

        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        matrix = vectorizer.fit_transform(texts)

        def rank(question: str, top_k: int) -> tuple[list[dict[str, Any]], float]:
            q_vec = vectorizer.transform([question])
            scores = linear_kernel(q_vec, matrix).ravel()
            ranked = scores.argsort()[::-1][:top_k]
            return [nodes[int(index)] for index in ranked], float(scores[int(ranked[0])]) if len(ranked) else 0.0

        return rank
    except Exception:

        def char_score(question: str, text: str) -> float:
            q_chars = set(clean_text(question))
            t_chars = set(clean_text(text))
            if not q_chars or not t_chars:
                return 0.0
            return len(q_chars & t_chars) / len(q_chars | t_chars)

        def rank(question: str, top_k: int) -> tuple[list[dict[str, Any]], float]:
            scores = [char_score(question, text) for text in texts]
            ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
            return [nodes[index] for index in ranked], float(scores[ranked[0]]) if ranked else 0.0

        return rank


def generate_submission(
    questions: list[dict[str, str]],
    nodes: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, str]]:
    rank = build_fast_ranker(nodes)
    rows: list[dict[str, str]] = []
    for question_row in questions:
        qid = submission_id(question_row.get("question_id", ""))
        question = normalize_question(question_row.get("question", ""))
        kind = question_kind(question)
        top_nodes, top_score = rank(question, top_k)

        # Generic platform-service questions are not fully covered by the product manuals,
        # so a policy-safe response is usually better than overfitting to a random manual.
        if kind != "manual" and top_score < 0.18:
            ret = policy_answer(kind, question)
        elif kind == "manual":
            ret = manual_answer(question, top_nodes, top_score)
        else:
            manual_ret = manual_answer(question, top_nodes, top_score)
            policy_ret = policy_answer(kind, question)
            ret = manual_ret if top_score >= 0.28 else policy_ret

        rows.append({"id": qid, "ret": clean_text(ret)})
    rows.sort(key=lambda row: int(row["id"]) if row["id"].isdigit() else row["id"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fast first-try DataFountain submission CSV.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    questions = read_csv(args.questions)
    nodes = [node for node in read_jsonl(args.nodes) if clean_text(node.get("content"))]
    if not questions:
        raise SystemExit(f"No questions found: {resolve_path(args.questions)}")
    if not nodes:
        raise SystemExit(f"No nodes found: {resolve_path(args.nodes)}")

    rows = generate_submission(questions, nodes, args.top_k)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"First row: {json.dumps(rows[0], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
