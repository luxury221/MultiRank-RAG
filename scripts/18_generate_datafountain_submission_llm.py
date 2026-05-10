from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

from ark_clients import (
    ArkChatClient,
    ArkError,
    ArkMultimodalEmbedder,
    JsonlEmbeddingCache,
    cosine_similarity,
)
from pipeline_common import clean_text, preview, read_csv, read_jsonl, resolve_path


DEFAULT_QUESTIONS = "outputs/datafountain_1165_full/questions.csv"
DEFAULT_NODES = "outputs/datafountain_1165_full/nodes.jsonl"
DEFAULT_OUTPUT = "outputs/datafountain_1165_full/submission_llm.csv"
DEFAULT_EMBEDDING_CACHE = "outputs/datafountain_1165_full/ark_embedding_cache.jsonl"


def load_fast_module():
    path = Path(__file__).with_name("17_generate_datafountain_submission.py")
    spec = importlib.util.spec_from_file_location("datafountain_fast_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load fast submission module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAST = load_fast_module()


def evidence_text(node: dict[str, Any]) -> str:
    text = FAST.answer_source_text(node)
    if not text:
        text = FAST.node_text(node)
    return clean_text(text)


def compact_answer(text: str, limit: int = 520) -> str:
    text = clean_text(text)
    text = text.replace("```", "")
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ["。", "；", ";", ".", "！", "!", "？", "?"]:
        pos = cut.rfind(sep)
        if pos >= limit * 0.55:
            return cut[: pos + 1]
    return cut.rstrip("，,、；;：: ") + "。"


def rerank_with_embedding(
    question: str,
    nodes: list[dict[str, Any]],
    embedder: ArkMultimodalEmbedder,
    cache: JsonlEmbeddingCache,
    top_k: int,
    embed_images: bool,
) -> list[tuple[dict[str, Any], float]]:
    query_embedding = embedder.embed_text(question, cache)
    scored: list[tuple[dict[str, Any], float]] = []
    for node in nodes:
        text = evidence_text(node) or FAST.node_text(node)
        if not text:
            scored.append((node, 0.0))
            continue
        image_ref = clean_text(node.get("crop_image_path")) or clean_text(node.get("page_image_path"))
        image_path = resolve_path(image_ref) if image_ref else None
        if embed_images and image_path and image_path.exists():
            node_embedding = embedder.embed_image_file(image_path, text=preview(text, 500), cache=cache)
        else:
            node_embedding = embedder.embed_text(preview(text, 1200), cache)
        scored.append((node, cosine_similarity(query_embedding, node_embedding)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def format_evidence(scored_nodes: list[tuple[dict[str, Any], float]]) -> str:
    blocks: list[str] = []
    for index, (node, score) in enumerate(scored_nodes, start=1):
        doc_id = clean_text(node.get("doc_id")) or "未知文档"
        node_type = clean_text(node.get("node_type")) or "text"
        source_ref = clean_text(node.get("source_ref")) or f"page {node.get('page', '')}".strip()
        text = preview(evidence_text(node), 520)
        blocks.append(
            f"[{index}] 文档: {doc_id} | 模态/块类型: {node_type} | 位置: {source_ref} | 相关性: {score:.3f}\n{text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, kind: str, scored_nodes: list[tuple[dict[str, Any], float]]) -> tuple[str, str]:
    english = FAST.is_english(question)
    language_rule = "Answer in English." if english else "请使用中文回答。"
    policy_hint = FAST.policy_answer(kind, question) if kind != "manual" else ""
    system_prompt = (
        "你是一个面向商品手册和售后问答的证据型 RAG 助手。"
        "回答必须基于给定证据；证据不足时要说明需要补充型号、故障现象或联系售后核实。"
        "不要编造价格、承诺、期限或页面没有出现的具体政策。"
        "最终只输出提交答案正文，不要输出分析过程、证据编号或 Markdown。"
    )
    user_prompt = f"""问题：
{question}

问题类别：
{kind}

可用证据：
{format_evidence(scored_nodes)}

通用售后安全口径：
{policy_hint}

生成要求：
1. {language_rule}
2. 优先回答用户真正问的内容，必要时给出操作建议或核实路径。
3. 如果证据来自图示、步骤或参数表，要把图示/步骤/参数含义转成自然语言。
4. 答案控制在 300 字以内。"""
    return system_prompt, user_prompt


def fallback_answer(question: str, kind: str, nodes: list[dict[str, Any]], score: float) -> str:
    if kind != "manual" and score < 0.18:
        return FAST.policy_answer(kind, question)
    if kind == "manual":
        return FAST.manual_answer(question, nodes, score)
    manual_ret = FAST.manual_answer(question, nodes, score)
    policy_ret = FAST.policy_answer(kind, question)
    return manual_ret if score >= 0.28 else policy_ret


def load_existing_rows(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = clean_text(row.get("id"))
            ret = clean_text(row.get("ret"))
            if qid and ret:
                rows[qid] = ret
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: int(row["id"]) if row["id"].isdigit() else row["id"])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(sorted_rows)


def generate_submission(
    questions: list[dict[str, str]],
    nodes: list[dict[str, Any]],
    output: Path,
    fast_top_k: int,
    evidence_top_k: int,
    use_ark_embedding: bool,
    embed_images: bool,
    use_llm: bool,
    limit: int | None,
    resume: bool,
    cache_path: Path,
) -> list[dict[str, str]]:
    rank: Callable[[str, int], tuple[list[dict[str, Any]], float]] = FAST.build_fast_ranker(nodes)
    existing = load_existing_rows(output) if resume else {}
    rows: list[dict[str, str]] = []

    embedder = ArkMultimodalEmbedder() if use_ark_embedding else None
    embedding_cache = JsonlEmbeddingCache(cache_path) if use_ark_embedding else None
    chat_client = ArkChatClient() if use_llm else None

    selected_questions = questions[:limit] if limit else questions
    for index, question_row in enumerate(selected_questions, start=1):
        qid = FAST.submission_id(question_row.get("question_id", ""))
        question = FAST.normalize_question(question_row.get("question", ""))
        if qid in existing:
            rows.append({"id": qid, "ret": existing[qid]})
            continue

        kind = FAST.question_kind(question)
        fast_nodes, fast_score = rank(question, fast_top_k)
        if embedder and embedding_cache:
            scored_nodes = rerank_with_embedding(
                question,
                fast_nodes,
                embedder,
                embedding_cache,
                evidence_top_k,
                embed_images,
            )
        else:
            scored_nodes = [(node, fast_score if node is fast_nodes[0] else 0.0) for node in fast_nodes[:evidence_top_k]]

        try:
            if chat_client:
                system_prompt, user_prompt = build_prompt(question, kind, scored_nodes)
                ret = chat_client.complete(system_prompt, user_prompt)
            else:
                ret = fallback_answer(question, kind, [node for node, _ in scored_nodes], fast_score)
        except ArkError as exc:
            print(f"[{index}/{len(selected_questions)}] Ark failed for id={qid}; using fallback. {exc}")
            ret = fallback_answer(question, kind, [node for node, _ in scored_nodes], fast_score)

        ret = compact_answer(ret)
        if not ret:
            ret = fallback_answer(question, kind, [node for node, _ in scored_nodes], fast_score)
        rows.append({"id": qid, "ret": clean_text(ret)})
        write_rows(output, rows)
        print(f"[{index}/{len(selected_questions)}] id={qid} done: {preview(ret, 90)}")

    write_rows(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a DataFountain submission with fast retrieval, optional Ark embedding rerank, and Ark LLM answers."
    )
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-cache", default=DEFAULT_EMBEDDING_CACHE)
    parser.add_argument("--fast-top-k", type=int, default=12)
    parser.add_argument("--evidence-top-k", type=int, default=5)
    parser.add_argument("--use-ark-embedding", action="store_true")
    parser.add_argument("--embed-images", action="store_true", help="Embed local figure images as data URLs when available.")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    questions = read_csv(args.questions)
    nodes = [node for node in read_jsonl(args.nodes) if clean_text(node.get("content"))]
    if not questions:
        raise SystemExit(f"No questions found: {resolve_path(args.questions)}")
    if not nodes:
        raise SystemExit(f"No nodes found: {resolve_path(args.nodes)}")

    output = resolve_path(args.output)
    rows = generate_submission(
        questions=questions,
        nodes=nodes,
        output=output,
        fast_top_k=args.fast_top_k,
        evidence_top_k=args.evidence_top_k,
        use_ark_embedding=args.use_ark_embedding,
        embed_images=args.embed_images,
        use_llm=not args.no_llm,
        limit=args.limit,
        resume=args.resume,
        cache_path=resolve_path(args.embedding_cache),
    )
    print(f"Wrote {len(rows)} rows to {output}")
    if rows:
        print(f"First row: {json.dumps(rows[0], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
