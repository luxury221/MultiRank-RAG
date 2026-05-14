from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

from pipeline_common import clean_text, read_csv, read_jsonl, resolve_path


VISUAL_TERMS = (
    "how",
    "steps",
    "install",
    "remove",
    "replace",
    "clean",
    "connect",
    "adjust",
    "setting",
    "set up",
    "setup",
    "use",
    "operate",
    "button",
    "indicator",
    "light",
    "led",
    "part",
    "component",
    "anatomy",
    "diagram",
    "size",
    "condition",
    "reset",
    "charge",
    "troubleshooting",
    "\u5982\u4f55",
    "\u600e\u4e48",
    "\u6b65\u9aa4",
    "\u5b89\u88c5",
    "\u62c6\u5378",
    "\u62c6\u9664",
    "\u66f4\u6362",
    "\u6e05\u6d01",
    "\u6e05\u6d17",
    "\u8fde\u63a5",
    "\u8bbe\u7f6e",
    "\u8c03\u8282",
    "\u8c03\u6574",
    "\u4f7f\u7528",
    "\u6309\u94ae",
    "\u6307\u793a\u706f",
    "\u90e8\u4ef6",
    "\u7ec4\u6210",
    "\u4f4d\u7f6e",
    "\u56fe",
    "\u56fe\u7247",
    "\u5c3a\u5bf8",
    "\u6761\u4ef6",
    "\u91cd\u7f6e",
    "\u5145\u7535",
    "\u6545\u969c",
)

SERVICE_ONLY_TERMS = (
    "return",
    "refund",
    "exchange",
    "invoice",
    "complaint",
    "shipping",
    "delivery",
    "freight",
    "warranty policy",
    "\u9000\u8d27",
    "\u6362\u8d27",
    "\u9000\u6b3e",
    "\u53d1\u7968",
    "\u5f00\u7968",
    "\u7269\u6d41",
    "\u5feb\u9012",
    "\u6295\u8bc9",
    "\u8fd0\u8d39",
    "\u5c11\u53d1",
    "\u9519\u53d1",
    "\u5047\u8d27",
    "\u4e8c\u624b",
)


def read_submission(path: str | Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_submission(path: str | Path, rows: list[dict[str, str]]) -> None:
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: int(row["id"]) if row["id"].isdigit() else row["id"])
    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(rows)


def parse_pic_suffix(text: str) -> tuple[str, list[str]]:
    text = clean_text(text)
    match = re.search(r"\s*<PIC>\s*;\s*(\[[^\]]+\])\s*$", text)
    if not match:
        return text, []
    ids: list[str] = []
    try:
        parsed = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        parsed = []
    if isinstance(parsed, list):
        for item in parsed:
            image_id = clean_text(str(item))
            if image_id and image_id not in ids:
                ids.append(image_id)
    return text[: match.start()].strip(), ids


def strip_all_pic_tokens(text: str) -> str:
    body, _ = parse_pic_suffix(text)
    return clean_text(re.sub(r"\s*<PIC>\s*", " ", body))


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


def load_nodes(path: str | Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    nodes = read_jsonl(path)
    nodes_by_id: dict[str, dict[str, Any]] = {}
    image_ids: set[str] = set()
    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if node_id:
            nodes_by_id[node_id] = node
        image_id = image_id_from_node(node)
        if image_id:
            image_ids.add(image_id)
    return nodes_by_id, image_ids


def load_routes(path: str | Path) -> dict[str, dict[str, str]]:
    routes: dict[str, dict[str, str]] = {}
    file_path = resolve_path(path)
    if not file_path.exists():
        return routes
    for row in read_csv(file_path):
        submission_id = clean_text(row.get("submission_id")) or re.sub(r"\D+", "", clean_text(row.get("question_id")))
        if submission_id:
            routes[submission_id] = row
    return routes


def load_ranking_images(path: str | Path, nodes_by_id: dict[str, dict[str, Any]], top_k: int = 25) -> dict[str, list[str]]:
    file_path = resolve_path(path)
    if not file_path.exists():
        return {}
    by_qid: dict[str, list[str]] = {}
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = clean_text(row.get("question_id"))
            match = re.search(r"(\d+)$", qid)
            sid = match.group(1) if match else qid
            if not sid:
                continue
            try:
                rank = int(row.get("rank") or 9999)
            except ValueError:
                rank = 9999
            if rank > top_k:
                continue
            node = nodes_by_id.get(clean_text(row.get("node_id")), {})
            if clean_text(node.get("node_type")) != "figure":
                continue
            image_id = image_id_from_node(node)
            if not image_id:
                continue
            bucket = by_qid.setdefault(sid, [])
            if image_id not in bucket:
                bucket.append(image_id)
    return by_qid


def wants_images(question: str, route: str, product: str, existing_ids: list[str]) -> bool:
    blob = clean_text(question).casefold()
    visual_hit = any(term.casefold() in blob for term in VISUAL_TERMS)
    service_only_hit = any(term.casefold() in blob for term in SERVICE_ONLY_TERMS)
    if route == "service" and not product:
        return False
    if service_only_hit and not visual_hit:
        return False
    return visual_hit or bool(existing_ids and route == "service_with_product")


def sentence_chunks(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？；;])\s+|(?<=。)|(?<=！)|(?<=？)|(?<=；)", text)
    raw_chunks = [clean_text(piece) for piece in pieces if clean_text(piece)]
    chunks: list[str] = []
    index = 0
    while index < len(raw_chunks):
        current = raw_chunks[index]
        if re.fullmatch(r"\d+\.", current) and index + 1 < len(raw_chunks):
            chunks.append(f"{current} {raw_chunks[index + 1]}")
            index += 2
            continue
        chunks.append(current)
        index += 1
    return chunks or [clean_text(text)]


def add_inline_pic_markers(body: str, image_count: int) -> str:
    body = strip_all_pic_tokens(body)
    if image_count <= 0:
        return body
    chunks = sentence_chunks(body)
    marked: list[str] = []
    inline_remaining = max(0, image_count - 1)
    for index, chunk in enumerate(chunks):
        if inline_remaining > 0 and index < len(chunks) - 1:
            marked.append(f"{chunk} <PIC>")
            inline_remaining -= 1
        else:
            marked.append(chunk)
    marked[-1] = f"{marked[-1]} <PIC>"
    return clean_text(" ".join(marked))


def trim_answer(body: str, limit: int) -> str:
    body = clean_text(body)
    if len(body) <= limit:
        return body
    cut = body[:limit]
    for sep in ("。", "！", "？", ";", "；", ".", "!", "?"):
        pos = cut.rfind(sep)
        if pos >= int(limit * 0.55):
            trimmed = cut[: pos + 1].strip()
            return re.sub(r"\s+\d+\.$", "", trimmed).rstrip()
    return re.sub(r"\s+\d+\.$", "", cut.rstrip("，,、；;：: ")).rstrip()


def postprocess_row(
    row: dict[str, str],
    route: dict[str, str],
    valid_image_ids: set[str],
    fallback_images: list[str],
    answer_limit: int,
) -> tuple[dict[str, str], list[str]]:
    sid = clean_text(row.get("id"))
    question = clean_text(route.get("question"))
    route_name = clean_text(route.get("route"))
    product = clean_text(route.get("product"))
    body, ids = parse_pic_suffix(row.get("ret", ""))
    reasons: list[str] = []

    cleaned_ids: list[str] = []
    for image_id in ids:
        if image_id in valid_image_ids and image_id not in cleaned_ids:
            cleaned_ids.append(image_id)
        else:
            reasons.append(f"drop_invalid_pic:{image_id}")
    if ids and not cleaned_ids:
        for image_id in fallback_images:
            if image_id in valid_image_ids and image_id not in cleaned_ids:
                cleaned_ids.append(image_id)
            if len(cleaned_ids) >= 3:
                break
        if cleaned_ids:
            reasons.append("replace_invalid_pic")

    keep_images = wants_images(question, route_name, product, cleaned_ids)
    if not keep_images:
        ret = trim_answer(strip_all_pic_tokens(row.get("ret", "")), answer_limit)
        if ids:
            reasons.append("remove_pic_for_nonvisual")
        return {"id": sid, "ret": ret}, reasons

    if not cleaned_ids:
        for image_id in fallback_images:
            if image_id in valid_image_ids and image_id not in cleaned_ids:
                cleaned_ids.append(image_id)
            if len(cleaned_ids) >= 3:
                break
        if cleaned_ids:
            reasons.append("add_ranked_pic")

    body = trim_answer(strip_all_pic_tokens(body), max(120, answer_limit - 80))
    if cleaned_ids:
        body = add_inline_pic_markers(body, min(3, len(cleaned_ids)))
        ret = f"{body} ;{json.dumps(cleaned_ids[:3], ensure_ascii=False)}"
    else:
        ret = body
    return {"id": sid, "ret": clean_text(ret)}, reasons


def conservative_postprocess_row(
    row: dict[str, str],
    valid_image_ids: set[str],
    fallback_images: list[str],
    answer_limit: int,
) -> tuple[dict[str, str], list[str]]:
    sid = clean_text(row.get("id"))
    original_ret = row.get("ret", "")
    body, ids = parse_pic_suffix(original_ret)
    reasons: list[str] = []

    if not ids:
        return {"id": sid, "ret": original_ret}, reasons

    cleaned_ids: list[str] = []
    for image_id in ids:
        if image_id in valid_image_ids and image_id not in cleaned_ids:
            cleaned_ids.append(image_id)
        else:
            reasons.append(f"drop_invalid_pic:{image_id}")

    if reasons and not cleaned_ids:
        for image_id in fallback_images:
            if image_id in valid_image_ids and image_id not in cleaned_ids:
                cleaned_ids.append(image_id)
            if len(cleaned_ids) >= 3:
                break
        if cleaned_ids:
            reasons.append("replace_invalid_pic")

    if not reasons:
        return {"id": sid, "ret": original_ret}, reasons

    body = trim_answer(strip_all_pic_tokens(body), max(120, answer_limit - 80))
    if cleaned_ids:
        ret = f"{body} <PIC> ;{json.dumps(cleaned_ids[:3], ensure_ascii=False)}"
    else:
        ret = body
        reasons.append("remove_unusable_pic")
    return {"id": sid, "ret": clean_text(ret)}, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess DataFountain submission to match the official PIC style.")
    parser.add_argument("--input", default="outputs/datafountain_submit_final.csv")
    parser.add_argument("--output", default="outputs/datafountain_submit_final_postprocessed.csv")
    parser.add_argument("--routes", default="outputs/after_sales_kb/question_routes.csv")
    parser.add_argument("--nodes", default="outputs/after_sales_kb/nodes.qwen_full.jsonl")
    parser.add_argument("--rankings", default="outputs/after_sales_kb/reranked_best.csv")
    parser.add_argument("--report", default="outputs/after_sales_kb/submission_postprocess_report.json")
    parser.add_argument("--answer-limit", type=int, default=680)
    parser.add_argument("--mode", choices=("aggressive", "conservative"), default="aggressive")
    args = parser.parse_args()

    nodes_by_id, valid_image_ids = load_nodes(args.nodes)
    routes = load_routes(args.routes)
    ranked_images = load_ranking_images(args.rankings, nodes_by_id)
    rows: list[dict[str, str]] = []
    reason_counts: dict[str, int] = {}
    changed: list[dict[str, Any]] = []

    for row in read_submission(args.input):
        sid = clean_text(row.get("id"))
        if args.mode == "conservative":
            processed, reasons = conservative_postprocess_row(
                row,
                valid_image_ids,
                ranked_images.get(sid, []),
                args.answer_limit,
            )
        else:
            processed, reasons = postprocess_row(
                row,
                routes.get(sid, {}),
                valid_image_ids,
                ranked_images.get(sid, []),
                args.answer_limit,
            )
        rows.append(processed)
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if processed.get("ret") != clean_text(row.get("ret")):
            changed.append({"id": sid, "reasons": reasons, "before": row.get("ret", "")[:180], "after": processed["ret"][:180]})

    write_submission(args.output, rows)
    report = {
        "rows": len(rows),
        "changed": len(changed),
        "reason_counts": reason_counts,
        "pic_answers": sum("<PIC>" in row["ret"] for row in rows),
        "invalid_pic_suffixes": sum("<PIC>" in row["ret"] and not re.search(r"<PIC>\s*;\s*\[[^\]]+\]\s*$", row["ret"]) for row in rows),
        "changed_examples": changed[:20],
    }
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
