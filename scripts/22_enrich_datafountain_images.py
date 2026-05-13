from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from pipeline_common import clean_text, preview, read_jsonl, resolve_path, write_jsonl


DEFAULT_NODES = "outputs/after_sales_kb/nodes.jsonl"
DEFAULT_CACHE = "outputs/after_sales_kb/qwen_image_caption_cache.jsonl"
QWEN_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_DEFAULT_MODEL = "qwen-vl-plus"
VISUAL_NODE_TYPES = {"figure", "table", "caption"}
PROMPT_VERSION = "datafountain-image-v2"


def env_value(name: str, default: str = "") -> str:
    try:
        from ark_clients import get_env

        return get_env(name, default)
    except Exception:
        return os.getenv(name, default)


def image_to_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def parse_json_object(text: str) -> dict[str, Any]:
    text = clean_text(text)
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def compact_value(value: Any, limit: int = 900) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        text = "；".join(clean_text(item) for item in value if clean_text(item))
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = clean_text(value)
    return preview(text, limit)


def node_image_path(node: dict[str, Any]) -> Path | None:
    for field in ("crop_image_path", "image_path", "page_image_path"):
        value = clean_text(node.get(field))
        if not value or re.match(r"^(https?|data):", value, flags=re.I):
            continue
        path = resolve_path(value)
        if path.exists() and path.is_file():
            return path
    return None


def node_context(node: dict[str, Any]) -> str:
    return clean_text(
        "\n".join(
            [
                f"doc_id: {node.get('doc_id', '')}",
                f"section: {node.get('section', '')}",
                f"source_ref: {node.get('source_ref', '')}",
                f"image_id: {node.get('image_id', '')}",
                f"previous: {node.get('previous_chunk_preview', '')}",
                f"content: {node.get('content', '')}",
                f"next: {node.get('next_chunk_preview', '')}",
            ]
        )
    )


def cache_key(model: str, image_path: Path, context: str) -> str:
    stat = image_path.stat()
    raw = json.dumps(
        {
            "prompt": PROMPT_VERSION,
            "model": model,
            "image_path": str(image_path.resolve()),
            "image_size": stat.st_size,
            "image_mtime_ns": stat.st_mtime_ns,
            "context_hash": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class JsonlCaptionCache:
    def __init__(self, path: str | Path) -> None:
        self.path = resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = clean_text(row.get("key"))
                    if key:
                        self.items[key] = row

    def get(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            return self.items.get(key)

    def put(self, key: str, row: dict[str, Any]) -> None:
        with self.lock:
            if key in self.items:
                return
            row = {"key": key, **row}
            self.items[key] = row
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


class QwenDataFountainCaptioner:
    def __init__(self, model: str, base_url: str, api_key_env: str, timeout: float) -> None:
        api_key = env_value(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set.")
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def caption(self, image_path: Path, node: dict[str, Any]) -> str:
        context = node_context(node)
        prompt = f"""你是 DataFountain 多模态客服智能体的图片解析器。
请只根据图片和给定上下文提取可用于客服问答的视觉证据。输出严格 JSON，不要 Markdown，不要代码块。

JSON 字段：
{{
  "visual_title": "图片主题或部件名称",
  "visual_type": "figure|table|diagram|photo|text",
  "key_objects": ["图片中能看清的关键部件、按钮、接口、工具或商品元素"],
  "visible_text": ["图片中可读的文字、数字、标签或警示语"],
  "operation_steps": ["图片表达的安装、拆卸、清洁、连接、调节或使用步骤"],
  "qa_evidence": "一句可直接支持客服回答的证据，优先写清对象、动作和注意事项",
  "limitations": "如果图片模糊、遮挡、裁剪不完整或无法判断，请说明；否则写空字符串",
  "confidence": "high|medium|low"
}}

上下文：
{context[:1200]}
"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.05,
            max_tokens=900,
        )
        content = completion.choices[0].message.content
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        return clean_text(content)


def build_caption_text(parsed: dict[str, Any], raw_text: str) -> str:
    if not parsed:
        return preview(raw_text, 900)
    parts = [
        f"主题：{compact_value(parsed.get('visual_title'), 160)}",
        f"类型：{compact_value(parsed.get('visual_type'), 80)}",
        f"关键对象：{compact_value(parsed.get('key_objects'), 260)}",
        f"可见文字：{compact_value(parsed.get('visible_text'), 260)}",
        f"操作步骤：{compact_value(parsed.get('operation_steps'), 360)}",
        f"问答证据：{compact_value(parsed.get('qa_evidence'), 360)}",
    ]
    limitations = compact_value(parsed.get("limitations"), 180)
    if limitations:
        parts.append(f"限制：{limitations}")
    return clean_text("。".join(part for part in parts if part and not part.endswith("：")) + "。")


def apply_caption(node: dict[str, Any], raw_text: str, model: str) -> None:
    parsed = parse_json_object(raw_text)
    node["visual_caption_provider"] = "qwen"
    node["visual_caption_model"] = model
    node["visual_caption_raw"] = raw_text
    node["visual_caption"] = build_caption_text(parsed, raw_text)

    if parsed:
        field_map = {
            "visual_title": "visual_title",
            "visual_type": "visual_type",
            "key_objects": "key_objects",
            "visible_text": "ocr_text",
            "operation_steps": "data_or_trends",
            "qa_evidence": "qa_evidence",
            "limitations": "limitations",
        }
        for src, dst in field_map.items():
            value = compact_value(parsed.get(src))
            if value:
                node[dst] = value
        confidence = compact_value(parsed.get("confidence"), 40)
        if confidence:
            node["visual_caption_confidence"] = confidence

    context = compact_value(node_context(node), 360)
    node["visual_summary"] = clean_text(
        " ".join(
            [
                "DataFountain visual evidence.",
                node["visual_caption"],
                f"Document context: {context}" if context else "",
            ]
        )
    )
    node.pop("visual_caption_error", None)


def should_process(node: dict[str, Any], wanted_ids: set[str], wanted_docs: set[str], force: bool) -> bool:
    node_id = clean_text(node.get("node_id"))
    if wanted_ids and node_id not in wanted_ids:
        return False
    if wanted_docs and clean_text(node.get("doc_id")) not in wanted_docs:
        return False
    if clean_text(node.get("node_type")) not in VISUAL_NODE_TYPES:
        return False
    if not node_image_path(node):
        return False
    if not force and (clean_text(node.get("qa_evidence")) or clean_text(node.get("visual_caption"))):
        return False
    return True


def split_arg(value: str) -> set[str]:
    return {item.strip() for item in re.split(r"[;,，；]\s*", value or "") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption DataFountain visual nodes with Qwen-VL and write them back.")
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--output", default="", help="Defaults to overwriting --nodes.")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--model", default=env_value("RAG_QWEN_VL_MODEL", QWEN_DEFAULT_MODEL))
    parser.add_argument("--base-url", default=env_value("RAG_QWEN_BASE_URL", QWEN_DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default=env_value("RAG_QWEN_API_KEY_ENV", "DASHSCOPE_API_KEY"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--ids", default="", help="Comma/semicolon separated node ids.")
    parser.add_argument("--doc-ids", default="", help="Comma/semicolon separated doc ids.")
    parser.add_argument("--force", action="store_true", help="Refresh nodes that already have captions.")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Qwen-VL caption workers.")
    parser.add_argument("--cache-only", action="store_true", help="Apply cached captions without calling Qwen-VL.")
    args = parser.parse_args()

    nodes_path = resolve_path(args.nodes)
    output_path = resolve_path(args.output or args.nodes)
    nodes = read_jsonl(nodes_path)
    wanted_ids = split_arg(args.ids)
    wanted_docs = split_arg(args.doc_ids)
    targets = [node for node in nodes if should_process(node, wanted_ids, wanted_docs, args.force)]
    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    print(f"Loaded {len(nodes)} nodes. Caption targets: {len(targets)}.")
    if not targets:
        write_jsonl(output_path, nodes)
        print(f"No caption work needed. Wrote {output_path}.")
        return

    cache = JsonlCaptionCache(args.cache)
    captioner = None
    if not args.cache_only:
        captioner = QwenDataFountainCaptioner(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
        )
    processed = 0
    failed = 0

    def process_node(node: dict[str, Any]) -> tuple[str, bool, str]:
        node_id = clean_text(node.get("node_id"))
        image_path = node_image_path(node)
        if image_path is None:
            return node_id, False, "missing image"
        context = node_context(node)
        key = cache_key(args.model, image_path, context)
        cached = cache.get(key)
        try:
            if cached and clean_text(cached.get("raw_text")):
                raw_text = clean_text(cached.get("raw_text"))
            else:
                if args.cache_only:
                    return node_id, False, "not cached"
                if captioner is None:
                    return node_id, False, "captioner unavailable"
                raw_text = captioner.caption(image_path, node)
                cache.put(
                    key,
                    {
                        "node_id": node_id,
                        "image_path": str(image_path),
                        "model": args.model,
                        "raw_text": raw_text,
                    },
                )
                time.sleep(max(0.0, args.sleep))
            apply_caption(node, raw_text, args.model)
            return node_id, True, ""
        except Exception as exc:
            node["visual_caption_error"] = preview(str(exc), 300)
            return node_id, False, str(exc)

    workers = max(1, args.workers)
    if workers == 1:
        for node in targets:
            node_id, ok, error = process_node(node)
            if ok:
                processed += 1
                if processed == 1 or processed % 20 == 0:
                    print(f"Captioned {processed}/{len(targets)}: {node_id}", flush=True)
            else:
                failed += 1
                print(f"Caption failed for {node_id}: {error}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_node, node) for node in targets]
            for future in as_completed(futures):
                node_id, ok, error = future.result()
                if ok:
                    processed += 1
                    if processed == 1 or processed % 20 == 0:
                        print(f"Captioned {processed}/{len(targets)}: {node_id}", flush=True)
                else:
                    failed += 1
                    print(f"Caption failed for {node_id}: {error}", flush=True)

    write_jsonl(output_path, nodes)
    print(f"Wrote {len(nodes)} nodes to {output_path}. Captioned={processed}, failed={failed}.")


if __name__ == "__main__":
    main()
