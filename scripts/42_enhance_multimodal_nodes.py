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


PROMPT_VERSION = "multimodal-node-cd-v1"
VISUAL_NODE_TYPES = {"figure", "table", "caption"}
VISION_PROVIDERS = {"none", "local", "doubao", "qwen", "xinference", "openai_compatible"}
ARK_DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
QWEN_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
XINFERENCE_DEFAULT_BASE_URL = "http://127.0.0.1:9997/v1"
OPENAI_COMPATIBLE_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


def env_value(name: str, default: str = "") -> str:
    try:
        from ark_clients import get_env

        return get_env(name, default)
    except Exception:
        return os.getenv(name, default)


def split_cell_line(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [clean_text(cell) for cell in line.split("|")]


def is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(bool(re.fullmatch(r":?-{2,}:?", cell.replace(" ", ""))) for cell in cells if cell)


def markdown_table_blocks(text: str) -> list[list[str]]:
    lines = text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.count("|") >= 2:
            current.append(line)
        else:
            if len(current) >= 2:
                blocks.append(current)
            current = []
    if len(current) >= 2:
        blocks.append(current)
    return blocks


def parse_markdown_table(text: str) -> dict[str, Any]:
    blocks = markdown_table_blocks(text)
    if not blocks:
        return {}
    block = max(blocks, key=len)
    rows = [split_cell_line(line) for line in block]
    rows = [row for row in rows if any(row)]
    rows = [row for row in rows if not is_separator_row(row)]
    if len(rows) < 2:
        return {}
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    headers = rows[0]
    data_rows = rows[1:]
    return {"headers": headers, "rows": data_rows}


def numeric_value(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def table_caption_from_content(content: str) -> str:
    lines = [clean_text(line) for line in content.splitlines()]
    for line in lines:
        if re.match(r"^(table|表)\s*\d*[:：.]?", line, flags=re.I):
            return preview(line, 220)
    for line in lines[:6]:
        if line and "|" not in line and "Table evidence" not in line:
            return preview(line, 220)
    return ""


def compact_json(value: Any, limit: int = 1400) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return preview(text, limit)


def build_table_enrichment(node: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_markdown_table(clean_text(node.get("content")))
    if not parsed:
        return {}
    headers: list[str] = parsed["headers"]
    rows: list[list[str]] = parsed["rows"]
    row_count = len(rows)
    col_count = len(headers)
    caption = table_caption_from_content(clean_text(node.get("content")))

    facts: list[str] = []
    for row in rows[:12]:
        row_label = clean_text(row[0]) or "row"
        values = []
        for header, cell in zip(headers[1:] or headers, row[1:] or row):
            if clean_text(cell):
                values.append(f"{clean_text(header) or 'value'}={clean_text(cell)}")
        if values:
            facts.append(f"{row_label}: " + "; ".join(values[:8]))

    numeric_facts: list[str] = []
    for col_index, header in enumerate(headers):
        values: list[tuple[float, str, str]] = []
        for row in rows:
            if col_index >= len(row):
                continue
            value = numeric_value(row[col_index])
            if value is None:
                continue
            row_label = clean_text(row[0]) if row else ""
            values.append((value, row_label, clean_text(row[col_index])))
        if len(values) < 2:
            continue
        low = min(values, key=lambda item: item[0])
        high = max(values, key=lambda item: item[0])
        header_text = clean_text(header) or f"column_{col_index + 1}"
        numeric_facts.append(f"Lowest {header_text}: {low[2]} at {low[1] or 'row'}.")
        numeric_facts.append(f"Highest {header_text}: {high[2]} at {high[1] or 'row'}.")

    header_text = "; ".join(header for header in headers if header)
    summary_parts = [
        f"Structured table with {row_count} rows and {col_count} columns.",
        f"Caption: {caption}." if caption else "",
        f"Headers: {header_text}." if header_text else "",
        f"Key facts: {' | '.join(facts[:8])}." if facts else "",
        f"Numeric facts: {' | '.join(numeric_facts[:8])}." if numeric_facts else "",
    ]
    summary = clean_text(" ".join(part for part in summary_parts if part))
    return {
        "table_caption": caption,
        "table_headers": header_text,
        "table_shape": f"{row_count}x{col_count}",
        "table_key_facts": " | ".join(facts[:16]),
        "table_numeric_facts": " | ".join(numeric_facts[:16]),
        "table_summary": summary,
        "table_structured_json": compact_json({"headers": headers, "rows": rows[:40]}),
    }


def apply_table_enrichment(node: dict[str, Any]) -> bool:
    enrichment = build_table_enrichment(node)
    if not enrichment:
        return False
    node.update(enrichment)
    summary = clean_text(enrichment.get("table_summary"))
    if summary:
        node["qa_evidence"] = summary
        node["visual_caption"] = summary
        node["visual_type"] = "table"
        node["visual_summary"] = summary
        searchable = clean_text(node.get("searchable_text"))
        node["searchable_text"] = clean_text(f"{searchable}\n{summary}") if searchable else summary
        content = clean_text(node.get("content"))
        marker = "Table structured evidence:"
        if marker not in content:
            node["content"] = clean_text(f"{content}\n\n{marker}\n{summary}")
    return True


def node_image_path(node: dict[str, Any]) -> Path | None:
    for field in ("crop_image_path", "image_path", "page_image_path"):
        value = clean_text(node.get(field))
        if not value or re.match(r"^(https?|data):", value, flags=re.I):
            continue
        path = resolve_path(value)
        if path.exists() and path.is_file():
            return path
    return None


def image_to_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def node_context(node: dict[str, Any]) -> str:
    return clean_text(
        "\n".join(
            [
                f"node_type: {node.get('node_type', '')}",
                f"doc_id: {node.get('doc_id', '')}",
                f"section: {node.get('section', '')}",
                f"source_ref: {node.get('source_ref', '')}",
                f"content: {preview(node.get('content', ''), 1400)}",
                f"existing_visual_caption: {preview(node.get('visual_caption', ''), 700)}",
            ]
        )
    )


def cache_key(model: str, image_path: Path, context: str) -> str:
    stat = image_path.stat()
    raw = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
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


class JsonlCache:
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


class VisionCaptioner:
    def __init__(self, provider: str, model: str, base_url: str, api_key_env: str, timeout: float) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.client = None
        if provider in {"none", "local"}:
            return
        api_key = env_value(api_key_env)
        if provider in {"xinference", "openai_compatible"} and not api_key:
            api_key = "not-used"
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set.")
        if not model:
            raise RuntimeError(f"No vision model configured for provider={provider}.")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout)

    def available(self) -> bool:
        return self.client is not None

    def caption(self, image_path: Path, node: dict[str, Any]) -> str:
        if not self.available():
            return ""
        context = node_context(node)
        prompt = f"""You are generating visual evidence for a multimodal RAG system over complex PDF documents.

Analyze the image directly, using the context only to disambiguate labels. Return strict JSON only.

JSON schema:
{{
  "visual_title": "short topic of the figure/table/image",
  "visual_type": "chart|diagram|table|equation|photo|page|other",
  "key_objects": ["main visible objects, symbols, axes, variables, or components"],
  "visible_text": ["readable labels, legends, numbers, axis titles, captions"],
  "data_or_trends": ["visual facts, comparisons, trends, structures, or relationships"],
  "qa_evidence": "one concise evidence sentence useful for answering questions",
  "limitations": "say if the crop is incomplete, blurry, or not visually informative; otherwise empty"
}}

Context:
{context[:1600]}
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
        text = "; ".join(clean_text(item) for item in value if clean_text(item))
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = clean_text(value)
    return preview(text, limit)


def local_visual_payload(node: dict[str, Any], image_path: Path | None) -> dict[str, Any]:
    metadata = ""
    if image_path:
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                metadata = f"{image.width}x{image.height}"
        except Exception:
            metadata = ""
    context = clean_text(node.get("visual_caption") or node.get("qa_evidence") or node.get("content"))
    return {
        "visual_title": preview(clean_text(node.get("source_ref")), 160),
        "visual_type": clean_text(node.get("node_type")) or "figure",
        "key_objects": [],
        "visible_text": [],
        "data_or_trends": [],
        "qa_evidence": preview(context, 420),
        "limitations": "local fallback without VLM caption",
        "image_metadata": metadata,
    }


def build_visual_caption_text(parsed: dict[str, Any], raw_text: str) -> str:
    if not parsed:
        return preview(raw_text, 900)
    parts = [
        f"title: {compact_value(parsed.get('visual_title'), 160)}",
        f"type: {compact_value(parsed.get('visual_type'), 80)}",
        f"objects: {compact_value(parsed.get('key_objects'), 260)}",
        f"visible_text: {compact_value(parsed.get('visible_text'), 260)}",
        f"visual_facts: {compact_value(parsed.get('data_or_trends'), 420)}",
        f"qa_evidence: {compact_value(parsed.get('qa_evidence'), 420)}",
    ]
    limitations = compact_value(parsed.get("limitations"), 180)
    if limitations:
        parts.append(f"limitations: {limitations}")
    return clean_text(". ".join(part for part in parts if not part.endswith(": ")) + ".")


def apply_visual_payload(node: dict[str, Any], raw_text: str, model: str, provider: str) -> None:
    parsed = parse_json_object(raw_text)
    node["visual_caption_provider"] = provider
    node["visual_caption_model"] = model
    node["visual_caption_raw"] = raw_text
    node["visual_caption"] = build_visual_caption_text(parsed, raw_text)
    if parsed:
        field_map = {
            "visual_title": "visual_title",
            "visual_type": "visual_type",
            "key_objects": "key_objects",
            "visible_text": "ocr_text",
            "data_or_trends": "data_or_trends",
            "qa_evidence": "qa_evidence",
            "limitations": "limitations",
            "image_metadata": "image_metadata",
        }
        for src, dst in field_map.items():
            value = compact_value(parsed.get(src))
            if value:
                node[dst] = value
    if not clean_text(node.get("qa_evidence")):
        node["qa_evidence"] = preview(node["visual_caption"], 420)
    node["visual_summary"] = clean_text(
        " ".join(
            [
                "Vision-grounded evidence.",
                node.get("visual_caption", ""),
                f"Context: {preview(node.get('content', ''), 360)}",
            ]
        )
    )
    searchable = clean_text(node.get("searchable_text"))
    node["searchable_text"] = clean_text(f"{searchable}\n{node['visual_summary']}") if searchable else node["visual_summary"]


def provider_defaults(provider: str, args: argparse.Namespace) -> tuple[str, str, str]:
    if provider == "doubao":
        model = args.vision_model or env_value("RAG_ARK_VISION_MODEL") or env_value("ARK_MODEL")
        return model, args.ark_base_url, args.ark_api_key_env
    if provider == "qwen":
        model = args.vision_model or env_value("RAG_QWEN_VL_MODEL", "qwen-vl-plus")
        return model, args.qwen_base_url, args.qwen_api_key_env
    if provider == "xinference":
        model = args.vision_model or env_value("XINFERENCE_VISION_MODEL") or env_value("RAG_VISION_MODEL")
        return model, args.xinference_base_url, args.xinference_api_key_env
    if provider == "openai_compatible":
        model = args.vision_model or env_value("OPENAI_COMPATIBLE_VISION_MODEL") or env_value("LOCAL_VISION_MODEL")
        return model, args.openai_compatible_base_url, args.openai_compatible_api_key_env
    return "", "", ""


def split_ids(value: str, ids_file: str = "") -> set[str]:
    items = {item.strip() for item in re.split(r"[;,，；]\s*", value or "") if item.strip()}
    if ids_file:
        path = resolve_path(ids_file)
        if path.exists():
            items.update(clean_text(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if clean_text(line))
    return items


def should_caption(node: dict[str, Any], force: bool, wanted_ids: set[str]) -> bool:
    if clean_text(node.get("node_type")) not in VISUAL_NODE_TYPES:
        return False
    if wanted_ids and clean_text(node.get("node_id")) not in wanted_ids:
        return False
    if not node_image_path(node):
        return False
    if not force and clean_text(node.get("visual_caption_provider")):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance table and visual nodes for CD ablation experiments.")
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="outputs/benchmarks/open_ragbench_100/cd_vision_caption_cache.jsonl")
    parser.add_argument("--vision-provider", choices=sorted(VISION_PROVIDERS), default="local")
    parser.add_argument("--vision-model", default="")
    parser.add_argument("--max-vision-captions", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force-vision", action="store_true")
    parser.add_argument("--ids", default="")
    parser.add_argument("--ids-file", default="", help="Optional newline-separated visual node ids to caption.")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--ark-base-url", default=env_value("ARK_BASE_URL", ARK_DEFAULT_BASE_URL))
    parser.add_argument("--ark-api-key-env", default="ARK_API_KEY")
    parser.add_argument("--qwen-base-url", default=env_value("RAG_QWEN_BASE_URL", QWEN_DEFAULT_BASE_URL))
    parser.add_argument("--qwen-api-key-env", default=env_value("RAG_QWEN_API_KEY_ENV", "DASHSCOPE_API_KEY"))
    parser.add_argument("--xinference-base-url", default=env_value("XINFERENCE_BASE_URL", XINFERENCE_DEFAULT_BASE_URL))
    parser.add_argument("--xinference-api-key-env", default=env_value("XINFERENCE_API_KEY_ENV", "XINFERENCE_API_KEY"))
    parser.add_argument(
        "--openai-compatible-base-url",
        default=env_value("OPENAI_COMPATIBLE_BASE_URL", env_value("LOCAL_MODEL_BASE_URL", OPENAI_COMPATIBLE_DEFAULT_BASE_URL)),
    )
    parser.add_argument("--openai-compatible-api-key-env", default=env_value("OPENAI_COMPATIBLE_API_KEY_ENV", "OPENAI_COMPATIBLE_API_KEY"))
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    nodes = read_jsonl(args.nodes)
    table_count = 0
    for node in nodes:
        if clean_text(node.get("node_type")) == "table" and apply_table_enrichment(node):
            table_count += 1

    wanted_ids = split_ids(args.ids, args.ids_file)
    targets = [node for node in nodes if should_caption(node, args.force_vision, wanted_ids)]
    if args.max_vision_captions > 0:
        targets = targets[: args.max_vision_captions]

    provider = args.vision_provider
    cache = JsonlCache(args.cache)
    processed = 0
    failed = 0
    captioner: VisionCaptioner | None = None
    model = ""
    if provider not in {"none", "local"}:
        model, base_url, api_key_env = provider_defaults(provider, args)
        captioner = VisionCaptioner(provider, model, base_url, api_key_env, args.timeout)

    def process_node(node: dict[str, Any]) -> tuple[str, bool, str]:
        node_id = clean_text(node.get("node_id"))
        image_path = node_image_path(node)
        if image_path is None:
            return node_id, False, "missing image"
        context = node_context(node)
        cache_model = model or provider
        key = cache_key(cache_model, image_path, context)
        try:
            cached = cache.get(key)
            if cached and clean_text(cached.get("raw_text")):
                raw_text = clean_text(cached.get("raw_text"))
                used_provider = clean_text(cached.get("provider")) or provider
                used_model = clean_text(cached.get("model")) or cache_model
            elif provider in {"none", "local"}:
                raw_text = json.dumps(local_visual_payload(node, image_path), ensure_ascii=False)
                used_provider = "local"
                used_model = "local-fallback"
                cache.put(key, {"node_id": node_id, "provider": used_provider, "model": used_model, "raw_text": raw_text})
            else:
                if captioner is None:
                    return node_id, False, "captioner unavailable"
                raw_text = captioner.caption(image_path, node)
                used_provider = provider
                used_model = captioner.model
                cache.put(key, {"node_id": node_id, "provider": used_provider, "model": used_model, "raw_text": raw_text})
                time.sleep(max(0.0, args.sleep))
            apply_visual_payload(node, raw_text, used_model, used_provider)
            node.pop("visual_caption_error", None)
            return node_id, True, ""
        except Exception as exc:
            node["visual_caption_error"] = preview(str(exc), 300)
            return node_id, False, str(exc)

    workers = max(1, args.workers)
    if targets and provider != "none":
        if workers == 1:
            for node in targets:
                node_id, ok, error = process_node(node)
                if ok:
                    processed += 1
                    if processed == 1 or processed % 20 == 0:
                        print(f"Vision-enhanced {processed}/{len(targets)}: {node_id}", flush=True)
                else:
                    failed += 1
                    print(f"Vision enhancement failed for {node_id}: {error}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_node, node) for node in targets]
                for future in as_completed(futures):
                    node_id, ok, error = future.result()
                    if ok:
                        processed += 1
                        if processed == 1 or processed % 20 == 0:
                            print(f"Vision-enhanced {processed}/{len(targets)}: {node_id}", flush=True)
                    else:
                        failed += 1
                        print(f"Vision enhancement failed for {node_id}: {error}", flush=True)

    write_jsonl(args.output, nodes)
    print(
        f"Wrote {len(nodes)} nodes to {resolve_path(args.output)}. "
        f"tables_enhanced={table_count}, vision_targets={len(targets)}, "
        f"vision_enhanced={processed}, failed={failed}."
    )


if __name__ == "__main__":
    main()
