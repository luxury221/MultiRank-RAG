from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pipeline_common import (
    DEFAULT_NODES,
    LEGACY_NODES,
    clean_text,
    copy_jsonl_alias,
    ensure_project_dirs,
    normalize_doc_id,
    read_jsonl,
    resolve_path,
    write_jsonl,
)


VISUAL_NODE_TYPES = {"figure", "table", "caption"}
VISUAL_SECTION_RE = re.compile(r"\n*\s*Visual summary:\s*.*$", re.S)
QWEN_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_DEFAULT_MODEL = "qwen-vl-plus"
ARK_DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def rel_path(path: Path) -> str:
    return path.relative_to(resolve_path(".")).as_posix()


def safe_stem(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", value)
    return value.strip("_") or hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def normalize_for_match(value: Any) -> str:
    text = clean_text(value).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def strip_inferred_prefix(text: str) -> str:
    text = VISUAL_SECTION_RE.sub("", clean_text(text))
    return re.sub(r"^(Figure|Table) node inferred from caption:\s*", "", text, flags=re.I)


def block_text(block: dict[str, Any]) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
        if line_text.strip():
            parts.append(line_text)
    return clean_text("\n".join(parts))


def rect_area(rect: list[float]) -> float:
    if len(rect) != 4:
        return 0.0
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def expand_bbox(bbox: list[float], page_rect, pad: float) -> list[float]:
    return [
        max(float(page_rect.x0), bbox[0] - pad),
        max(float(page_rect.y0), bbox[1] - pad),
        min(float(page_rect.x1), bbox[2] + pad),
        min(float(page_rect.y1), bbox[3] + pad),
    ]


def union_bbox(items: list[list[float]]) -> list[float]:
    items = [item for item in items if len(item) == 4 and rect_area(item) > 0]
    if not items:
        return []
    return [
        min(item[0] for item in items),
        min(item[1] for item in items),
        max(item[2] for item in items),
        max(item[3] for item in items),
    ]


def parse_bbox(value: Any) -> list[float]:
    if not value:
        return []
    try:
        if isinstance(value, str):
            payload = json.loads(value)
        else:
            payload = value
        bbox = [float(item) for item in payload]
    except Exception:
        return []
    if len(bbox) != 4 or rect_area(bbox) <= 0:
        return []
    return bbox


def horizontal_overlap(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    overlap = max(0.0, right - left)
    return overlap / max(1.0, min(a[2] - a[0], b[2] - b[0]))


def vertical_gap(a: list[float], b: list[float]) -> float:
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0


def find_best_text_block(node: dict[str, Any], text_blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not text_blocks:
        return None
    node_text = strip_inferred_prefix(node.get("content", ""))
    needle = normalize_for_match(node_text)
    if not needle:
        return None

    best_score = 0.0
    best: dict[str, Any] | None = None
    for block in text_blocks:
        haystack = normalize_for_match(block.get("text", ""))
        if not haystack:
            continue
        score = 0.0
        short = needle[: min(len(needle), 180)]
        if short and short in haystack:
            score = 1.0
        elif haystack[: min(len(haystack), 180)] in needle:
            score = 0.92
        else:
            score = SequenceMatcher(None, short, haystack[: max(180, len(short))]).ratio()
        if score > best_score:
            best_score = score
            best = block
    return best if best_score >= 0.42 else None


def looks_like_table_text(text: str) -> bool:
    text = clean_text(text)
    numeric_tokens = len(re.findall(r"\d+(?:[.,]\d+)?", text))
    separators = text.count("\n") + text.count("|")
    return numeric_tokens >= 6 or separators >= 4


def choose_nearest_image(caption_bbox: list[float], image_blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not caption_bbox or not image_blocks:
        return None
    scored: list[tuple[float, dict[str, Any]]] = []
    for block in image_blocks:
        bbox = block["bbox"]
        overlap = horizontal_overlap(caption_bbox, bbox)
        gap = vertical_gap(caption_bbox, bbox)
        area_bonus = math.log1p(rect_area(bbox)) / 20.0
        score = overlap * 4.0 - gap / 120.0 + area_bonus
        scored.append((score, block))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else None


def choose_related_images(caption_bbox: list[float], image_blocks: list[dict[str, Any]], page_rect) -> list[dict[str, Any]]:
    if not caption_bbox:
        return []
    related: list[dict[str, Any]] = []
    max_gap = float(page_rect.height) * 0.48
    for block in image_blocks:
        bbox = block["bbox"]
        if vertical_gap(caption_bbox, bbox) > max_gap:
            continue
        if horizontal_overlap(caption_bbox, bbox) < 0.03:
            continue
        related.append(block)
    return related


def choose_table_region(
    caption_block: dict[str, Any] | None,
    text_blocks: list[dict[str, Any]],
    page_rect,
) -> tuple[list[float], str]:
    if not caption_block:
        return [], ""
    caption_bbox = caption_block["bbox"]
    nearby: list[dict[str, Any]] = []
    for block in text_blocks:
        if block is caption_block:
            continue
        bbox = block["bbox"]
        if horizontal_overlap(caption_bbox, bbox) < 0.25:
            continue
        if vertical_gap(caption_bbox, bbox) > float(page_rect.height) * 0.28:
            continue
        if looks_like_table_text(block.get("text", "")):
            nearby.append(block)
    if not nearby:
        return expand_bbox(caption_bbox, page_rect, 8), "caption_bbox"
    nearby.sort(key=lambda item: vertical_gap(caption_bbox, item["bbox"]))
    bbox = union_bbox([caption_bbox, nearby[0]["bbox"]])
    return expand_bbox(bbox, page_rect, 8), "caption_plus_table_text"


def choose_figure_region(
    caption_block: dict[str, Any] | None,
    image_blocks: list[dict[str, Any]],
    page_rect,
) -> tuple[list[float], str]:
    if caption_block:
        caption_bbox = caption_block["bbox"]
        related_images = choose_related_images(caption_bbox, image_blocks, page_rect)
        if related_images:
            bbox = union_bbox([caption_bbox] + [image["bbox"] for image in related_images])
            return expand_bbox(bbox, page_rect, 8), "caption_plus_images"
        image = choose_nearest_image(caption_bbox, image_blocks)
        if image:
            bbox = union_bbox([caption_bbox, image["bbox"]])
            return expand_bbox(bbox, page_rect, 8), "caption_plus_image"
        y0 = max(float(page_rect.y0), caption_bbox[1] - float(page_rect.height) * 0.36)
        y1 = min(float(page_rect.y1), caption_bbox[3] + float(page_rect.height) * 0.08)
        bbox = [float(page_rect.x0) + 32, y0, float(page_rect.x1) - 32, y1]
        return expand_bbox(bbox, page_rect, 4), "caption_projected_region"
    if image_blocks:
        largest = max(image_blocks, key=lambda block: rect_area(block["bbox"]))
        return expand_bbox(largest["bbox"], page_rect, 8), "image_block"
    return [], ""


class VisualCaptioner:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device_name = "cpu"
        self.processor = None
        self.model = None
        if not model_name:
            return
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device_name = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device_name == "auto":
            self.device_name = "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device_name)
        self.model.eval()

    def available(self) -> bool:
        return self.processor is not None and self.model is not None

    def caption(self, image_path: Path) -> str:
        if not self.available():
            return ""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device_name)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=42)
        return clean_text(self.processor.decode(output[0], skip_special_tokens=True))


def image_to_data_url(image_path: Path) -> str:
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{data}"


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


class QwenVisionCaptioner:
    def __init__(
        self,
        model_name: str = QWEN_DEFAULT_MODEL,
        base_url: str = QWEN_DEFAULT_BASE_URL,
        api_key_env: str = "DASHSCOPE_API_KEY",
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name or QWEN_DEFAULT_MODEL
        self.base_url = base_url or QWEN_DEFAULT_BASE_URL
        self.api_key_env = api_key_env or "DASHSCOPE_API_KEY"
        self.timeout = timeout
        self.client = None
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout)

    def available(self) -> bool:
        return self.client is not None

    def caption(self, image_path: Path, node: dict[str, Any] | None = None) -> str:
        if not self.available():
            return ""
        node = node or {}
        node_type = clean_text(node.get("node_type")) or "visual evidence"
        context = clean_text(strip_inferred_prefix(node.get("content", "")))
        prompt = f"""你正在为一个学术 PDF 多模态 RAG 系统生成视觉证据摘要。

请只根据图片内容和给定上下文分析这张裁剪图。输出严格 JSON，不要 Markdown，不要代码块。

JSON 字段：
{{
  "visual_title": "图表/表格主题",
  "visual_type": "table|figure|caption|text",
  "key_objects": ["关键视觉元素或表头"],
  "data_or_trends": ["能从图表中读出的数值、趋势、对比或结构关系"],
  "qa_evidence": "可用于回答问题的一句话证据摘要",
  "limitations": "如果图片过小、裁剪不完整或无法判断，请说明"
}}

节点类型：{node_type}
文档上下文：{context[:900]}
"""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=700,
        )
        content = completion.choices[0].message.content
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        return clean_text(content)


class ArkVisionCaptioner:
    def __init__(
        self,
        model_name: str = "",
        base_url: str = ARK_DEFAULT_BASE_URL,
        api_key_env: str = "ARK_API_KEY",
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name or os.getenv("ARK_MODEL", "")
        self.base_url = base_url or ARK_DEFAULT_BASE_URL
        self.api_key_env = api_key_env or "ARK_API_KEY"
        self.timeout = timeout
        self.client = None
        api_key = os.getenv(self.api_key_env)
        if not api_key or not self.model_name:
            return
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout)

    def available(self) -> bool:
        return self.client is not None

    def caption(self, image_path: Path, node: dict[str, Any] | None = None) -> str:
        if not self.available():
            return ""
        node = node or {}
        node_type = clean_text(node.get("node_type")) or "visual evidence"
        context = clean_text(strip_inferred_prefix(node.get("content", "")))
        prompt = f"""你正在为一个学术 PDF 多模态 RAG 系统生成视觉证据摘要。

请只根据图片内容和给定上下文分析这张裁剪图。输出严格 JSON，不要 Markdown，不要代码块。

JSON 字段：
{{
  "visual_title": "图表/表格主题",
  "visual_type": "table|figure|caption|text",
  "key_objects": ["关键视觉元素或表头"],
  "data_or_trends": ["能从图表中读出的数值、趋势、对比或结构关系"],
  "qa_evidence": "可用于回答问题的一句话证据摘要",
  "limitations": "如果图片过小、裁剪不完整或无法判断，请说明"
}}

节点类型：{node_type}
文档上下文：{context[:900]}
"""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=700,
        )
        content = completion.choices[0].message.content
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        return clean_text(content)


def build_visual_summary(node: dict[str, Any], bbox_source: str, caption: str = "") -> str:
    parts = [
        f"Grounded {node.get('node_type')} crop on page {node.get('page')} ({bbox_source}).",
    ]
    if caption:
        parts.append(f"VLM caption: {caption}.")
    context = clean_text(strip_inferred_prefix(node.get("content", "")))
    if context:
        parts.append(f"Document caption/context: {context[:360]}")
    return clean_text(" ".join(parts))


def remove_old_visual_summary(content: Any) -> str:
    return clean_text(VISUAL_SECTION_RE.sub("", clean_text(content)))


def prefixed_field(prefix: str, field: str) -> str:
    prefix = clean_text(prefix)
    return f"{prefix}_{field}" if prefix else field


def caption_field_names(prefix: str) -> list[str]:
    return [
        prefixed_field(prefix, field)
        for field in [
            "visual_caption",
            "visual_caption_model",
            "visual_caption_error",
            "visual_title",
            "visual_type",
            "key_objects",
            "data_or_trends",
            "qa_evidence",
            "limitations",
            "visual_summary",
        ]
    ]


def write_caption_fields(node: dict[str, Any], prefix: str, caption: str, model_name: str) -> None:
    node[prefixed_field(prefix, "visual_caption")] = caption
    node[prefixed_field(prefix, "visual_caption_model")] = model_name
    structured = parse_json_object(caption)
    if structured:
        for field in [
            "visual_title",
            "visual_type",
            "key_objects",
            "data_or_trends",
            "qa_evidence",
            "limitations",
        ]:
            if field in structured:
                value = structured[field]
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                node[prefixed_field(prefix, field)] = clean_text(value)


def save_page_image(page, output_dir: Path, page_no: int, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"page_{page_no:03d}.png"
    matrix = page.get_pixmap(dpi=dpi, alpha=False)
    matrix.save(str(path))
    return path


def save_crop(page, bbox: list[float], output_path: Path, dpi: int) -> bool:
    if len(bbox) != 4 or rect_area(bbox) <= 25:
        return False
    import fitz

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = fitz.Rect(*bbox) & page.rect
    if clip.is_empty:
        return False
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    pix.save(str(output_path))
    return True


def process_document(
    pdf_path: Path,
    doc_nodes: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
    captioner: VisualCaptioner,
    max_captions: int,
    caption_count: int,
    caption_node_ids: set[str],
    skip_existing_captions: bool,
    caption_field_prefix: str,
) -> int:
    import fitz

    doc = fitz.open(str(pdf_path))
    doc_id = normalize_doc_id(pdf_path.name)
    nodes_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in doc_nodes:
        try:
            page_no = int(node.get("page") or 0)
        except (TypeError, ValueError):
            continue
        if 1 <= page_no <= len(doc):
            nodes_by_page[page_no].append(node)

    page_dir = output_dir / "pages" / safe_stem(doc_id)
    crop_dir = output_dir / "crops" / safe_stem(doc_id)
    for page_no, page_nodes in nodes_by_page.items():
        page = doc[page_no - 1]
        page_image = save_page_image(page, page_dir, page_no, dpi)
        raw_blocks = page.get_text("dict").get("blocks", [])
        text_blocks = [
            {"bbox": [float(v) for v in block.get("bbox", [])], "text": block_text(block)}
            for block in raw_blocks
            if block.get("type") == 0 and block_text(block)
        ]
        image_blocks = [
            {"bbox": [float(v) for v in block.get("bbox", [])]}
            for block in raw_blocks
            if block.get("type") == 1 and rect_area([float(v) for v in block.get("bbox", [])]) > 600
        ]
        for node in page_nodes:
            node["page_image_path"] = rel_path(page_image)
            node_type = clean_text(node.get("node_type"))
            if node_type == "page":
                continue

            bbox: list[float] = []
            bbox_source = ""
            layout_bbox = parse_bbox(node.get("bbox"))
            if layout_bbox:
                bbox = expand_bbox(layout_bbox, page.rect, 6)
                bbox_source = clean_text(node.get("bbox_source")) or "layout_bbox"

            matched_block = find_best_text_block(node, text_blocks)
            if bbox:
                pass
            elif node_type == "figure":
                bbox, bbox_source = choose_figure_region(matched_block, image_blocks, page.rect)
            elif node_type == "table" and str(node.get("content", "")).startswith("Table node inferred"):
                bbox, bbox_source = choose_table_region(matched_block, text_blocks, page.rect)
            elif matched_block:
                bbox = expand_bbox(matched_block["bbox"], page.rect, 6)
                bbox_source = "text_block"
            elif node_type in VISUAL_NODE_TYPES:
                bbox, bbox_source = choose_figure_region(None, image_blocks, page.rect)

            if not bbox:
                continue
            crop_path = crop_dir / f"{safe_stem(clean_text(node.get('node_id')))}.png"
            if not save_crop(page, bbox, crop_path, dpi):
                continue

            node["bbox"] = json.dumps([round(value, 2) for value in bbox], ensure_ascii=False)
            node["bbox_source"] = bbox_source
            node["crop_image_path"] = rel_path(crop_path)

            caption_key = prefixed_field(caption_field_prefix, "visual_caption")
            caption = clean_text(node.get(caption_key))
            should_caption = (
                captioner.available()
                and node_type in VISUAL_NODE_TYPES
                and (not caption_node_ids or clean_text(node.get("node_id")) in caption_node_ids)
                and (max_captions <= 0 or caption_count < max_captions)
                and not (skip_existing_captions and caption)
            )
            if should_caption:
                for stale_key in caption_field_names(caption_field_prefix):
                    node.pop(stale_key, None)
                caption = ""
                try:
                    try:
                        caption = captioner.caption(crop_path, node)
                    except TypeError:
                        caption = captioner.caption(crop_path)
                    caption_count += 1
                    print(f"Captioned {caption_count}: {node.get('node_id')} ({captioner.model_name})", flush=True)
                except Exception as exc:
                    node[prefixed_field(caption_field_prefix, "visual_caption_error")] = clean_text(str(exc))[:240]
            if caption:
                write_caption_fields(node, caption_field_prefix, caption, captioner.model_name)

            visual_summary = build_visual_summary(node, bbox_source, caption)
            summary_key = prefixed_field(caption_field_prefix, "visual_summary")
            node[summary_key] = visual_summary
            if node_type in VISUAL_NODE_TYPES and not caption_field_prefix:
                base_content = remove_old_visual_summary(node.get("content", ""))
                node["content"] = clean_text(f"{base_content}\n\nVisual summary: {visual_summary}")
    doc.close()
    return caption_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Add page crops, bboxes and optional VLM captions to evidence nodes.")
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--nodes", default=str(DEFAULT_NODES.relative_to(DEFAULT_NODES.parents[2])))
    parser.add_argument("--output", default=str(DEFAULT_NODES.relative_to(DEFAULT_NODES.parents[2])))
    parser.add_argument("--visual-dir", default="outputs/visual")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--caption-provider", choices=["local", "qwen", "doubao"], default="local")
    parser.add_argument("--caption-model", default="", help="Optional BLIP model, e.g. Salesforce/blip-image-captioning-base.")
    parser.add_argument("--caption-device", default="auto")
    parser.add_argument("--max-captions", type=int, default=0, help="0 means no limit when --caption-model is set.")
    parser.add_argument("--caption-node-ids", default="", help="Optional comma/semicolon separated node ids to caption.")
    parser.add_argument("--caption-field-prefix", default="", help="Write caption fields with this prefix, e.g. doubao.")
    parser.add_argument("--skip-existing-captions", action="store_true")
    parser.add_argument("--qwen-model", default=QWEN_DEFAULT_MODEL)
    parser.add_argument("--qwen-base-url", default=QWEN_DEFAULT_BASE_URL)
    parser.add_argument("--qwen-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--qwen-timeout", type=float, default=60.0)
    parser.add_argument("--ark-model", default="")
    parser.add_argument("--ark-base-url", default=ARK_DEFAULT_BASE_URL)
    parser.add_argument("--ark-api-key-env", default="ARK_API_KEY")
    parser.add_argument("--ark-timeout", type=float, default=60.0)
    args = parser.parse_args()

    ensure_project_dirs()
    nodes = read_jsonl(args.nodes)
    pdfs = {normalize_doc_id(path.name): path for path in resolve_path(args.pdf_dir).glob("*.pdf")}
    nodes_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        nodes_by_doc[clean_text(node.get("doc_id"))].append(node)

    if args.caption_provider == "qwen":
        captioner = QwenVisionCaptioner(
            model_name=args.caption_model or args.qwen_model,
            base_url=args.qwen_base_url,
            api_key_env=args.qwen_api_key_env,
            timeout=args.qwen_timeout,
        )
        if not captioner.available():
            raise SystemExit(
                f"{args.qwen_api_key_env} is not set. Set it in this terminal before using --caption-provider qwen."
            )
    elif args.caption_provider == "doubao":
        captioner = ArkVisionCaptioner(
            model_name=args.caption_model or args.ark_model,
            base_url=args.ark_base_url,
            api_key_env=args.ark_api_key_env,
            timeout=args.ark_timeout,
        )
        if not captioner.available():
            raise SystemExit(
                f"{args.ark_api_key_env} or ARK_MODEL is not set. Set both before using --caption-provider doubao."
            )
    else:
        captioner = VisualCaptioner(args.caption_model, args.caption_device) if args.caption_model else VisualCaptioner("")
    caption_node_ids = {item.strip() for item in re.split(r"[;,，；]\s*", args.caption_node_ids) if item.strip()}
    caption_count = 0
    processed_docs = 0
    for doc_id, pdf_path in sorted(pdfs.items()):
        if doc_id not in nodes_by_doc:
            continue
        caption_count = process_document(
            pdf_path,
            nodes_by_doc[doc_id],
            resolve_path(args.visual_dir),
            args.dpi,
            captioner,
            args.max_captions,
            caption_count,
            caption_node_ids,
            args.skip_existing_captions,
            args.caption_field_prefix,
        )
        processed_docs += 1

    write_jsonl(args.output, nodes)
    copy_jsonl_alias(args.output, LEGACY_NODES)
    visual_nodes = [node for node in nodes if node.get("crop_image_path")]
    print(f"Updated {len(nodes)} nodes across {processed_docs} PDFs")
    print(f"Added visual crops to {len(visual_nodes)} nodes")
    if args.caption_provider in {"qwen", "doubao"} or args.caption_model:
        print(f"Generated {caption_count} VLM captions with {captioner.model_name}")
    print(f"Wrote visual-enhanced nodes to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
