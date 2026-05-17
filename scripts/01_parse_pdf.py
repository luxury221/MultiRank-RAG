from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import winreg
except ImportError:  # pragma: no cover - only used on Windows.
    winreg = None  # type: ignore[assignment]

from pipeline_common import (
    DEFAULT_NODES,
    LEGACY_NODES,
    NODE_FIELDS,
    clean_text,
    copy_jsonl_alias,
    empty_questions_template,
    ensure_project_dirs,
    make_node_id,
    normalize_doc_id,
    preview,
    read_csv,
    read_jsonl,
    resolve_path,
    write_jsonl,
)


CAPTION_RE = re.compile(r"^\s*((图|圖|Fig\.?|Figure|表|Table)\s*[\w\-\.]+[:：]?.*)", re.I)
TABLE_HINT_RE = re.compile(r"(\d+\s+\d+\s+\d+)|(\|.+\|)|(指标|方法|准确率|召回率|F1|Accuracy|Recall)")
EQUATION_HINT_RE = re.compile(
    r"(\\begin\{equation\}|\\\[|\\\(|\bEq\.?\s*\(?\d+|\([0-9]{1,3}\)\s*$|"
    r"[=<>]\s*[-+*/()A-Za-z0-9_{}^\\]+|[∑∫√∞≤≥≈≠±])"
)
PROTECTED_BLOCK_RE = re.compile(
    r"^\s*((Algorithm|Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Proof|"
    r"Equation|Regression|Table|Figure|Fig\.?)\s*[\w.\-:：]*|"
    r"(算法|定理|引理|命题|推论|定义|假设|证明|方程|回归|图|表)\s*[\w.\-:：]*)",
    re.I,
)


@dataclass(frozen=True)
class PaperChunkTemplate:
    domain: str
    label: str
    chunk_size: int
    keywords: tuple[str, ...]
    section_aliases: tuple[str, ...]
    protected_terms: tuple[str, ...]


@dataclass(frozen=True)
class TemplateDecision:
    template: PaperChunkTemplate
    requested_template: str
    auto_template: str
    auto_confidence: float
    domain_candidates: tuple[tuple[str, float], ...]
    selection_reason: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "chunk_templates"
_LOCAL_ENV_CACHE: dict[str, str] | None = None


COMMON_SECTION_ALIASES = (
    "abstract",
    "摘要",
    "introduction",
    "引言",
    "related work",
    "background",
    "preliminaries",
    "problem formulation",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiment",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "limitations",
    "references",
    "appendix",
    "相关工作",
    "背景",
    "预备知识",
    "问题定义",
    "方法",
    "实验",
    "结果",
    "讨论",
    "结论",
    "局限",
    "参考文献",
    "附录",
)


PAPER_CHUNK_TEMPLATES: dict[str, PaperChunkTemplate] = {
    "general": PaperChunkTemplate(
        domain="general",
        label="通用论文",
        chunk_size=900,
        keywords=(),
        section_aliases=COMMON_SECTION_ALIASES,
        protected_terms=("abstract", "introduction", "method", "experiment", "result", "conclusion"),
    ),
    "ai": PaperChunkTemplate(
        domain="ai",
        label="AI/计算机论文",
        chunk_size=820,
        keywords=(
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural",
            "transformer",
            "large language model",
            "llm",
            "rag",
            "retrieval",
            "dataset",
            "benchmark",
            "baseline",
            "ablation",
            "training",
            "inference",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "模型",
            "数据集",
            "基线",
            "消融",
            "准确率",
            "召回率",
        ),
        section_aliases=COMMON_SECTION_ALIASES
        + (
            "model",
            "architecture",
            "algorithm",
            "training",
            "evaluation",
            "ablation study",
            "implementation details",
            "模型",
            "架构",
            "算法",
            "训练",
            "评估",
            "消融实验",
            "实现细节",
        ),
        protected_terms=("algorithm", "dataset", "baseline", "ablation", "loss", "metric"),
    ),
    "math": PaperChunkTemplate(
        domain="math",
        label="数学/理论论文",
        chunk_size=720,
        keywords=(
            "theorem",
            "lemma",
            "proposition",
            "corollary",
            "proof",
            "definition",
            "assumption",
            "equation",
            "matrix",
            "convergence",
            "bound",
            "optimization",
            "定理",
            "引理",
            "命题",
            "推论",
            "证明",
            "定义",
            "假设",
            "方程",
            "矩阵",
            "收敛",
            "上界",
        ),
        section_aliases=COMMON_SECTION_ALIASES
        + (
            "notation",
            "theory",
            "proofs",
            "main theorem",
            "mathematical formulation",
            "符号",
            "理论",
            "证明",
            "主要定理",
            "数学形式化",
        ),
        protected_terms=("theorem", "lemma", "proposition", "corollary", "proof", "definition", "assumption"),
    ),
    "finance": PaperChunkTemplate(
        domain="finance",
        label="金融/经济论文",
        chunk_size=880,
        keywords=(
            "finance",
            "financial",
            "stock",
            "return",
            "volatility",
            "portfolio",
            "asset",
            "option",
            "risk",
            "sharpe",
            "var",
            "regression",
            "market",
            "firm",
            "price",
            "revenue",
            "cash flow",
            "inflation",
            "fraud",
            "insurance",
            "claim",
            "金融",
            "股票",
            "收益率",
            "波动率",
            "投资组合",
            "风险",
            "回归",
            "市场",
            "现金流",
            "通胀",
            "医保",
            "欺诈",
            "保险",
        ),
        section_aliases=COMMON_SECTION_ALIASES
        + (
            "data",
            "variables",
            "empirical strategy",
            "regression results",
            "robustness",
            "risk analysis",
            "数据",
            "变量",
            "实证策略",
            "回归结果",
            "稳健性",
            "风险分析",
        ),
        protected_terms=("regression", "variable", "portfolio", "risk", "return", "robustness"),
    ),
    "medical": PaperChunkTemplate(
        domain="medical",
        label="医学/生命科学论文",
        chunk_size=780,
        keywords=(
            "clinical",
            "patient",
            "patients",
            "cohort",
            "randomized",
            "trial",
            "treatment",
            "diagnosis",
            "outcome",
            "adverse",
            "hazard ratio",
            "survival",
            "p-value",
            "confidence interval",
            "therapy",
            "disease",
            "医学",
            "临床",
            "患者",
            "队列",
            "随机",
            "试验",
            "治疗",
            "诊断",
            "结局",
            "不良事件",
            "生存",
        ),
        section_aliases=COMMON_SECTION_ALIASES
        + (
            "materials and methods",
            "study design",
            "participants",
            "intervention",
            "outcomes",
            "statistical analysis",
            "adverse events",
            "材料与方法",
            "研究设计",
            "受试者",
            "干预",
            "结局指标",
            "统计分析",
            "不良事件",
        ),
        protected_terms=("patient", "cohort", "trial", "outcome", "hazard ratio", "adverse event", "p-value"),
    ),
}

DOMAIN_STRONG_KEYWORDS = {
    "ai": (
        "machine learning",
        "deep learning",
        "neural",
        "transformer",
        "large language model",
        "llm",
        "rag",
        "benchmark",
        "ablation",
        "机器学习",
        "深度学习",
        "神经网络",
        "大语言模型",
        "消融",
    ),
    "math": (
        "theorem",
        "lemma",
        "proposition",
        "corollary",
        "proof",
        "definition",
        "定理",
        "引理",
        "命题",
        "推论",
        "证明",
        "定义",
    ),
    "finance": (
        "stock",
        "return",
        "volatility",
        "portfolio",
        "asset",
        "risk",
        "regression",
        "market",
        "fraud",
        "insurance",
        "claim",
        "股票",
        "收益率",
        "波动率",
        "投资组合",
        "风险",
        "回归",
        "市场",
        "欺诈",
        "保险",
        "医保",
    ),
    "medical": (
        "clinical",
        "patient",
        "cohort",
        "randomized",
        "trial",
        "treatment",
        "diagnosis",
        "outcome",
        "adverse",
        "hazard ratio",
        "survival",
        "临床",
        "患者",
        "队列",
        "随机",
        "试验",
        "治疗",
        "诊断",
        "结局",
        "不良事件",
    ),
}

DOMAIN_MIN_SCORES = {
    "ai": 8,
    "math": 6,
    "finance": 8,
    "medical": 8,
}


STRUCTURE_TYPE_PATTERNS: dict[str, tuple[tuple[str, str], ...]] = {
    "ai": (
        ("algorithm", r"\bAlgorithm\s*\d+|算法\s*\d+"),
        ("dataset", r"\b(dataset|benchmark|corpus)\b|数据集|基准"),
        ("baseline", r"\b(baseline|comparison)\b|基线|对比实验"),
        ("ablation", r"\b(ablation)\b|消融"),
        ("metric", r"\b(accuracy|precision|recall|f1|auc|metric)\b|准确率|召回率|指标"),
    ),
    "math": (
        ("theorem", r"\b(Theorem|Lemma|Proposition|Corollary)\s*\d*|定理|引理|命题|推论"),
        ("definition", r"\b(Definition|Assumption)\s*\d*|定义|假设"),
        ("proof", r"\bProof\b|证明"),
        ("equation", r"\b(Eq\.?|Equation)\s*\(?\d+|方程|[∑∫√∞≤≥≈≠±]"),
    ),
    "finance": (
        ("variable", r"\b(variable|definition of variables)\b|变量"),
        ("regression", r"\b(regression|coefficient|fixed effects?)\b|回归|系数|固定效应"),
        ("robustness", r"\b(robustness|sensitivity)\b|稳健性|敏感性"),
        ("risk", r"\b(risk|volatility|var|sharpe|portfolio)\b|风险|波动率|投资组合"),
        ("fraud", r"\b(fraud|insurance claim)\b|欺诈|医保|保险"),
    ),
    "medical": (
        ("study_design", r"\b(study design|randomized|trial|cohort)\b|研究设计|随机|试验|队列"),
        ("participants", r"\b(participants?|patients?|eligibility)\b|受试者|患者|纳入|排除"),
        ("intervention", r"\b(intervention|treatment|therapy)\b|干预|治疗"),
        ("outcome", r"\b(outcomes?|endpoint|survival|hazard ratio)\b|结局|终点|生存"),
        ("adverse_event", r"\b(adverse events?|safety)\b|不良事件|安全性"),
        ("statistical_analysis", r"\b(statistical analysis|p-value|confidence interval)\b|统计分析|置信区间"),
    ),
    "general": (
        ("method", r"\b(methods?|methodology|approach)\b|方法"),
        ("experiment", r"\b(experiments?|results?)\b|实验|结果"),
    ),
}


def _tuple_from_config(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(clean_text(item) for item in value if clean_text(item))
    return ()


def _template_from_config(name: str, payload: dict[str, Any], fallback: PaperChunkTemplate) -> PaperChunkTemplate:
    def merged(field: str, fallback_values: tuple[str, ...]) -> tuple[str, ...]:
        values = _tuple_from_config(payload.get(field)) or fallback_values
        extras = _tuple_from_config(payload.get(f"extra_{field}"))
        return tuple(dict.fromkeys([*values, *extras]))

    return PaperChunkTemplate(
        domain=clean_text(payload.get("domain")) or fallback.domain or name,
        label=clean_text(payload.get("label")) or fallback.label,
        chunk_size=int(payload.get("chunk_size") or fallback.chunk_size),
        keywords=merged("keywords", fallback.keywords),
        section_aliases=merged("section_aliases", fallback.section_aliases),
        protected_terms=merged("protected_terms", fallback.protected_terms),
    )


def load_configured_templates(base_templates: dict[str, PaperChunkTemplate]) -> dict[str, PaperChunkTemplate]:
    config_path = CONFIG_ROOT / "paper_templates.json"
    if not config_path.exists():
        return base_templates
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return base_templates

    configured = dict(base_templates)
    templates_payload = payload.get("templates", payload if isinstance(payload, dict) else {})
    if not isinstance(templates_payload, dict):
        return configured
    for name, template_payload in templates_payload.items():
        if not isinstance(template_payload, dict):
            continue
        domain = clean_text(name).lower()
        fallback = configured.get(domain, PaperChunkTemplate(domain, domain, 900, (), COMMON_SECTION_ALIASES, ()))
        configured[domain] = _template_from_config(domain, template_payload, fallback)
    return configured


PAPER_CHUNK_TEMPLATES = load_configured_templates(PAPER_CHUNK_TEMPLATES)


def extract_pages_with_python(pdf_path: Path) -> list[str]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return [clean_text(page.extract_text() or "") for page in reader.pages]
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        return [clean_text(page.extract_text() or "") for page in reader.pages]
    except Exception:
        pass

    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        return [clean_text(page.get_text("text")) for page in doc]
    except Exception:
        pass

    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            return [clean_text(page.extract_text() or "") for page in pdf.pages]
    except Exception:
        pass

    return []


def extract_pages_with_pdftotext(pdf_path: Path) -> list[str]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.txt"
            subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), str(out)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            text = out.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    pages = [clean_text(page) for page in text.split("\f")]
    return [page for page in pages if page]


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    pages = extract_pages_with_python(pdf_path)
    if pages:
        return pages
    return extract_pages_with_pdftotext(pdf_path)


def bbox_to_json(bbox: list[float]) -> str:
    if len(bbox) != 4:
        return ""
    return json.dumps([round(float(value), 2) for value in bbox], ensure_ascii=False)


def json_to_bbox(value: Any) -> list[float]:
    if not value:
        return []
    try:
        payload = json.loads(value) if isinstance(value, str) else value
        bbox = [float(item) for item in payload]
    except Exception:
        return []
    if len(bbox) != 4 or bbox_area(bbox) <= 0:
        return []
    return bbox


def bbox_area(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def union_bbox(bboxes: list[list[float]]) -> list[float]:
    bboxes = [bbox for bbox in bboxes if len(bbox) == 4 and bbox_area(bbox) > 0]
    if not bboxes:
        return []
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def fitz_block_text(block: dict[str, Any]) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        line_text = "".join(str(span.get("text", "")) for span in spans)
        if line_text.strip():
            parts.append(line_text)
    return clean_text("\n".join(parts))


def fitz_block_font_stats(block: dict[str, Any]) -> tuple[float, int]:
    sizes: list[float] = []
    flags = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            try:
                sizes.append(float(span.get("size") or 0.0))
            except (TypeError, ValueError):
                pass
            try:
                flags |= int(span.get("flags") or 0)
            except (TypeError, ValueError):
                pass
    if not sizes:
        return 0.0, flags
    return round(sum(sizes) / len(sizes), 2), flags


def normalize_repeated_margin_text(text: Any) -> str:
    text = re.sub(r"\s+", " ", clean_text(text)).lower()
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
    return text[:100]


def is_margin_block(block: dict[str, Any], page_height: float) -> bool:
    bbox = block.get("bbox", [])
    if len(bbox) != 4 or page_height <= 0:
        return False
    center_y = (bbox[1] + bbox[3]) / 2.0
    return center_y <= page_height * 0.08 or center_y >= page_height * 0.92


def is_page_number_text(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return False
    return bool(re.fullmatch(r"(?:page\s*)?\d{1,4}|[ivxlcdm]{1,8}", text, re.I))


def is_side_margin_noise(block: dict[str, Any], page_width: float, page_height: float) -> bool:
    bbox = block.get("bbox", [])
    if len(bbox) != 4 or page_width <= 0 or page_height <= 0:
        return False
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    center_x = (bbox[0] + bbox[2]) / 2.0
    near_side = center_x <= page_width * 0.12 or center_x >= page_width * 0.88
    slender_vertical = width <= page_width * 0.12 and height >= page_height * 0.18
    text = clean_text(block.get("text"))
    margin_marker = bool(re.search(r"arxiv|copyright|doi|preprint|submitted|accepted", text, re.I))
    return near_side and slender_vertical and (margin_marker or height >= page_height * 0.28)


def block_width(block: dict[str, Any]) -> float:
    bbox = block.get("bbox", [])
    if len(bbox) != 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0])


def block_center_x(block: dict[str, Any]) -> float:
    bbox = block.get("bbox", [])
    if len(bbox) != 4:
        return 0.0
    return (bbox[0] + bbox[2]) / 2.0


def block_center_y(block: dict[str, Any]) -> float:
    bbox = block.get("bbox", [])
    if len(bbox) != 4:
        return 0.0
    return (bbox[1] + bbox[3]) / 2.0


def detect_column_count(text_blocks: list[dict[str, Any]], page_width: float) -> int:
    if page_width <= 0:
        return 1
    candidates = [
        block
        for block in text_blocks
        if block_width(block) <= page_width * 0.72 and len(clean_text(block.get("text"))) >= 20
    ]
    if len(candidates) < 8:
        return 1

    midpoint = page_width / 2.0
    left = [block_center_x(block) for block in candidates if block_center_x(block) < midpoint]
    right = [block_center_x(block) for block in candidates if block_center_x(block) >= midpoint]
    if len(left) < 3 or len(right) < 3:
        return 1
    left_median = sorted(left)[len(left) // 2]
    right_median = sorted(right)[len(right) // 2]
    if left_median < page_width * 0.43 and right_median > page_width * 0.57:
        return 2
    return 1


def mark_single_column_order(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(text_blocks, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, block in enumerate(ordered, start=1):
        block["layout_column"] = "single"
        block["reading_order"] = index
    return ordered


def order_band_columns(blocks: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    midpoint = page_width / 2.0
    for block in blocks:
        block["layout_column"] = "left" if block_center_x(block) < midpoint else "right"
    return sorted(blocks, key=lambda item: (0 if item["layout_column"] == "left" else 1, item["bbox"][1], item["bbox"][0]))


def mark_two_column_order(text_blocks: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    full_width: list[dict[str, Any]] = []
    column_blocks: list[dict[str, Any]] = []
    for block in text_blocks:
        bbox = block.get("bbox", [])
        crosses_midline = len(bbox) == 4 and bbox[0] < page_width * 0.42 and bbox[2] > page_width * 0.58
        if block_width(block) >= page_width * 0.72 or crosses_midline:
            block["layout_column"] = "full"
            full_width.append(block)
        else:
            column_blocks.append(block)

    ordered: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for full_block in sorted(full_width, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        band = [
            block
            for block in column_blocks
            if block.get("block_id") not in consumed and block_center_y(block) < full_block["bbox"][1]
        ]
        ordered.extend(order_band_columns(band, page_width))
        consumed.update(clean_text(block.get("block_id")) for block in band)
        ordered.append(full_block)

    remaining = [block for block in column_blocks if block.get("block_id") not in consumed]
    ordered.extend(order_band_columns(remaining, page_width))
    for index, block in enumerate(ordered, start=1):
        block["reading_order"] = index
    return ordered


def apply_layout_cleanup(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    page_count = len(pages)
    margin_counts: dict[str, int] = {}
    for page in pages:
        page_height = float(page.get("height") or 0.0)
        seen_on_page: set[str] = set()
        for block in page.get("text_blocks", []):
            text_key = normalize_repeated_margin_text(block.get("text"))
            if not text_key or len(text_key) < 4:
                continue
            if is_margin_block(block, page_height):
                seen_on_page.add(text_key)
        for text_key in seen_on_page:
            margin_counts[text_key] = margin_counts.get(text_key, 0) + 1

    repeat_threshold = max(2, int(math.ceil(page_count * 0.25))) if page_count else 2
    repeated_margin_text = {text for text, count in margin_counts.items() if count >= repeat_threshold}

    for page in pages:
        page_width = float(page.get("width") or 0.0)
        page_height = float(page.get("height") or 0.0)
        kept: list[dict[str, Any]] = []
        filtered = 0
        for block in page.get("text_blocks", []):
            text = clean_text(block.get("text"))
            text_key = normalize_repeated_margin_text(text)
            is_margin = is_margin_block(block, page_height)
            if is_side_margin_noise(block, page_width, page_height):
                filtered += 1
                continue
            if is_margin and (text_key in repeated_margin_text or is_page_number_text(text)):
                filtered += 1
                continue
            kept.append(block)

        column_count = detect_column_count(kept, page_width)
        ordered = mark_two_column_order(kept, page_width) if column_count == 2 else mark_single_column_order(kept)
        page["text_blocks"] = ordered
        page["text"] = clean_text("\n\n".join(block["text"] for block in ordered))
        page["layout_column_count"] = column_count
        page["filtered_header_footer_blocks"] = filtered
    return pages


def extract_layout_pages_with_pymupdf(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        import fitz
    except Exception:
        return []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return []

    pages: list[dict[str, Any]] = []
    try:
        for page_index, page in enumerate(doc, start=1):
            raw_blocks = page.get_text("dict").get("blocks", [])
            text_blocks: list[dict[str, Any]] = []
            image_blocks: list[dict[str, Any]] = []
            for block_index, block in enumerate(raw_blocks, start=1):
                bbox = [float(value) for value in block.get("bbox", [])]
                if len(bbox) != 4 or bbox_area(bbox) <= 0:
                    continue
                if block.get("type") == 0:
                    text = fitz_block_text(block)
                    if not text:
                        continue
                    font_size, font_flags = fitz_block_font_stats(block)
                    text_blocks.append(
                        {
                            "block_id": f"p{page_index}_b{block_index}",
                            "text": text,
                            "bbox": bbox,
                            "font_size": font_size,
                            "font_flags": font_flags,
                            "line_count": len(block.get("lines", [])),
                        }
                    )
                elif block.get("type") == 1:
                    image_blocks.append(
                        {
                            "block_id": f"p{page_index}_img{block_index}",
                            "bbox": bbox,
                            "area": round(bbox_area(bbox), 2),
                        }
                    )
            image_blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
            pages.append(
                {
                    "page": page_index,
                    "width": round(float(page.rect.width), 2),
                    "height": round(float(page.rect.height), 2),
                    "text": clean_text("\n\n".join(block["text"] for block in text_blocks)),
                    "text_blocks": text_blocks,
                    "image_blocks": image_blocks,
                }
            )
    finally:
        doc.close()
    return apply_layout_cleanup(pages)


def table_to_text(rows: list[list[Any]]) -> str:
    clean_rows: list[list[str]] = []
    max_cols = 0
    for row in rows:
        clean_row = [clean_text(cell) for cell in (row or [])]
        if any(clean_row):
            clean_rows.append(clean_row)
            max_cols = max(max_cols, len(clean_row))
    if not clean_rows:
        return ""
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in clean_rows]
    return clean_text("\n".join(" | ".join(row) for row in normalized_rows))


def extract_tables_with_pdfplumber(pdf_path: Path) -> dict[int, list[dict[str, Any]]]:
    try:
        import pdfplumber
    except Exception:
        return {}

    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_tables: list[dict[str, Any]] = []
                try:
                    tables = page.find_tables()
                except Exception:
                    tables = []
                for table_index, table in enumerate(tables, start=1):
                    try:
                        rows = table.extract() or []
                    except Exception:
                        rows = []
                    content = table_to_text(rows)
                    if not content:
                        continue
                    row_count = len([row for row in rows if any(clean_text(cell) for cell in (row or []))])
                    col_count = max((len(row or []) for row in rows), default=0)
                    if row_count < 2 or col_count < 2:
                        continue
                    bbox = [float(value) for value in getattr(table, "bbox", []) or []]
                    page_tables.append(
                        {
                            "table_index": table_index,
                            "bbox": bbox if len(bbox) == 4 else [],
                            "content": content,
                            "rows": row_count,
                            "cols": col_count,
                        }
                    )
                if page_tables:
                    tables_by_page[page_index] = page_tables
    except Exception:
        return {}
    return tables_by_page


def count_keywords(text: str, keywords: tuple[str, ...]) -> int:
    lowered = text.lower()
    score = 0
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower:
            score += lowered.count(keyword_lower)
    return score


def score_template(text: str, template: PaperChunkTemplate) -> int:
    if template.domain == "general":
        return 0
    return count_keywords(text, template.keywords)


def format_domain_candidates(candidates: tuple[tuple[str, float], ...]) -> str:
    return ";".join(f"{domain}:{score:.3f}" for domain, score in candidates)


def compute_domain_scores(sample: str) -> tuple[dict[str, int], dict[str, int], tuple[tuple[str, float], ...]]:
    scores = {
        name: score_template(sample, template)
        for name, template in PAPER_CHUNK_TEMPLATES.items()
        if name != "general"
    }
    strong_scores = {name: count_keywords(sample, DOMAIN_STRONG_KEYWORDS.get(name, ())) for name in scores}
    weighted_scores = {name: float(scores[name] + strong_scores[name] * 2) for name in scores}
    total = sum(weighted_scores.values())
    if total <= 0:
        candidates = tuple((name, 0.0) for name in sorted(weighted_scores))
    else:
        candidates = tuple(
            sorted(
                ((name, round(weight / total, 4)) for name, weight in weighted_scores.items()),
                key=lambda item: item[1],
                reverse=True,
            )
        )
    return scores, strong_scores, candidates


def select_auto_template_from_scores(
    scores: dict[str, int],
    strong_scores: dict[str, int],
) -> tuple[str, str]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ordered[0] if ordered else ("general", 0)
    second_name, second_score = ordered[1] if len(ordered) > 1 else ("general", 0)

    if best_score < DOMAIN_MIN_SCORES.get(best_name, 2):
        return "general", "weak_domain_signal"
    if DOMAIN_STRONG_KEYWORDS.get(best_name) and strong_scores.get(best_name, 0) == 0:
        return "general", "missing_strong_domain_keyword"
    if second_score and best_score < second_score * 1.15 and strong_scores.get(best_name, 0) <= strong_scores.get(second_name, 0):
        return "general", "ambiguous_domain_signal"
    return best_name, "auto_domain_selected"


def chunk_template_decision(
    requested: str,
    pdf_path: Path,
    pages: list[str],
) -> TemplateDecision:
    requested = clean_text(requested).lower() or "auto"
    sample = clean_text("\n".join([pdf_path.stem, *pages[: min(5, len(pages))]]))
    scores, strong_scores, candidates = compute_domain_scores(sample)
    auto_name, auto_reason = select_auto_template_from_scores(scores, strong_scores)
    auto_confidence = dict(candidates).get(auto_name, 0.0) if auto_name != "general" else 0.0

    if requested != "auto" and requested in PAPER_CHUNK_TEMPLATES:
        selected_name = requested
        selection_reason = "user_selected"
        if auto_name not in {"general", selected_name}:
            selection_reason = f"user_selected_auto_disagrees:{auto_name}"
    else:
        selected_name = auto_name
        selection_reason = auto_reason

    return TemplateDecision(
        template=PAPER_CHUNK_TEMPLATES.get(selected_name, PAPER_CHUNK_TEMPLATES["general"]),
        requested_template=requested if requested in {"auto", *PAPER_CHUNK_TEMPLATES.keys()} else "auto",
        auto_template=auto_name,
        auto_confidence=round(float(auto_confidence), 4),
        domain_candidates=candidates,
        selection_reason=selection_reason,
    )


def select_chunk_template(
    requested: str,
    pdf_path: Path,
    pages: list[str],
) -> PaperChunkTemplate:
    return chunk_template_decision(requested, pdf_path, pages).template


def effective_chunk_size(template: PaperChunkTemplate, requested: int | None) -> int:
    if requested and requested != 900:
        return requested
    return template.chunk_size


def normalize_heading_text(line: str) -> str:
    line = clean_text(line)
    line = re.sub(r"^\s*(?:[IVXLC]+|\d+(?:\.\d+)*|[一二三四五六七八九十]+)[\.\s、-]+", "", line, flags=re.I)
    line = re.sub(r"[:：]\s*$", "", line)
    return line.strip()


def canonical_section(line: str, template: PaperChunkTemplate) -> str:
    raw = clean_text(line)
    if not raw:
        return ""
    normalized = normalize_heading_text(raw)
    lowered = normalized.lower()
    if len(normalized) > 140:
        return ""

    aliases = sorted(set(template.section_aliases), key=len, reverse=True)
    for alias in aliases:
        alias_lower = alias.lower()
        if lowered == alias_lower or lowered.startswith(f"{alias_lower} "):
            return alias
        if normalized.startswith(alias):
            return alias

    word_count = len(re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", normalized))
    if word_count <= 8 and re.match(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+[A-Z][A-Za-z /&-]+$", raw):
        return normalized
    return ""


def strip_heading_prefix(block: str, heading: str) -> str:
    text = clean_text(block)
    if not text or not heading:
        return text
    heading_pattern = re.escape(heading)
    text = re.sub(rf"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?\.?\s*{heading_pattern}\s*[:：.-]?\s*", "", text, flags=re.I)
    return clean_text(text)


def infer_structure_type(block: str, template: PaperChunkTemplate) -> str:
    text = clean_text(block)
    if not text:
        return ""
    if CAPTION_RE.search(text):
        return "caption"
    pattern_groups = STRUCTURE_TYPE_PATTERNS.get(template.domain, ()) + STRUCTURE_TYPE_PATTERNS.get("general", ())
    for structure_type, pattern in pattern_groups:
        if re.search(pattern, text, re.I):
            return structure_type
    return ""


def is_template_protected_block(block: str, template: PaperChunkTemplate) -> bool:
    text = clean_text(block)
    if PROTECTED_BLOCK_RE.search(text):
        return True
    if infer_structure_type(text, template) in {
        "algorithm",
        "theorem",
        "definition",
        "proof",
        "equation",
        "regression",
        "variable",
        "study_design",
        "participants",
        "intervention",
        "outcome",
        "adverse_event",
        "statistical_analysis",
    }:
        return True
    lowered = normalize_heading_text(text).lower()
    for term in template.protected_terms:
        term_lower = term.lower()
        if lowered == term_lower or lowered.startswith(f"{term_lower} "):
            return True
        if text.startswith(term):
            return True
    return False


REF_RE = re.compile(r"(图|圖|Fig\.?|Figure|表|Table)\s*([0-9A-Za-z\.\-]+)", re.I)


def extract_document_refs(text: str) -> str:
    refs: list[str] = []
    for match in REF_RE.finditer(text or ""):
        raw_type = match.group(1).lower()
        kind = "table" if raw_type.startswith("表") or raw_type.startswith("table") else "figure"
        refs.append(f"{kind}:{match.group(2).strip('.-')}")
    return ";".join(dict.fromkeys(refs))


def median_font_size(blocks: list[dict[str, Any]]) -> float:
    sizes = sorted(float(block.get("font_size") or 0.0) for block in blocks if float(block.get("font_size") or 0.0) > 0)
    if not sizes:
        return 0.0
    midpoint = len(sizes) // 2
    if len(sizes) % 2:
        return sizes[midpoint]
    return (sizes[midpoint - 1] + sizes[midpoint]) / 2.0


def looks_like_layout_title(block: dict[str, Any], template: PaperChunkTemplate, median_font: float) -> bool:
    text = clean_text(block.get("text"))
    if not text or len(text) > 160:
        return False
    if canonical_section(text, template):
        return True
    font_size = float(block.get("font_size") or 0.0)
    if median_font > 0 and font_size >= median_font * 1.22:
        word_count = len(re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", normalize_heading_text(text)))
        return 1 <= word_count <= 14
    return False


def looks_like_layout_table(text: str) -> bool:
    text = clean_text(text)
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    table_terms = re.search(r"\b(accuracy|precision|recall|f1|auc|p-value|ci|or|hr)\b|指标|准确率|召回率|变量|回归|置信区间", text, re.I)
    column_like_lines = sum(1 for line in lines if len([part for part in re.split(r"\s{2,}|\|", line) if part.strip()]) >= 3)
    numeric_lines = sum(1 for line in lines if len(re.findall(r"\d+(?:[.,]\d+)?", line)) >= 2)
    return bool(table_terms and (column_like_lines >= 1 or numeric_lines >= 1)) or column_like_lines >= 2 or numeric_lines >= 3


def layout_node_type_for_block(block: dict[str, Any], template: PaperChunkTemplate, median_font: float) -> str:
    text = clean_text(block.get("text"))
    if looks_like_layout_title(block, template, median_font):
        return "title"
    if CAPTION_RE.search(text):
        return "caption"
    if looks_like_layout_table(text):
        return "table"
    if template.domain == "math" and EQUATION_HINT_RE.search(text):
        return "equation"
    if EQUATION_HINT_RE.search(text) and len(text) < 500 and len(re.findall(r"[=∑∫≤≥≈]", text)) >= 2:
        return "equation"
    return "text"


def should_keep_image_block(block: dict[str, Any], page_width: float, page_height: float) -> bool:
    area = float(block.get("area") or bbox_area(block.get("bbox", [])))
    page_area = max(1.0, page_width * page_height)
    bbox = block.get("bbox", [])
    if len(bbox) != 4:
        return False
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    if area / page_area >= 0.012:
        return True
    return area >= 1800 and width >= 48 and height >= 36


def base_node_metadata(
    doc_id: str,
    page_index: int,
    pdf_path: Path,
    template_meta: dict[str, Any],
    section: str,
    section_id: str,
    page_layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "doc_id": doc_id,
        "page": page_index,
        "source_ref": f"{pdf_path.name} page {page_index}",
        **template_meta,
        "section": section,
        "section_id": section_id,
    }
    if page_layout:
        meta["page_width"] = page_layout.get("width", "")
        meta["page_height"] = page_layout.get("height", "")
        meta["layout_column_count"] = page_layout.get("layout_column_count", "")
        meta["filtered_header_footer_blocks"] = page_layout.get("filtered_header_footer_blocks", "")
    return meta


def can_merge_layout_text(left: dict[str, Any], right: dict[str, Any], chunk_size: int) -> bool:
    if clean_text(left.get("node_type")) != "text" or clean_text(right.get("node_type")) != "text":
        return False
    if clean_text(left.get("layout_parser")) != "pymupdf" or clean_text(right.get("layout_parser")) != "pymupdf":
        return False
    if clean_text(left.get("doc_id")) != clean_text(right.get("doc_id")):
        return False
    if str(left.get("page", "")) != str(right.get("page", "")):
        return False
    if clean_text(left.get("section_id")) != clean_text(right.get("section_id")):
        return False
    merge_blocking_structures = {
        "algorithm",
        "theorem",
        "definition",
        "proof",
        "equation",
        "regression",
        "variable",
        "study_design",
        "participants",
        "intervention",
        "outcome",
        "adverse_event",
        "statistical_analysis",
    }
    if clean_text(left.get("structure_type")) in merge_blocking_structures:
        return False
    if clean_text(right.get("structure_type")) in merge_blocking_structures:
        return False
    merged_len = len(clean_text(left.get("content"))) + len(clean_text(right.get("content"))) + 2
    return merged_len <= chunk_size


def merge_two_layout_text_nodes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged["content"] = clean_text(f"{left.get('content', '')}\n{right.get('content', '')}")
    merged["bbox"] = bbox_to_json(union_bbox([json_to_bbox(left.get("bbox")), json_to_bbox(right.get("bbox"))]))
    merged["bbox_source"] = "pymupdf_text_block_merged"
    merged["layout_block_id"] = ";".join(
        item
        for item in [clean_text(left.get("layout_block_id")), clean_text(right.get("layout_block_id"))]
        if item
    )
    try:
        merged["line_count"] = int(left.get("line_count") or 0) + int(right.get("line_count") or 0)
    except (TypeError, ValueError):
        merged["line_count"] = left.get("line_count", "")
    merged["chunk_strategy"] = "layout_merged_text"
    return merged


def merge_adjacent_layout_text_nodes(nodes: list[dict[str, Any]], chunk_size: int) -> list[dict[str, Any]]:
    merged_nodes: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            merged_nodes.append(pending)
            pending = None

    for node in nodes:
        if pending is None:
            if clean_text(node.get("node_type")) == "text" and clean_text(node.get("layout_parser")) == "pymupdf":
                pending = dict(node)
            else:
                merged_nodes.append(node)
            continue

        if can_merge_layout_text(pending, node, chunk_size):
            pending = merge_two_layout_text_nodes(pending, node)
            continue

        flush_pending()
        if clean_text(node.get("node_type")) == "text" and clean_text(node.get("layout_parser")) == "pymupdf":
            pending = dict(node)
        else:
            merged_nodes.append(node)

    flush_pending()
    return merged_nodes


def split_long_block(block: str, chunk_size: int) -> list[str]:
    block = clean_text(block)
    if not block:
        return []
    if len(block) <= chunk_size:
        return [block]

    sentences = re.split(r"(?<=[。！？.!?])\s+", block)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(sentence), chunk_size):
                chunks.append(sentence[start : start + chunk_size].strip())
            continue
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def split_blocks(text: str, chunk_size: int = 900) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) <= 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        blocks = []
        current: list[str] = []
        for line in lines:
            current.append(line)
            if len(" ".join(current)) >= chunk_size or CAPTION_RE.search(line):
                blocks.append("\n".join(current))
                current = []
        if current:
            blocks.append("\n".join(current))
    chunks: list[str] = []
    for block in blocks:
        chunks.extend(split_long_block(block, chunk_size))
    return chunks


def split_template_blocks(
    text: str,
    template: PaperChunkTemplate,
    chunk_size: int,
    active_section: str = "",
) -> tuple[list[dict[str, str]], str]:
    text = clean_text(text)
    if not text:
        return [], active_section

    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(raw_blocks) <= 1:
        raw_blocks = [line.strip() for line in text.splitlines() if line.strip()]

    chunks: list[dict[str, str]] = []
    current_lines: list[str] = []
    current_section = active_section
    current_kind = "paragraph"

    def flush(kind: str | None = None) -> None:
        nonlocal current_lines, current_kind
        if not current_lines:
            return
        block = clean_text("\n".join(current_lines))
        block_kind = kind or current_kind
        for part in split_long_block(block, chunk_size):
            chunks.append(
                {
                    "content": part,
                    "section": current_section,
                    "chunk_strategy": block_kind,
                    "structure_type": infer_structure_type(part, template),
                }
            )
        current_lines = []
        current_kind = "paragraph"

    for block in raw_blocks:
        is_caption = bool(CAPTION_RE.search(block))
        is_protected = is_template_protected_block(block, template)

        if is_caption or is_protected:
            flush()
            current_lines = [block]
            current_kind = "caption_or_protected" if is_caption else "protected_paper_block"
            flush()
            continue

        heading = canonical_section(block, template)
        if heading:
            flush()
            current_section = heading
            chunks.append(
                {
                    "content": clean_text(heading),
                    "section": current_section,
                    "chunk_strategy": "section_title",
                    "structure_type": "section_title",
                }
            )
            remainder = strip_heading_prefix(block, heading)
            if remainder and remainder != clean_text(block):
                current_lines = [remainder]
                current_kind = "section_aware_paragraph"
            continue

        if not current_lines:
            current_lines = [block]
            current_kind = "section_aware_paragraph"
            continue

        candidate = clean_text("\n".join([*current_lines, block]))
        if len(candidate) > chunk_size:
            flush()
            current_lines = [block]
            current_kind = "section_aware_paragraph"
        else:
            current_lines.append(block)

    flush()
    return chunks, current_section


def node_type_for_block(block: str, chunk_strategy: str = "", template: PaperChunkTemplate | None = None) -> str:
    if chunk_strategy == "section_title":
        return "title"
    if CAPTION_RE.search(block):
        return "caption"
    if TABLE_HINT_RE.search(block) and (len(block.splitlines()) >= 2 or re.search(r"\d+\s+\d+\s+\d+", block)):
        return "table"
    if template and template.domain == "math" and EQUATION_HINT_RE.search(block):
        return "equation"
    if EQUATION_HINT_RE.search(block) and len(block) < 500 and len(re.findall(r"[=∑∫≤≥≈]", block)) >= 2:
        return "equation"
    return "text"


def flatten_mineru_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return clean_text("\n".join(flatten_mineru_value(item) for item in value if flatten_mineru_value(item)))
    if isinstance(value, dict):
        preferred = [
            value.get("text"),
            value.get("content"),
            value.get("html"),
            value.get("latex"),
            value.get("paragraph_content"),
            value.get("table_body"),
            value.get("table_data"),
            value.get("table_caption"),
            value.get("image_caption"),
            value.get("chart_caption"),
            value.get("image_path"),
            value.get("img_path"),
        ]
        text = "\n".join(flatten_mineru_value(item) for item in preferred if flatten_mineru_value(item))
        return text or clean_text(json.dumps(value, ensure_ascii=False))
    return clean_text(value)


def mineru_item_page(item: dict[str, Any]) -> int:
    if "page_idx" in item:
        return int(item.get("page_idx") or 0) + 1
    if "page_index" in item:
        return int(item.get("page_index") or 0) + 1
    return max(1, int(item.get("page") or item.get("page_no") or 1))


def mineru_item_node_type(item: dict[str, Any], content: str, template: PaperChunkTemplate) -> str:
    raw_type = clean_text(
        item.get("type")
        or item.get("block_type")
        or item.get("category_type")
        or item.get("category")
        or item.get("label")
        or "text"
    ).lower()
    normalized = raw_type.replace("-", "_").replace(" ", "_")
    if normalized in {"image", "image_body", "figure", "figure_body", "chart", "chart_body"}:
        return "figure"
    if normalized in {"table", "table_body"}:
        return "table"
    if normalized in {"image_caption", "table_caption", "chart_caption", "caption"}:
        return "caption"
    if normalized in {"equation", "interline_equation", "inline_equation", "formula"}:
        return "equation"
    if normalized in {"title", "doc_title"}:
        return "title"
    return node_type_for_block(content, template=template)


def mineru_item_image_path(item: dict[str, Any], content_list_path: Path) -> str:
    value = clean_text(item.get("image_path") or item.get("img_path") or item.get("path") or "")
    if not value or re.match(r"^(https?|data):", value, flags=re.I):
        return value
    path = Path(value)
    if not path.is_absolute():
        path = content_list_path.parent / path
    return str(path)


def mineru_item_content(item: dict[str, Any], content_list_path: Path) -> str:
    parts = [
        item.get("text"),
        item.get("content"),
        item.get("html"),
        item.get("table_body"),
        item.get("table_data"),
        item.get("table_caption"),
        item.get("table_footnote"),
        item.get("image_caption"),
        item.get("image_footnote"),
        item.get("chart_caption"),
        item.get("chart_footnote"),
        item.get("latex"),
    ]
    image_path = mineru_item_image_path(item, content_list_path)
    if image_path:
        parts.append(f"Image path: {image_path}")
    return clean_text("\n".join(flatten_mineru_value(part) for part in parts if flatten_mineru_value(part)))


def load_mineru_items(content_list_path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(content_list_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("content_list") or payload.get("items") or payload.get("pages") or []
        source_name = payload.get("file_path") or payload.get("doc_id") or content_list_path.stem
    else:
        items = payload
        source_name = content_list_path.stem
    if items and isinstance(items[0], list):
        flattened: list[dict[str, Any]] = []
        for page_index, page_blocks in enumerate(items):
            for block in page_blocks:
                if isinstance(block, dict):
                    block.setdefault("page_idx", page_index)
                    flattened.append(block)
        items = flattened
    if items and isinstance(items[0], dict) and "blocks" in items[0]:
        flattened: list[dict[str, Any]] = []
        for page_index, page in enumerate(items):
            for block in page.get("blocks") or []:
                if isinstance(block, dict):
                    block.setdefault("page_idx", page.get("page_idx", page.get("page", page_index)))
                    flattened.append(block)
        items = flattened
    return [item for item in items if isinstance(item, dict)], str(source_name)


def content_list_to_nodes(
    content_list_path: Path,
    chunk_template: str = "auto",
    chunk_size: int = 900,
    source_pdf_path: Path | None = None,
) -> list[dict[str, Any]]:
    items, source_name = load_mineru_items(content_list_path)
    if source_pdf_path:
        source_name = source_pdf_path.name
    doc_id = normalize_doc_id(str(source_name))
    text_pages_by_index: dict[int, list[str]] = {}
    for item in items:
        content = mineru_item_content(item, content_list_path)
        if content:
            text_pages_by_index.setdefault(mineru_item_page(item), []).append(content)
    pages = ["\n".join(text_pages_by_index[index]) for index in sorted(text_pages_by_index)]
    decision = chunk_template_decision(chunk_template, source_pdf_path or Path(source_name), pages)
    template = decision.template
    template_meta = decision_metadata(decision)
    resolved_chunk_size = effective_chunk_size(template, chunk_size)
    counters: dict[tuple[str, int], int] = {}
    nodes: list[dict[str, Any]] = []
    page_seen: set[int] = set()
    active_section = ""
    active_section_id = ""
    for item in items:
        page = mineru_item_page(item)
        if page not in page_seen:
            page_seen.add(page)
            nodes.append(
                {
                    "node_id": make_node_id("page", doc_id, page),
                    "doc_id": doc_id,
                    "page": page,
                    "node_type": "page",
                    "content": f"Page {page} of {doc_id}",
                    "source_ref": f"{source_name} page {page}",
                    **template_meta,
                    "section": active_section,
                    "section_id": active_section_id,
                    "parent_chunk_id": "",
                    "chunk_level": "page",
                    "structure_type": "page",
                    "layout_parser": "mineru",
                }
            )
        content = mineru_item_content(item, content_list_path)
        if not content:
            continue
        node_type = mineru_item_node_type(item, content, template)
        chunks = [content]
        if node_type == "text":
            chunks = split_long_block(content, resolved_chunk_size)
        for chunk in chunks:
            node_type_for_chunk = node_type_for_block(chunk, template=template) if node_type == "text" else node_type
            counters[(node_type_for_chunk, page)] = counters.get((node_type_for_chunk, page), 0) + 1
            node_id = make_node_id(node_type_for_chunk, doc_id, page, counters[(node_type_for_chunk, page)])
            section = active_section
            section_id = active_section_id
            if node_type_for_chunk == "title":
                section = clean_text(normalize_heading_text(chunk))
                active_section = section
                active_section_id = node_id
                section_id = active_section_id
            parent_chunk_id = "" if node_type_for_chunk == "title" else active_section_id
            node = {
                "node_id": node_id,
                "doc_id": doc_id,
                "page": page,
                "node_type": node_type_for_chunk,
                "content": chunk,
                "source_ref": f"{source_name} page {page} {node_type_for_chunk}",
                **template_meta,
                "section": section,
                "section_id": section_id,
                "parent_chunk_id": parent_chunk_id,
                "chunk_level": "section" if node_type_for_chunk == "title" else "block",
                "chunk_strategy": "mineru_content_list",
                "structure_type": "section_title" if node_type_for_chunk == "title" else infer_structure_type(chunk, template),
                "layout_parser": "mineru",
                "layout_role": clean_text(item.get("type") or item.get("block_type") or node_type_for_chunk),
                "bbox": bbox_to_json(item.get("bbox", [])),
                "bbox_source": "mineru",
            }
            image_path = mineru_item_image_path(item, content_list_path)
            if image_path and node_type_for_chunk in {"figure", "table", "caption"}:
                node["image_path"] = image_path
                node["crop_image_path"] = image_path
            nodes.append(node)
    return enrich_node_context(nodes)


def find_mineru_content_list(output_dir: Path, pdf_path: Path | None = None) -> Path | None:
    if not output_dir.exists():
        return None
    stems = []
    if pdf_path:
        stems = [pdf_path.stem, normalize_doc_id(pdf_path.name), normalize_doc_id(pdf_path.stem)]
    patterns = ["*_content_list_v2.json", "*_content_list.json", "content_list_v2.json", "content_list.json"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(output_dir.rglob(pattern))
    candidates = sorted(dict.fromkeys(candidates), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    if not stems:
        return candidates[0] if candidates else None
    for candidate in candidates:
        name_blob = " ".join([candidate.name, candidate.parent.name, str(candidate.parent)])
        if any(stem and stem in name_blob for stem in stems):
            return candidate
    return candidates[0] if len(candidates) == 1 else None


def env_bool(name: str, default: bool = False) -> bool:
    value = clean_text(read_env(name, "1" if default else "0")).lower()
    return value in {"1", "true", "yes", "on"}


def windows_user_env(name: str) -> str:
    if winreg is None:
        return ""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value).strip()
    except OSError:
        return ""


def load_local_env() -> dict[str, str]:
    global _LOCAL_ENV_CACHE
    if _LOCAL_ENV_CACHE is not None:
        return _LOCAL_ENV_CACHE
    values: dict[str, str] = {}
    for path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key:
                values[key] = value
    _LOCAL_ENV_CACHE = values
    return values


def read_env(name: str, default: str = "") -> str:
    return clean_text(os.getenv(name)) or clean_text(load_local_env().get(name)) or windows_user_env(name) or default


def mineru_api_token() -> str:
    for name in ("MINERU_API_KEY", "MINERU_TOKEN", "MINERU_API_TOKEN"):
        token = read_env(name)
        if token:
            return token
    return ""


def mineru_api_base(api_url: str) -> str:
    url = clean_text(api_url or read_env("MINERU_API_URL") or "https://mineru.net/api/v4")
    url = url.rstrip("/")
    for suffix in (
        "/file-urls/batch",
        "/extract/task/batch",
        "/extract/task",
    ):
        if url.endswith(suffix):
            return url[: -len(suffix)].rstrip("/")
    return url


def mineru_cloud_api_enabled(api_url: str) -> bool:
    mode = clean_text(read_env("MINERU_API_MODE", "")).lower()
    if mode in {"local", "cli", "off", "false", "0"}:
        return False
    if mode in {"cloud", "openapi", "api", "remote"}:
        return bool(mineru_api_token())
    return bool(mineru_api_token())


def mineru_json_request(method: str, url: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {
        "Accept": "*/*",
        "Authorization": f"Bearer {token}",
    }
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=int(read_env("MINERU_API_HTTP_TIMEOUT", "90"))) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MinerU API HTTP {exc.code}: {preview(detail, 360)}") from exc
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MinerU API returned non-JSON response: {preview(body, 360)}") from exc
    if int(result.get("code", 0)) != 0:
        raise RuntimeError(f"MinerU API error: {result.get('msg') or preview(result, 360)}")
    return result


def mineru_upload_file(upload_url: str, pdf_path: Path) -> None:
    with pdf_path.open("rb") as f:
        data = f.read()
    request = urllib.request.Request(upload_url, data=data, method="PUT")
    try:
        with urllib.request.urlopen(request, timeout=int(read_env("MINERU_API_UPLOAD_TIMEOUT", "600"))) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"upload returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MinerU API upload failed HTTP {exc.code}: {preview(detail, 360)}") from exc


def first_upload_url(payload: dict[str, Any]) -> str:
    file_urls = payload.get("data", {}).get("file_urls", [])
    if not file_urls:
        return ""
    first = file_urls[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        for key in ("url", "upload_url", "file_url"):
            if clean_text(first.get(key)):
                return clean_text(first.get(key))
    return ""


def find_first_key(payload: Any, names: set[str]) -> str:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in names and isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
        for value in payload.values():
            found = find_first_key(value, names)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_first_key(item, names)
            if found:
                return found
    return ""


def collect_states(payload: Any) -> set[str]:
    states: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"state", "status"} and isinstance(value, str):
                states.add(value.lower())
            else:
                states.update(collect_states(value))
    elif isinstance(payload, list):
        for item in payload:
            states.update(collect_states(item))
    return states


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    target_root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            try:
                target.relative_to(target_root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe file path in MinerU zip: {member.filename}")
            archive.extract(member, output_dir)


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=int(read_env("MINERU_API_DOWNLOAD_TIMEOUT", "600"))) as response:
        with target.open("wb") as f:
            shutil.copyfileobj(response, f)


def run_mineru_cloud_api(
    pdf_path: Path,
    output_dir: Path,
    method: str = "auto",
    lang: str = "",
    api_url: str = "",
) -> Path:
    token = mineru_api_token()
    if not token:
        raise RuntimeError("MINERU_API_KEY is required when MINERU_API_MODE=cloud.")
    base_url = mineru_api_base(api_url)
    model_version = read_env("MINERU_API_MODEL_VERSION") or read_env("MINERU_MODEL_VERSION") or "vlm"
    language = clean_text(lang or read_env("MINERU_API_LANGUAGE") or read_env("MINERU_LANG") or "ch")
    data_id = normalize_doc_id(pdf_path.stem)
    payload: dict[str, Any] = {
        "files": [{"name": pdf_path.name, "data_id": data_id}],
        "model_version": model_version,
        "enable_formula": env_bool("MINERU_API_ENABLE_FORMULA", True),
        "enable_table": env_bool("MINERU_API_ENABLE_TABLE", True),
        "language": language,
    }
    if read_env("MINERU_API_IS_OCR"):
        payload["files"][0]["is_ocr"] = env_bool("MINERU_API_IS_OCR", method == "ocr")
    elif method == "ocr":
        payload["files"][0]["is_ocr"] = True

    print(f"> MinerU API request upload URL: {base_url}/file-urls/batch")
    upload_payload = mineru_json_request("POST", f"{base_url}/file-urls/batch", token, payload)
    batch_id = clean_text(upload_payload.get("data", {}).get("batch_id"))
    upload_url = first_upload_url(upload_payload)
    if not batch_id or not upload_url:
        raise RuntimeError(f"MinerU API did not return batch_id/file_urls: {preview(upload_payload, 360)}")

    print("> MinerU API upload PDF")
    mineru_upload_file(upload_url, pdf_path)

    poll_url = f"{base_url}/extract-results/batch/{batch_id}"
    timeout_s = int(read_env("MINERU_API_TIMEOUT", "1800"))
    interval_s = max(2, int(read_env("MINERU_API_POLL_INTERVAL", "5")))
    deadline = time.time() + timeout_s
    zip_url = ""
    last_payload: dict[str, Any] = {}
    print(f"> MinerU API polling batch_id={batch_id}")
    while time.time() < deadline:
        last_payload = mineru_json_request("GET", poll_url, token)
        zip_url = find_first_key(last_payload, {"full_zip_url", "zip_url"})
        if zip_url:
            break
        states = collect_states(last_payload)
        if states & {"failed", "fail", "error"}:
            raise RuntimeError(f"MinerU API task failed: {preview(last_payload, 360)}")
        time.sleep(interval_s)
    if not zip_url:
        raise RuntimeError(f"MinerU API timed out waiting for result: {preview(last_payload, 360)}")

    zip_path = output_dir / f"{data_id}_mineru_result.zip"
    extract_dir = output_dir / data_id
    print("> MinerU API download result zip")
    download_file(zip_url, zip_path)
    safe_extract_zip(zip_path, extract_dir)
    content_list = find_mineru_content_list(output_dir, pdf_path)
    if not content_list:
        raise RuntimeError(f"MinerU API result downloaded but no content_list JSON was found under {output_dir}.")
    return content_list


def run_mineru(
    pdf_path: Path,
    output_dir: Path,
    backend: str = "pipeline",
    method: str = "auto",
    lang: str = "",
    api_url: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = find_mineru_content_list(output_dir, pdf_path)
    if existing and read_env("RAG_MINERU_FORCE", "0").lower() not in {"1", "true", "yes"}:
        return existing
    api_url = api_url or read_env("MINERU_API_URL", "")
    if mineru_cloud_api_enabled(api_url):
        return run_mineru_cloud_api(pdf_path, output_dir, method=method, lang=lang, api_url=api_url)

    mineru_exe = read_env("MINERU_BIN") or shutil.which("mineru")
    if not mineru_exe:
        raise RuntimeError(
            "MinerU is not installed or `mineru` is not on PATH. "
            'Install it with: pip install uv && uv pip install -U "mineru[all]"'
        )
    cmd = [mineru_exe, "-p", str(pdf_path), "-o", str(output_dir), "-b", backend, "-m", method]
    if api_url:
        cmd.extend(["--api-url", api_url])
    if lang:
        cmd.extend(["-l", lang])
    extra = read_env("MINERU_EXTRA_ARGS", "")
    if extra:
        cmd.extend(extra.split())
    env = os.environ.copy()
    if read_env("MINERU_MODEL_SOURCE"):
        env["MINERU_MODEL_SOURCE"] = read_env("MINERU_MODEL_SOURCE", "")
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    content_list = find_mineru_content_list(output_dir, pdf_path)
    if not content_list:
        raise RuntimeError(f"MinerU finished but no content_list JSON was found under {output_dir}.")
    return content_list


def mineru_pdf_to_nodes(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int = 900,
    chunk_template: str = "auto",
    backend: str = "pipeline",
    method: str = "auto",
    lang: str = "",
    api_url: str = "",
) -> list[dict[str, Any]]:
    content_list = run_mineru(pdf_path, output_dir, backend=backend, method=method, lang=lang, api_url=api_url)
    return content_list_to_nodes(content_list, chunk_template=chunk_template, chunk_size=chunk_size, source_pdf_path=pdf_path)


def decision_metadata(decision: TemplateDecision) -> dict[str, Any]:
    return {
        "paper_domain": decision.template.domain,
        "chunk_template": decision.template.domain,
        "requested_chunk_template": decision.requested_template,
        "auto_chunk_template": decision.auto_template,
        "auto_domain_confidence": decision.auto_confidence,
        "domain_candidates": format_domain_candidates(decision.domain_candidates),
        "template_selection_reason": decision.selection_reason,
    }


def context_preview(text: Any, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", clean_text(text))
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def enrich_node_context(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    captions_by_doc_page: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for node in nodes:
        doc_id = clean_text(node.get("doc_id"))
        page = str(node.get("page", ""))
        by_doc.setdefault(doc_id, []).append(node)
        if clean_text(node.get("node_type")) == "caption":
            captions_by_doc_page.setdefault((doc_id, page), []).append(node)

    for doc_nodes in by_doc.values():
        evidence_nodes = [node for node in doc_nodes if clean_text(node.get("node_type")) != "page"]
        for index, node in enumerate(evidence_nodes, start=1):
            prev_node = evidence_nodes[index - 2] if index > 1 else None
            next_node = evidence_nodes[index] if index < len(evidence_nodes) else None
            doc_id = clean_text(node.get("doc_id"))
            page = str(node.get("page", ""))
            nearby_captions = [
                caption
                for caption in captions_by_doc_page.get((doc_id, page), [])
                if caption.get("node_id") != node.get("node_id")
            ]

            node["chunk_order"] = index
            node["previous_node_id"] = prev_node.get("node_id", "") if prev_node else ""
            node["next_node_id"] = next_node.get("node_id", "") if next_node else ""
            node["previous_chunk_preview"] = context_preview(prev_node.get("content", "")) if prev_node else ""
            node["next_chunk_preview"] = context_preview(next_node.get("content", "")) if next_node else ""
            node["nearby_caption_refs"] = ";".join(
                clean_text(caption.get("node_id")) for caption in nearby_captions[:6] if clean_text(caption.get("node_id"))
            )
            node["explicit_refs"] = extract_document_refs(node.get("content", ""))

        for index, node in enumerate(doc_nodes, start=1):
            node.setdefault("chunk_order", index)
            node.setdefault("previous_node_id", "")
            node.setdefault("next_node_id", "")
            node.setdefault("previous_chunk_preview", "")
            node.setdefault("next_chunk_preview", "")
            node.setdefault("nearby_caption_refs", "")
            node.setdefault("explicit_refs", extract_document_refs(node.get("content", "")))

    return nodes


def build_nodes_from_layout_pages(
    pdf_path: Path,
    doc_id: str,
    layout_pages: list[dict[str, Any]],
    tables_by_page: dict[int, list[dict[str, Any]]],
    decision: TemplateDecision,
    chunk_size: int,
) -> list[dict[str, Any]]:
    template = decision.template
    template_meta = decision_metadata(decision)
    nodes: list[dict[str, Any]] = []
    counters: dict[tuple[str, int], int] = {}
    active_section = ""
    active_section_id = ""

    for page_layout in layout_pages:
        page_index = int(page_layout.get("page") or len(nodes) + 1)
        page_node_id = make_node_id("page", doc_id, page_index)
        nodes.append(
            {
                "node_id": page_node_id,
                "node_type": "page",
                "content": f"Page {page_index} of {doc_id}",
                "chunk_level": "page",
                "structure_type": "page",
                "parent_chunk_id": "",
                "layout_parser": "pymupdf",
                **base_node_metadata(doc_id, page_index, pdf_path, template_meta, active_section, active_section_id, page_layout),
            }
        )

        text_blocks = list(page_layout.get("text_blocks") or [])
        median_font = median_font_size(text_blocks)
        for block in text_blocks:
            block_text = clean_text(block.get("text"))
            if not block_text:
                continue
            node_type = layout_node_type_for_block(block, template, median_font)
            heading = canonical_section(block_text, template) if node_type == "title" else ""
            if node_type == "title":
                content_parts = [clean_text(heading or normalize_heading_text(block_text))]
            elif node_type in {"caption", "table", "equation"} or is_template_protected_block(block_text, template):
                content_parts = [block_text]
            else:
                content_parts = split_long_block(block_text, chunk_size)

            for part_index, part in enumerate(content_parts, start=1):
                if not part:
                    continue
                part_type = node_type
                if part_index > 1 and node_type == "title":
                    part_type = "text"
                counters[(part_type, page_index)] = counters.get((part_type, page_index), 0) + 1
                node_id = make_node_id(part_type, doc_id, page_index, counters[(part_type, page_index)])
                section = active_section
                section_id = active_section_id
                if part_type == "title":
                    section = clean_text(heading or normalize_heading_text(part))
                    active_section = section
                    active_section_id = node_id
                    section_id = active_section_id
                parent_chunk_id = "" if part_type == "title" else active_section_id
                structure_type = "section_title" if part_type == "title" else infer_structure_type(part, template)
                nodes.append(
                    {
                        "node_id": node_id,
                        "node_type": part_type,
                        "content": part,
                        "bbox": bbox_to_json(block.get("bbox", [])),
                        "bbox_source": "pymupdf_text_block",
                        "layout_parser": "pymupdf",
                        "layout_block_id": block.get("block_id", ""),
                        "layout_role": part_type,
                        "layout_column": block.get("layout_column", ""),
                        "reading_order": block.get("reading_order", ""),
                        "font_size": block.get("font_size", ""),
                        "font_flags": block.get("font_flags", ""),
                        "line_count": block.get("line_count", ""),
                        "chunk_level": "section" if part_type == "title" else "block",
                        "parent_chunk_id": parent_chunk_id,
                        "chunk_strategy": "layout_section_title" if part_type == "title" else "layout_block",
                        "structure_type": structure_type,
                        **base_node_metadata(doc_id, page_index, pdf_path, template_meta, section, section_id, page_layout),
                    }
                )

                caption_match = CAPTION_RE.search(part)
                if caption_match and re.search(r"^(图|圖|Fig|Figure)", caption_match.group(1), re.I):
                    counters[("figure", page_index)] = counters.get(("figure", page_index), 0) + 1
                    nodes.append(
                        {
                            "node_id": make_node_id("figure", doc_id, page_index, counters[("figure", page_index)]),
                            "node_type": "figure",
                            "content": f"Figure node inferred from caption: {part}",
                            "bbox": bbox_to_json(block.get("bbox", [])),
                            "bbox_source": "caption_bbox_layout",
                            "layout_parser": "pymupdf",
                            "layout_block_id": block.get("block_id", ""),
                            "layout_role": "figure_from_caption",
                            "layout_column": block.get("layout_column", ""),
                            "reading_order": block.get("reading_order", ""),
                            "chunk_level": "block",
                            "parent_chunk_id": parent_chunk_id,
                            "chunk_strategy": "inferred_from_caption",
                            "structure_type": "figure",
                            **base_node_metadata(doc_id, page_index, pdf_path, template_meta, section, section_id, page_layout),
                        }
                    )
                if caption_match and re.search(r"^(表|Table)", caption_match.group(1), re.I):
                    counters[("table", page_index)] = counters.get(("table", page_index), 0) + 1
                    nodes.append(
                        {
                            "node_id": make_node_id("table", doc_id, page_index, counters[("table", page_index)]),
                            "node_type": "table",
                            "content": f"Table node inferred from caption: {part}",
                            "bbox": bbox_to_json(block.get("bbox", [])),
                            "bbox_source": "caption_bbox_layout",
                            "layout_parser": "pymupdf",
                            "layout_block_id": block.get("block_id", ""),
                            "layout_role": "table_from_caption",
                            "layout_column": block.get("layout_column", ""),
                            "reading_order": block.get("reading_order", ""),
                            "chunk_level": "block",
                            "parent_chunk_id": parent_chunk_id,
                            "chunk_strategy": "inferred_from_caption",
                            "structure_type": "table",
                            **base_node_metadata(doc_id, page_index, pdf_path, template_meta, section, section_id, page_layout),
                        }
                    )

        for table in tables_by_page.get(page_index, []):
            counters[("table", page_index)] = counters.get(("table", page_index), 0) + 1
            nodes.append(
                {
                    "node_id": make_node_id("table", doc_id, page_index, counters[("table", page_index)]),
                    "node_type": "table",
                    "content": f"Structured table extracted by pdfplumber:\n{table.get('content', '')}",
                    "bbox": bbox_to_json(table.get("bbox", [])),
                    "bbox_source": "pdfplumber_table",
                    "layout_parser": "pdfplumber",
                    "layout_block_id": f"p{page_index}_table{table.get('table_index', 0)}",
                    "layout_role": "structured_table",
                    "table_rows": table.get("rows", ""),
                    "table_cols": table.get("cols", ""),
                    "chunk_level": "block",
                    "parent_chunk_id": active_section_id,
                    "chunk_strategy": "structured_table",
                    "structure_type": "table",
                    **base_node_metadata(doc_id, page_index, pdf_path, template_meta, active_section, active_section_id, page_layout),
                }
            )

        for image in page_layout.get("image_blocks") or []:
            if not should_keep_image_block(image, float(page_layout.get("width") or 0.0), float(page_layout.get("height") or 0.0)):
                continue
            counters[("figure", page_index)] = counters.get(("figure", page_index), 0) + 1
            nodes.append(
                {
                    "node_id": make_node_id("figure", doc_id, page_index, counters[("figure", page_index)]),
                    "node_type": "figure",
                    "content": f"Figure region detected from PDF layout on page {page_index}.",
                    "bbox": bbox_to_json(image.get("bbox", [])),
                    "bbox_source": "pymupdf_image_block",
                    "layout_parser": "pymupdf",
                    "layout_block_id": image.get("block_id", ""),
                    "layout_role": "image_block",
                    "image_area": image.get("area", ""),
                    "chunk_level": "block",
                    "parent_chunk_id": active_section_id,
                    "chunk_strategy": "layout_image_block",
                    "structure_type": "figure",
                    **base_node_metadata(doc_id, page_index, pdf_path, template_meta, active_section, active_section_id, page_layout),
                }
            )

    nodes = merge_adjacent_layout_text_nodes(nodes, chunk_size)
    return enrich_node_context(nodes)


def pdf_to_nodes(pdf_path: Path, chunk_size: int = 900, chunk_template: str = "auto") -> list[dict[str, Any]]:
    doc_id = normalize_doc_id(pdf_path.name)
    layout_pages = extract_layout_pages_with_pymupdf(pdf_path)
    pages = [clean_text(page.get("text")) for page in layout_pages if clean_text(page.get("text"))]
    if not pages:
        pages = extract_pdf_pages(pdf_path)
    if not pages:
        pages = [
            (
                f"{pdf_path.name} could not be text-extracted in the current environment. "
                "Install pypdf/pdfplumber or export RAG-Anything content_list JSON, then rerun parsing."
            )
        ]

    decision = chunk_template_decision(chunk_template, pdf_path, pages)
    template = decision.template
    template_meta = decision_metadata(decision)
    resolved_chunk_size = effective_chunk_size(template, chunk_size)
    if layout_pages:
        tables_by_page = extract_tables_with_pdfplumber(pdf_path)
        return build_nodes_from_layout_pages(
            pdf_path,
            doc_id,
            layout_pages,
            tables_by_page,
            decision,
            resolved_chunk_size,
        )

    nodes: list[dict[str, Any]] = []
    counters: dict[tuple[str, int], int] = {}
    active_section = ""
    active_section_id = ""
    for page_index, page_text in enumerate(pages, start=1):
        nodes.append(
            {
                "node_id": make_node_id("page", doc_id, page_index),
                "doc_id": doc_id,
                "page": page_index,
                "node_type": "page",
                "content": f"Page {page_index} of {doc_id}",
                "source_ref": f"{pdf_path.name} page {page_index}",
                **template_meta,
                "section": active_section,
                "section_id": active_section_id,
                "parent_chunk_id": "",
                "chunk_level": "page",
                "structure_type": "page",
            }
        )
        block_infos, active_section = split_template_blocks(
            page_text,
            template,
            chunk_size=resolved_chunk_size,
            active_section=active_section,
        )
        if not block_infos:
            block_infos = [
                {
                    "content": f"No extractable text on page {page_index} of {pdf_path.name}.",
                    "section": active_section,
                    "chunk_strategy": "fallback_empty_page",
                    "structure_type": "",
                }
            ]
        for block_info in block_infos:
            block = block_info["content"]
            section = block_info.get("section", active_section)
            chunk_strategy = block_info.get("chunk_strategy", "section_aware_paragraph")
            structure_type = block_info.get("structure_type") or infer_structure_type(block, template)
            node_type = node_type_for_block(block, chunk_strategy=chunk_strategy, template=template)
            counters[(node_type, page_index)] = counters.get((node_type, page_index), 0) + 1
            node_id = make_node_id(node_type, doc_id, page_index, counters[(node_type, page_index)])
            if node_type == "title":
                active_section = section
                active_section_id = node_id
            parent_chunk_id = "" if node_type == "title" else active_section_id
            section_id = active_section_id if active_section_id else ""
            nodes.append(
                {
                    "node_id": node_id,
                    "doc_id": doc_id,
                    "page": page_index,
                    "node_type": node_type,
                    "content": block,
                    "source_ref": f"{pdf_path.name} page {page_index}",
                    **template_meta,
                    "section": section,
                    "section_id": section_id,
                    "parent_chunk_id": parent_chunk_id,
                    "chunk_level": "section" if node_type == "title" else "block",
                    "chunk_strategy": chunk_strategy,
                    "structure_type": structure_type,
                }
            )
            caption_match = CAPTION_RE.search(block)
            if caption_match and re.search(r"^(图|圖|Fig|Figure)", caption_match.group(1), re.I):
                counters[("figure", page_index)] = counters.get(("figure", page_index), 0) + 1
                nodes.append(
                    {
                        "node_id": make_node_id("figure", doc_id, page_index, counters[("figure", page_index)]),
                        "doc_id": doc_id,
                        "page": page_index,
                        "node_type": "figure",
                        "content": f"Figure node inferred from caption: {block}",
                        "source_ref": f"{pdf_path.name} page {page_index} figure/caption",
                        **template_meta,
                        "section": section,
                        "section_id": section_id,
                        "parent_chunk_id": parent_chunk_id,
                        "chunk_level": "block",
                        "chunk_strategy": "inferred_from_caption",
                        "structure_type": "figure",
                    }
                )
            if caption_match and re.search(r"^(表|Table)", caption_match.group(1), re.I):
                counters[("table", page_index)] = counters.get(("table", page_index), 0) + 1
                nodes.append(
                    {
                        "node_id": make_node_id("table", doc_id, page_index, counters[("table", page_index)]),
                        "doc_id": doc_id,
                        "page": page_index,
                        "node_type": "table",
                        "content": f"Table node inferred from caption: {block}",
                        "source_ref": f"{pdf_path.name} page {page_index} table/caption",
                        **template_meta,
                        "section": section,
                        "section_id": section_id,
                        "parent_chunk_id": parent_chunk_id,
                        "chunk_level": "block",
                        "chunk_strategy": "inferred_from_caption",
                        "structure_type": "table",
                    }
                )
    return enrich_node_context(nodes)


def load_manual_nodes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    rows = read_csv(path)
    nodes: list[dict[str, Any]] = []
    counters: dict[tuple[str, int], int] = {}
    for row in rows:
        doc_id = normalize_doc_id(row.get("doc_id") or row.get("document") or "manual")
        page = int(row.get("page") or 1)
        node_type = (row.get("node_type") or "text").strip()
        counters[(node_type, page)] = counters.get((node_type, page), 0) + 1
        row = {field: row.get(field, "") for field in NODE_FIELDS}
        row["doc_id"] = doc_id
        row["page"] = page
        row["node_type"] = node_type
        row["node_id"] = row.get("node_id") or make_node_id(node_type, doc_id, page, counters[(node_type, page)])
        nodes.append(row)
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDFs with MinerU or content_list into evidence nodes.")
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--content-list", default="", help="Optional MinerU/RAG-Anything content_list JSON.")
    parser.add_argument("--manual-nodes", default="data/manual_nodes.csv")
    parser.add_argument("--output", default=str(DEFAULT_NODES.relative_to(DEFAULT_NODES.parents[2])))
    parser.add_argument(
        "--parser",
        choices=["mineru", "native"],
        default=os.getenv("RAG_PDF_PARSER", "mineru"),
        help="PDF parser backend. mineru is the default; native keeps the old PyMuPDF/pdfplumber path.",
    )
    parser.add_argument("--mineru-output-dir", default=os.getenv("RAG_MINERU_OUTPUT_DIR", "outputs/mineru"))
    parser.add_argument("--mineru-api-url", default=os.getenv("MINERU_API_URL", ""), help="Optional existing MinerU FastAPI base URL.")
    parser.add_argument(
        "--mineru-backend",
        default=os.getenv("MINERU_BACKEND", "pipeline"),
        help="MinerU backend, e.g. pipeline, hybrid-auto-engine, vlm-auto-engine.",
    )
    parser.add_argument("--mineru-method", default=os.getenv("MINERU_METHOD", "auto"), help="MinerU method: auto, txt, or ocr.")
    parser.add_argument("--mineru-lang", default=os.getenv("MINERU_LANG", ""), help="Optional MinerU OCR language.")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument(
        "--chunk-template",
        choices=["auto", "general", "ai", "math", "finance", "medical"],
        default="auto",
        help="Paper-aware chunking template. Use auto to infer a domain from the paper text.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    nodes: list[dict[str, Any]] = []

    if args.content_list:
        nodes.extend(
            content_list_to_nodes(
                resolve_path(args.content_list),
                chunk_template=args.chunk_template,
                chunk_size=args.chunk_size,
            )
        )

    pdf_dir = resolve_path(args.pdf_dir)
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        if args.parser == "mineru":
            nodes.extend(
                mineru_pdf_to_nodes(
                    pdf_path,
                    output_dir=resolve_path(args.mineru_output_dir),
                    chunk_size=args.chunk_size,
                    chunk_template=args.chunk_template,
                    backend=args.mineru_backend,
                    method=args.mineru_method,
                    lang=args.mineru_lang,
                    api_url=args.mineru_api_url,
                )
            )
        else:
            nodes.extend(pdf_to_nodes(pdf_path, chunk_size=args.chunk_size, chunk_template=args.chunk_template))

    manual_path = resolve_path(args.manual_nodes)
    nodes.extend(load_manual_nodes(manual_path))

    write_jsonl(args.output, nodes)
    copy_jsonl_alias(args.output, LEGACY_NODES)
    if not resolve_path("data/questions.csv").exists():
        empty_questions_template()
    print(f"Wrote {len(nodes)} nodes to {resolve_path(args.output)}")
    if not nodes:
        print("No PDFs or manual nodes found. Put PDFs in data/pdfs/ or add data/manual_nodes.csv.")


if __name__ == "__main__":
    main()
