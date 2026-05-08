from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from pipeline_common import clean_text, ensure_project_dirs, preview, read_csv, resolve_path, write_csv


CARD_W = 1920
CARD_H = 1080
MARGIN = 48
LEFT_W = 580
GAP = 24
CHAIN_X = MARGIN + LEFT_W + GAP
CHAIN_W = 760
RIGHT_X = CHAIN_X + CHAIN_W + GAP
RIGHT_W = CARD_W - RIGHT_X - MARGIN

BG = (246, 248, 251)
PANEL = (255, 255, 255)
INK = (29, 37, 52)
MUTED = (90, 101, 118)
FAINT = (224, 230, 238)
BLUE = (53, 100, 225)
GREEN = (25, 135, 84)
ORANGE = (210, 110, 24)
PURPLE = (118, 80, 180)

ROLE_LABELS = {
    "main_evidence": "\u4e3b\u8bc1\u636e",
    "explicit_reference": "\u7f16\u53f7\u8bc1\u636e",
    "table_or_figure": "\u56fe\u8868\u8bc1\u636e",
    "caption": "\u56fe\u6ce8/\u8868\u9898",
    "context_text": "\u4e0a\u4e0b\u6587",
    "graph_neighbor": "\u5173\u7cfb\u8865\u5168",
    "visual_companion": "\u89c6\u89c9\u4f34\u968f\u8bc1\u636e",
}

TYPE_COLORS = {
    "text": BLUE,
    "table": GREEN,
    "figure": PURPLE,
    "caption": ORANGE,
    "page": MUTED,
}


def find_font() -> str | None:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = find_font()
    if not font_path:
        return ImageFont.load_default()
    if bold and "Windows/Fonts/msyh.ttc" in font_path:
        return ImageFont.truetype(font_path, size=size, index=1)
    return ImageFont.truetype(font_path, size=size)


FONT_TITLE = load_font(40, bold=True)
FONT_H2 = load_font(25, bold=True)
FONT_BODY = load_font(22)
FONT_SMALL = load_font(18)
FONT_TINY = load_font(16)


def text_width(text: str, font: ImageFont.ImageFont) -> int:
    bbox = font.getbbox(text)
    return max(0, bbox[2] - bbox[0])


def line_height(font: ImageFont.ImageFont, pad: int = 7) -> int:
    bbox = font.getbbox("Ag\u95ee")
    return max(1, bbox[3] - bbox[1] + pad)


def wrap_text(text: Any, font: ImageFont.ImageFont, max_width: int, max_lines: int | None = None) -> list[str]:
    text = clean_text(text).replace("\n", " ")
    if not text:
        return []
    lines: list[str] = []
    current = ""
    for char in text:
        trial = current + char
        if current and text_width(trial, font) > max_width:
            lines.append(current)
            current = char.lstrip()
            if max_lines and len(lines) >= max_lines:
                break
        else:
            current = trial
    if (not max_lines or len(lines) < max_lines) and current:
        lines.append(current)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
    if max_lines and len(lines) == max_lines and len("".join(lines)) < len(text):
        while lines[-1] and text_width(lines[-1] + "...", font) > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "..."
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: Any,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    max_lines: int | None = None,
    spacing: int = 7,
) -> int:
    x, y = xy
    lines = wrap_text(text, font, max_width, max_lines=max_lines)
    lh = line_height(font, spacing)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += lh
    return y


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int], outline=None, width=1, radius=18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_tag(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int]) -> int:
    text = clean_text(text)
    w = text_width(text, FONT_TINY) + 24
    rounded(draw, (x, y, x + w, y + 30), fill=(245, 248, 255), outline=color, width=1, radius=12)
    draw.text((x + 12, y + 5), text, font=FONT_TINY, fill=color)
    return x + w + 8


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return None
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, PANEL)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def grouped_steps(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("question_id", "")].append(row)
    for qid in groups:
        groups[qid].sort(key=lambda row: int(float(row.get("chain_step") or 0)))
    return groups


def draw_header(draw: ImageDraw.ImageDraw, question: dict[str, str]) -> None:
    qid = clean_text(question.get("question_id"))
    qtype = clean_text(question.get("question_type"))
    doc_id = clean_text(question.get("doc_id"))
    draw.text((MARGIN, 34), "G4 Evidence Chain Card", font=FONT_TITLE, fill=INK)
    x = CARD_W - MARGIN - 520
    x = draw_tag(draw, x, 48, qid, BLUE)
    x = draw_tag(draw, x, 48, qtype, GREEN)
    draw_tag(draw, x, 48, doc_id, PURPLE)
    draw.line((MARGIN, 112, CARD_W - MARGIN, 112), fill=FAINT, width=2)


def draw_question_answer(draw: ImageDraw.ImageDraw, question: dict[str, str], steps: list[dict[str, str]]) -> None:
    y = 146
    h = CARD_H - y - 104
    rounded(draw, (MARGIN, y, MARGIN + LEFT_W, y + h), PANEL, outline=FAINT, radius=20)
    draw.text((MARGIN + 28, y + 26), "\u95ee\u9898", font=FONT_H2, fill=BLUE)
    y2 = draw_wrapped(draw, (MARGIN + 28, y + 66), question.get("question", ""), FONT_BODY, INK, LEFT_W - 56, 5)
    draw.text((MARGIN + 28, y2 + 22), "\u7b54\u6848", font=FONT_H2, fill=GREEN)
    draw_wrapped(draw, (MARGIN + 28, y2 + 62), question.get("answer", ""), FONT_BODY, INK, LEFT_W - 56, 8)

    pages = sorted({clean_text(step.get("page")) for step in steps if clean_text(step.get("page"))}, key=lambda x: int(float(x)) if x.isdigit() else 9999)
    meta_y = y + h - 112
    draw.line((MARGIN + 28, meta_y - 20, MARGIN + LEFT_W - 28, meta_y - 20), fill=FAINT, width=2)
    draw.text((MARGIN + 28, meta_y), "\u6765\u6e90\u9875", font=FONT_H2, fill=PURPLE)
    draw_wrapped(draw, (MARGIN + 28, meta_y + 40), ", ".join(pages[:12]) or "-", FONT_BODY, MUTED, LEFT_W - 56, 2)


def step_reason(step: dict[str, str]) -> str:
    visual_evidence = extract_visual_field(step.get("visual_caption"), "qa_evidence")
    if not visual_evidence:
        visual_evidence = extract_visual_field(step.get("visual_summary"), "qa_evidence")
    if visual_evidence:
        return preview(visual_evidence, 170)
    content = clean_text(step.get("content_preview"))
    if content:
        return preview(strip_visual_json(content), 170)
    reason = clean_text(step.get("reason"))
    if reason:
        return preview(reason, 150)
    visual = clean_text(step.get("visual_summary"))
    if visual:
        return preview(strip_visual_json(visual.replace("Visual summary:", "")), 170)
    return ""


def extract_visual_field(text: Any, field: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return ""
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ""
    value = data.get(field, "")
    if isinstance(value, list):
        return "\uff1b".join(clean_text(item) for item in value if clean_text(item))
    return clean_text(value)


def strip_visual_json(text: Any) -> str:
    text = clean_text(text)
    text = re.sub(r"Visual summary:.*?(Document caption/context:|$)", r"\1", text, flags=re.S)
    text = re.sub(r"```json.*?```", "", text, flags=re.S)
    return clean_text(text)


def draw_step(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, idx: int, step: dict[str, str]) -> int:
    h = 128
    rounded(draw, (x, y, x + w, y + h), (252, 254, 255), outline=FAINT, radius=18)
    role = ROLE_LABELS.get(clean_text(step.get("role")), clean_text(step.get("role")) or "\u8bc1\u636e")
    node_type = clean_text(step.get("node_type")) or "node"
    color = TYPE_COLORS.get(node_type, BLUE)
    rounded(draw, (x + 22, y + 22, x + 62, y + 62), color, radius=13)
    draw.text((x + 36 - text_width(str(idx), FONT_H2) // 2, y + 27), str(idx), font=FONT_H2, fill=(255, 255, 255))
    draw.text((x + 78, y + 21), role, font=FONT_H2, fill=INK)
    meta = f"{node_type} | p.{clean_text(step.get('page'))} | {clean_text(step.get('node_id'))}"
    draw.text((x + 78, y + 55), meta, font=FONT_SMALL, fill=MUTED)
    relation = clean_text(step.get("relation"))
    if relation:
        draw.text((x + 78, y + 80), relation, font=FONT_SMALL, fill=color)
    draw_wrapped(draw, (x + 78, y + 104), step_reason(step), FONT_TINY, INK, w - 116, 2, spacing=3)
    return y + h + 14


def draw_visual_strip(card: Image.Image, draw: ImageDraw.ImageDraw, steps: list[dict[str, str]]) -> None:
    y = 146
    rounded(draw, (RIGHT_X, y, CARD_W - MARGIN, CARD_H - 104), PANEL, outline=FAINT, radius=20)
    draw.text((RIGHT_X + 24, y + 24), "\u89c6\u89c9\u8bc1\u636e\u88c1\u526a", font=FONT_H2, fill=PURPLE)
    visual_steps = [step for step in steps if clean_text(step.get("crop_image_path"))]
    if not visual_steps:
        draw_wrapped(draw, (RIGHT_X + 24, y + 74), "\u5f53\u524d\u8bc1\u636e\u94fe\u6ca1\u6709\u53ef\u7528\u88c1\u526a\u56fe\u3002", FONT_BODY, MUTED, RIGHT_W - 48, 3)
        return

    slot_w = RIGHT_W - 48
    slot_h = 150
    cursor = y + 76
    for step in visual_steps[:4]:
        if cursor + slot_h + 66 > CARD_H - 128:
            break
        path = resolve_path(step.get("crop_image_path", ""))
        thumb = fit_image(path, (slot_w, slot_h))
        rounded(draw, (RIGHT_X + 24, cursor, RIGHT_X + 24 + slot_w, cursor + slot_h), (250, 252, 255), outline=FAINT, radius=14)
        if thumb:
            mask = Image.new("L", thumb.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rounded_rectangle((0, 0, thumb.width, thumb.height), radius=14, fill=255)
            card.paste(thumb, (RIGHT_X + 24, cursor), mask)
        caption = f"{ROLE_LABELS.get(clean_text(step.get('role')), clean_text(step.get('role')))} | {clean_text(step.get('node_type'))} | p.{clean_text(step.get('page'))}"
        draw_wrapped(draw, (RIGHT_X + 24, cursor + slot_h + 10), caption, FONT_TINY, MUTED, slot_w, 2, spacing=3)
        cursor += slot_h + 60


def draw_footer(draw: ImageDraw.ImageDraw, question: dict[str, str], steps: list[dict[str, str]]) -> None:
    source = f"{clean_text(question.get('doc_id'))} | Generated from local G4 evidence chain"
    draw.line((MARGIN, CARD_H - 72, CARD_W - MARGIN, CARD_H - 72), fill=FAINT, width=2)
    draw.text((MARGIN, CARD_H - 50), source, font=FONT_SMALL, fill=MUTED)


def build_card(question: dict[str, str], steps: list[dict[str, str]], output: Path, max_steps: int) -> None:
    card = Image.new("RGB", (CARD_W, CARD_H), BG)
    draw = ImageDraw.Draw(card)
    draw_header(draw, question)
    draw_question_answer(draw, question, steps)
    y = 146
    rounded(draw, (CHAIN_X, y, CHAIN_X + CHAIN_W, CARD_H - 104), PANEL, outline=FAINT, radius=20)
    draw.text((CHAIN_X + 24, y + 24), "\u8bc1\u636e\u94fe", font=FONT_H2, fill=INK)
    y += 76
    for idx, step in enumerate(steps[:max_steps], start=1):
        if y + 140 > CARD_H - 124:
            break
        y = draw_step(draw, CHAIN_X + 24, y, CHAIN_W - 48, idx, step)
    draw_visual_strip(card, draw, steps)
    draw_footer(draw, question, steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    card.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render G4 evidence chains as deterministic PNG evidence cards.")
    parser.add_argument("--questions", default="data/questions.csv")
    parser.add_argument("--chain-steps", default="outputs/evidence_chains/chain_steps.csv")
    parser.add_argument("--output-dir", default="outputs/evidence_cards")
    parser.add_argument("--question-id", default="", help="Optional single question id.")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    ensure_project_dirs()
    questions = [row for row in read_csv(args.questions) if clean_text(row.get("question"))]
    if args.question_id:
        questions = [row for row in questions if row.get("question_id") == args.question_id]
    steps_by_qid = grouped_steps(read_csv(args.chain_steps))
    output_dir = resolve_path(args.output_dir)

    manifest: list[dict[str, Any]] = []
    for question in questions:
        qid = clean_text(question.get("question_id"))
        steps = steps_by_qid.get(qid, [])
        if not qid or not steps:
            continue
        output = output_dir / f"{qid}_evidence_card.png"
        build_card(question, steps, output, max(1, args.max_steps))
        manifest.append(
            {
                "question_id": qid,
                "doc_id": question.get("doc_id", ""),
                "question_type": question.get("question_type", ""),
                "card_path": str(output.relative_to(resolve_path("."))),
                "num_steps": len(steps),
            }
        )

    write_csv(output_dir / "cards_manifest.csv", manifest, ["question_id", "doc_id", "question_type", "card_path", "num_steps"])
    print(f"Wrote {len(manifest)} evidence cards to {output_dir}")


if __name__ == "__main__":
    main()
