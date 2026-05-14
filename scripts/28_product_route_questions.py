from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pipeline_common import clean_text, read_csv, resolve_path, write_csv, write_jsonl


PRODUCT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("air_conditioner", ("空调", "air conditioner", "air conditioning", "ac remote", "remote controller")),
    ("drill", ("电钻", "drill", "cordless drill", "dcb101", "keyless chuck")),
    ("air_purifier", ("空气净化器", "air purifier", "filter", "plasma filter")),
    ("hair_dryer", ("吹风机", "hair dryer", "blower")),
    ("dishwasher", ("洗碗机", "dishwasher", "dish washer")),
    ("exercise_bike", ("健身单车", "exercise bike", "fitness bike", "stationary bike")),
    ("steam_cleaner", ("蒸汽清洁机", "steam cleaner", "steam mop")),
    ("kids_motorcycle", ("儿童电动摩托车", "kids electric motorcycle", "toy motorcycle")),
    ("refrigerator", ("冰箱", "refrigerator", "fridge")),
    ("jetski", ("摩托艇", "jetski", "jet ski", "waverunner", "watercraft")),
    ("ergonomic_chair", ("人体工学椅", "ergonomic chair", "office chair", "armrest")),
    ("keyboard", ("功能键盘", "keyboard", "keycap", "switch", "axis body")),
    ("oven", ("烤箱", "oven")),
    ("camera", ("相机", "camera", "photo", "image", "lens", "shutter", "cf card", "af mode")),
    ("thermostat", ("可编程温控器", "thermostat", "programmable thermostat", "temperature schedule")),
    ("fitness_tracker", ("健身追踪器", "fitness tracker", "activity tracker", "heart rate", "skin temperature")),
    ("water_pump", ("水泵", "water pump", "pump")),
    ("generator", ("发电机", "generator")),
    ("vr_headset", ("vr头显", "vr headset", "virtual reality headset")),
    ("mouse", ("蓝牙激光鼠标", "bluetooth laser mouse", "laser mouse", "mouse")),
    ("earphones", ("耳机", "earphones", "earbuds", "headphones")),
    ("ereader", ("电子书", "阅读器", "ereader", "e-reader", "ebook reader", "music mode")),
    ("fax", ("传真机", "fax", "fax machine")),
    ("grill", ("烤架", "grill", "barbecue", "bbq")),
    ("landline", ("座机", "landline", "handset", "base station", "phone settings", "cordless")),
    ("lawn_mower", ("割草机", "lawn mower", "mower", "deck lift", "blade-control")),
    ("microwave", ("微波炉", "over-the-range microwave", "microwave")),
    ("motherboard", ("主板", "motherboard", "bios", "pci express", "sata", "cpu", "raid")),
    ("pressure_cooker", ("压力锅", "空气炸锅", "multi-use pressure cooker", "pressure cooker", "air fryer")),
    ("vacuum", ("扫地机", "吸尘器", "vacuum cleaner", "robot vacuum", "home base", "extractors")),
    ("snowmobile", ("雪地摩托", "snowmobile")),
    ("tv_radio", ("电视", "收音", "television", "tv", "radio", "dvd player", "outdoor antenna")),
    ("toothbrush", ("电动牙刷", "electric toothbrush", "toothbrush")),
)

EXTRA_PRODUCT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("coffee_machine", ("coffee machine", "coffee maker", "nespresso", "espresso", "lungo")),
    ("boat", ("boat", "anchor light", "sail", "stern light", "navigation light")),
    ("loudspeaker", ("loudspeaker", "speaker", "wireless speaker")),
    ("gps_navigation", ("gps", "navigator", "navigation", "nav", "route guidance")),
    ("robot_vacuum", ("vacuum", "robot vacuum", "roomba", "home base", "full bin sensor", "charging contacts")),
)

SERVICE_TERMS = (
    "售后",
    "退货",
    "换货",
    "退款",
    "发票",
    "投诉",
    "保修",
    "维修",
    "运费",
    "包装破损",
    "假货",
    "临期",
    "少发",
    "错发",
    "return",
    "refund",
    "exchange",
    "warranty",
    "repair",
    "invoice",
    "complaint",
)


def alias_in_text(alias: str, text: str) -> bool:
    alias = clean_text(alias).casefold()
    if not alias:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in alias):
        return alias in text
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None


def route_question(row: dict[str, str]) -> dict[str, Any]:
    question_id = clean_text(row.get("question_id"))
    question = clean_text(row.get("question"))
    qtype = clean_text(row.get("question_type"))
    blob = f"{question} {qtype}".casefold()
    service_hits = [term for term in SERVICE_TERMS if term.casefold() in blob]

    scores: dict[str, float] = {}
    matched_aliases: dict[str, list[str]] = {}
    for product, aliases in (*PRODUCT_ALIASES, *EXTRA_PRODUCT_ALIASES):
        hits = [alias for alias in aliases if alias_in_text(alias, blob)]
        if not hits:
            continue
        score = sum(2.0 if any("\u4e00" <= ch <= "\u9fff" for ch in alias) else 1.0 for alias in hits)
        scores[product] = score
        matched_aliases[product] = hits[:6]

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    product = ranked[0][0] if ranked else ""
    confidence = 0.0
    if ranked:
        top_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = min(1.0, 0.35 + 0.18 * top_score + 0.12 * max(0.0, top_score - second_score))
    route = "service" if service_hits and not product else "manual"
    if service_hits and product:
        route = "service_with_product"
    return {
        "question_id": question_id,
        "submission_id": re.search(r"(\d+)$", question_id).group(1) if re.search(r"(\d+)$", question_id) else question_id,
        "question": question,
        "question_type": qtype,
        "route": route,
        "product": product,
        "confidence": round(confidence, 4),
        "matched_aliases": matched_aliases.get(product, []),
        "service_terms": service_hits[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Route questions to product/service domains.")
    parser.add_argument("--questions", default="outputs/after_sales_kb/questions.csv")
    parser.add_argument("--output-jsonl", default="outputs/after_sales_kb/question_routes.jsonl")
    parser.add_argument("--output-csv", default="outputs/after_sales_kb/question_routes.csv")
    args = parser.parse_args()

    rows = [route_question(row) for row in read_csv(args.questions) if clean_text(row.get("question"))]
    write_jsonl(args.output_jsonl, rows)
    csv_rows = [
        {
            **row,
            "matched_aliases": ";".join(row["matched_aliases"]),
            "service_terms": ";".join(row["service_terms"]),
        }
        for row in rows
    ]
    write_csv(
        args.output_csv,
        csv_rows,
        [
            "question_id",
            "submission_id",
            "question",
            "question_type",
            "route",
            "product",
            "confidence",
            "matched_aliases",
            "service_terms",
        ],
    )
    print(f"Wrote {len(rows)} routes to {resolve_path(args.output_jsonl)}")


if __name__ == "__main__":
    main()
