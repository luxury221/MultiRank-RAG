from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from pipeline_common import clean_text, resolve_path


PRODUCT_HINTS = {
    "coffee_machine": "coffee machine espresso lungo Nespresso energy saving mode factory reset water volume descaling cleaning",
    "boat": "boat anchor light bimini top livewell swim platform engine steering cooling system fuse throttle cable",
    "loudspeaker": "loudspeaker speaker pairing Bluetooth buttons volume charging indicator",
    "gps_navigation": "GPS navigator navigation NAV route guidance screen reset pairing",
    "robot_vacuum": "robot vacuum Roomba Home Base CLEAN button full bin sensors charging contacts troubleshooting indicator",
}


def submission_id(question_id: str) -> str:
    match = re.search(r"(\d+)$", clean_text(question_id))
    return match.group(1) if match else clean_text(question_id)


def load_routes(path: str | Path) -> dict[str, dict[str, str]]:
    file_path = resolve_path(path)
    if not file_path.exists():
        return {}
    routes: dict[str, dict[str, str]] = {}
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = clean_text(row.get("submission_id")) or submission_id(row.get("question_id", ""))
            if sid:
                routes[sid] = dict(row)
    return routes


def expand_question(question: dict[str, Any], route: dict[str, str] | None) -> dict[str, Any]:
    if not route:
        return question
    product = clean_text(route.get("product"))
    hint = PRODUCT_HINTS.get(product)
    if not hint:
        return question
    expanded = dict(question)
    original = clean_text(question.get("question"))
    expanded["question"] = f"{original}\nSearch context: product={product}; aliases={hint}"
    expanded["original_question"] = original
    expanded["route_product"] = product
    return expanded
