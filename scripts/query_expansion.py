from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from pipeline_common import clean_text, resolve_path


PRODUCT_HINTS = {
    "air_conditioner": "air conditioner AC indoor unit filter grille air inlet outlet operation lamp remote control clean safety",
    "air_purifier": "air purifier filter plastic packaging top cover clean replacement indicator safety",
    "camera": "camera shutter button battery CF card lens sensor power switch authorized repair disassembly",
    "coffee_machine": "coffee machine espresso lungo Nespresso energy saving mode factory reset water volume descaling cleaning",
    "boat": "boat anchor light bimini top livewell swim platform engine steering cooling system fuse throttle cable",
    "dishwasher": "dishwasher installation water supply drain electrical connection detergent salt child safety",
    "drill": "electric drill battery indicator charging overheat overcool delay bit torque speed",
    "earphones": "earphones earbuds headphones pairing bluetooth charging indicator reset audio",
    "ereader": "eReader ebook reader micro SD card reset screen battery charging troubleshooting",
    "exercise_bike": "exercise bike fitness bike assembly seat pedal resistance console heart rate belt",
    "fax": "fax fax machine product safety guide power cord warning maintenance telephone line",
    "fitness_tracker": "fitness tracker activity tracker strap band charging cable package contents size wrist",
    "generator": "generator engine fuel oil spark plug choke starter switch overload safety maintenance",
    "grill": "grill barbecue burner gas propane ignition cleaning safety cooking grate",
    "hair_dryer": "hair dryer blow dryer cool shot heat airflow filter nozzle cleaning indicator",
    "jetski": "jetski jet ski watercraft engine start lanyard safety warning riding practice boarding throttle",
    "keyboard": "keyboard function keyboard key switch warranty cable backlight cleaning",
    "landline": "landline telephone handset base station cordless phone battery voicemail caller ID",
    "loudspeaker": "loudspeaker speaker pairing Bluetooth buttons volume charging indicator",
    "lawn_mower": "lawn mower blade start handle safety oil grass bag maintenance",
    "microwave": "microwave oven turntable installation vent filter control panel safety",
    "motherboard": "motherboard BIOS SATA ODD USB OS installation DDR4 DIMM CPU memory socket driver",
    "mouse": "bluetooth laser mouse pairing battery button DPI receiver reset",
    "oven": "oven installation cabinet electrical connection heat resistant door damage authorized service",
    "pressure_cooker": "pressure cooker air fryer lid valve pressure release basket cooking safety clean",
    "refrigerator": "refrigerator fridge freezer temperature door seal water filter defrost safety",
    "gps_navigation": "GPS navigator navigation NAV route guidance screen reset pairing",
    "robot_vacuum": "robot vacuum Roomba Home Base CLEAN button full bin sensors charging contacts troubleshooting indicator",
    "snowmobile": "snowmobile engine start throttle brake safety warning maintenance track",
    "steam_cleaner": "steam cleaner steam mop water tank nozzle brush descaling safety",
    "toothbrush": "electric toothbrush brush head charging mode timer safety cleaning",
    "tv_radio": "television TV radio DVD antenna caption channel power cord safety",
    "vacuum": "vacuum cleaner robot vacuum home base filter bin sensor charging contacts clean",
    "vr_headset": "VR headset processor unit HDMI USB AUX DC IN status indicator ventilation cable",
    "water_pump": "water pump engine fuel switch priming plug drain plug throttle handle inlet outlet hose filter",
}

ROUTE_HINTS = {
    "service": "after-sales policy customer service proof order number receipt warranty return refund exchange shipping invoice complaint",
    "service_with_product": "product manual troubleshooting operation parts installation safety specification diagram",
    "manual": "manual instructions operation setup maintenance safety warning specification parts",
    "manual_visual": "manual diagram figure picture button indicator part location table step visual evidence",
    "visual": "figure diagram table image caption OCR label part position visual evidence",
}

TYPE_HINTS = {
    "退换货": "return refund exchange eligibility proof shipping fee RMA",
    "退款": "return refund exchange eligibility proof shipping fee RMA",
    "发票": "invoice receipt title tax number issue correction delivery time",
    "物流": "shipping delivery pickup tracking receipt freight rural area",
    "保修": "warranty repair fault responsibility proof RMA service scope",
    "维修": "warranty repair fault responsibility proof RMA service scope",
    "安装": "installation setup authorized service parts fee damage proof",
    "故障": "troubleshooting fault error indicator reset repair safety",
    "部件": "parts component diagram label position interface connector",
    "安全": "safety warning caution danger personal injury electric shock",
}

SERVICE_TERM_HINTS = {
    "售后": "after-sales customer service handling procedure responsibility proof",
    "换货": "exchange eligibility proof RMA return shipping",
    "退款": "refund return eligibility proof processing",
    "退货": "return refund eligibility proof shipping",
    "运费": "shipping fee freight return postage responsibility",
    "发票": "invoice receipt title tax number issue time",
    "保修": "warranty repair proof service period fault",
    "维修": "repair warranty fault responsibility service center",
    "物流": "shipping delivery pickup tracking receipt",
    "投诉": "complaint evidence photos order number escalation",
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


def route_hints(route: dict[str, str]) -> list[str]:
    hints: list[str] = []
    route_name = clean_text(route.get("route"))
    product = clean_text(route.get("product"))
    question_type = clean_text(route.get("question_type"))
    service_terms = [clean_text(term) for term in re.split(r"[;,；，]\s*", route.get("service_terms", "")) if clean_text(term)]
    matched_aliases = [clean_text(term) for term in re.split(r"[;,；，]\s*", route.get("matched_aliases", "")) if clean_text(term)]

    if route_name in ROUTE_HINTS:
        hints.append(f"route={route_name}; intent aliases={ROUTE_HINTS[route_name]}")
    if product and product in PRODUCT_HINTS:
        hints.append(f"product={product}; product aliases={PRODUCT_HINTS[product]}")
    elif product:
        hints.append(f"product={product}")
    if matched_aliases:
        hints.append("matched product aliases=" + " ".join(matched_aliases[:10]))
    for key, hint in TYPE_HINTS.items():
        if key in question_type:
            hints.append(f"question_type_hint={hint}")
    include_service_terms = route_name == "service" or any(term not in {"售后"} for term in service_terms)
    for term in service_terms if include_service_terms else []:
        hint = SERVICE_TERM_HINTS.get(term)
        if hint:
            hints.append(f"service_term={term}; {hint}")
    return list(dict.fromkeys(hints))


def expand_question(question: dict[str, Any], route: dict[str, str] | None) -> dict[str, Any]:
    if not route:
        return question
    hints = route_hints(route)
    if not hints:
        return question
    expanded = dict(question)
    original = clean_text(question.get("question"))
    expanded["question"] = f"{original}\nSearch context: " + " | ".join(hints[:8])
    expanded["original_question"] = original
    expanded["route_product"] = clean_text(route.get("product"))
    expanded["route_name"] = clean_text(route.get("route"))
    return expanded
