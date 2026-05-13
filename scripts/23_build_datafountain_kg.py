from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from pipeline_common import clean_text, normalize_doc_id, preview, read_jsonl, write_jsonl
from rerank_lib import AFTER_SALES_INTENT_TERMS, DATAFOUNTAIN_PRODUCT_ALIASES, VISUAL_NODE_TYPES


DEFAULT_NODES = "outputs/after_sales_kb/nodes.jsonl"
DEFAULT_KG_DIR = "outputs/kg"
DEFAULT_VISUAL_DIR = "outputs/visual_index"
DEFAULT_TEXT_DIR = "outputs/text_index"

PART_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("盐箱", ("盐箱", "salt tank", "salt reservoir")),
    ("喷淋臂", ("喷淋臂", "spray arm")),
    ("滤网", ("滤网", "过滤网", "filter", "strainer")),
    ("电池", ("电池", "battery", "batteries")),
    ("脚轮", ("脚轮", "wheel", "caster")),
    ("座椅", ("座椅", "seat")),
    ("引擎盖", ("引擎盖", "hood")),
    ("盖子", ("盖子", "cap", "lid", "cover")),
    ("开关", ("开关", "switch")),
    ("操纵杆", ("操纵杆", "lever")),
    ("选择器", ("选择器", "selector", "qsts")),
    ("燃油表", ("燃油表", "fuel meter")),
    ("小时表", ("小时表", "hour meter")),
    ("基站", ("基站", "base station")),
    ("手柄/听筒", ("听筒", "手柄", "handset")),
    ("指示灯", ("指示灯", "led", "indicator light")),
    ("滚轮", ("滚轮", "wheel", "roller")),
    ("轴体", ("轴体", "switch stem", "switch")),
    ("键帽", ("键帽", "keycap")),
    ("腕托", ("腕托", "wrist rest")),
    ("门", ("门", "door")),
    ("灯", ("灯", "light", "lamp")),
    ("加热元件", ("加热元件", "heating element")),
    ("传感器", ("传感器", "sensor")),
    ("接口", ("接口", "connector", "port", "interface")),
    ("跳线", ("跳线", "jumper")),
    ("CPU", ("cpu", "processor")),
    ("BIOS", ("bios",)),
    ("阀门", ("阀门", "valve")),
    ("密封圈", ("密封圈", "sealing ring", "seal ring")),
    ("浮子阀", ("浮子阀", "float valve")),
    ("防堵罩", ("防堵罩", "anti-block shield")),
    ("集水盒", ("集水盒", "condensation collector")),
    ("尘盒", ("尘盒", "bin", "dust bin")),
    ("刷子/滚刷", ("刷子", "滚刷", "brush", "extractor")),
    ("充电座", ("充电座", "home base", "charging base")),
    ("刹车", ("刹车", "brake")),
    ("油门线", ("油门线", "throttle cable")),
    ("火花塞", ("火花塞", "spark plug")),
    ("天线", ("天线", "antenna")),
)

ACTION_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("安装", ("安装", "装上", "install", "mount", "assemble")),
    ("拆卸", ("拆卸", "拆除", "取下", "remove", "detach", "disassemble")),
    ("更换", ("更换", "replace", "change")),
    ("清洁", ("清洁", "清洗", "clean", "wash")),
    ("连接", ("连接", "connect", "pair", "配对")),
    ("设置", ("设置", "set", "setup", "set up", "configure")),
    ("调节", ("调节", "调整", "adjust")),
    ("打开", ("打开", "open")),
    ("关闭", ("关闭", "close")),
    ("添加", ("添加", "加入", "add", "fill")),
    ("使用", ("使用", "operate", "use")),
    ("检查", ("检查", "inspect", "check")),
    ("启动", ("启动", "start")),
    ("重置", ("重置", "reset")),
    ("充电", ("充电", "charge")),
    ("更新", ("更新", "update")),
    ("创建", ("创建", "create")),
    ("排查", ("排查", "troubleshoot", "troubleshooting")),
)

FAULT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("不工作", ("不工作", "无法工作", "won't work", "does not work", "not working")),
    ("无响应", ("无响应", "没有反应", "no response")),
    ("报错", ("报错", "error")),
    ("漏水", ("漏水", "leak", "leaking")),
    ("无法连接", ("无法连接", "can't connect", "cannot connect")),
    ("异常噪音", ("噪音", "noise")),
    ("指示灯异常", ("指示灯异常", "led indicator", "indicator behavior")),
)

INTENT_LABELS = {
    "return_refund": "退换货退款",
    "invoice": "发票开票",
    "shipping_damage": "物流包装破损",
    "warranty_repair": "保修维修",
    "troubleshooting": "故障排查",
    "usage_operation": "安装使用",
    "spec_parts": "规格配件",
    "safety": "安全警告",
}


def term_in_text(term: str, text: str) -> bool:
    term = clean_text(term).casefold()
    if not term:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in term):
        return term in text
    import re

    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def entity_id(entity_type: str, name: str) -> str:
    return f"{entity_type}:{normalize_doc_id(name).lower()}"


def node_text(node: dict[str, Any]) -> str:
    return clean_text(
        " ".join(
            [
                node.get("doc_id", ""),
                node.get("section", ""),
                node.get("source_ref", ""),
                node.get("content", ""),
                node.get("previous_chunk_preview", ""),
                node.get("next_chunk_preview", ""),
                node.get("visual_title", ""),
                node.get("key_objects", ""),
                node.get("ocr_text", ""),
                node.get("data_or_trends", ""),
                node.get("qa_evidence", ""),
                node.get("visual_caption", ""),
                node.get("visual_caption_raw", ""),
                node.get("visual_summary", ""),
                node.get("limitations", ""),
            ]
        )
    )


def match_catalog(text: str, catalog: tuple[tuple[str, tuple[str, ...]], ...]) -> list[tuple[str, tuple[str, ...]]]:
    blob = clean_text(text).casefold()
    matches: list[tuple[str, tuple[str, ...]]] = []
    for name, aliases in catalog:
        if term_in_text(name, blob) or any(term_in_text(alias, blob) for alias in aliases):
            matches.append((name, aliases))
    return matches


def product_aliases_for_name(name: str) -> tuple[str, ...]:
    blob = clean_text(name).casefold()
    for canonical, aliases in DATAFOUNTAIN_PRODUCT_ALIASES:
        if term_in_text(canonical, blob) or any(term_in_text(alias, blob) for alias in aliases):
            return aliases
    return (name,)


def products_for_node(node: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    text = node_text(node)
    matches = match_catalog(text, DATAFOUNTAIN_PRODUCT_ALIASES)
    doc_id = clean_text(node.get("doc_id"))
    if doc_id and doc_id != "售后通用政策" and doc_id != "汇总英文手册":
        product = doc_id.removesuffix("手册")
        aliases = product_aliases_for_name(product)
        if not any(item[0] == product for item in matches):
            matches.insert(0, (product, aliases))
    return matches


def split_intents(value: Any) -> list[str]:
    return [item.strip() for item in clean_text(value).split(";") if item.strip()]


def add_entity(
    entities: dict[str, dict[str, Any]],
    entity_type: str,
    name: str,
    aliases: tuple[str, ...] = (),
    node: dict[str, Any] | None = None,
) -> str:
    eid = entity_id(entity_type, name)
    item = entities.setdefault(
        eid,
        {
            "entity_id": eid,
            "entity_type": entity_type,
            "name": name,
            "aliases": set(),
            "doc_ids": set(),
            "node_ids": set(),
            "evidence_count": 0,
        },
    )
    item["aliases"].update(clean_text(alias) for alias in aliases if clean_text(alias))
    if node:
        item["evidence_count"] += 1
        if clean_text(node.get("doc_id")):
            item["doc_ids"].add(clean_text(node.get("doc_id")))
        if clean_text(node.get("node_id")):
            item["node_ids"].add(clean_text(node.get("node_id")))
    return eid


def add_relation(
    relations: dict[tuple[str, str, str, str], dict[str, Any]],
    source_id: str,
    target_id: str,
    relation_type: str,
    node: dict[str, Any],
    weight: float = 1.0,
) -> None:
    evidence_node_id = clean_text(node.get("node_id"))
    key = (source_id, target_id, relation_type, evidence_node_id)
    item = relations.setdefault(
        key,
        {
            "relation_id": f"rel:{len(relations) + 1}",
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "weight": 0.0,
            "evidence_node_id": evidence_node_id,
            "doc_id": clean_text(node.get("doc_id")),
            "source_ref": preview(node.get("source_ref", ""), 260),
        },
    )
    item["weight"] = round(float(item["weight"]) + weight, 4)


def serializable_entities(entities: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in entities.values():
        row = dict(item)
        row["aliases"] = sorted(row["aliases"])
        row["doc_ids"] = sorted(row["doc_ids"])
        row["node_ids"] = sorted(row["node_ids"])
        rows.append(row)
    return sorted(rows, key=lambda row: (row["entity_type"], row["name"]))


def build_visual_row(node: dict[str, Any], related_entities: list[str]) -> dict[str, Any]:
    return {
        "image_id": clean_text(node.get("image_id")) or clean_text(node.get("node_id")),
        "node_id": clean_text(node.get("node_id")),
        "doc_id": clean_text(node.get("doc_id")),
        "path": clean_text(node.get("crop_image_path")) or clean_text(node.get("image_path")) or clean_text(node.get("page_image_path")),
        "page_image_path": clean_text(node.get("page_image_path")),
        "visual_title": clean_text(node.get("visual_title")),
        "visual_type": clean_text(node.get("visual_type")),
        "key_objects": clean_text(node.get("key_objects")),
        "ocr_text": clean_text(node.get("ocr_text")),
        "qa_evidence": clean_text(node.get("qa_evidence")),
        "visual_caption": clean_text(node.get("visual_caption")),
        "source_ref": preview(node.get("source_ref", ""), 260),
        "related_entities": related_entities,
    }


def build_text_row(node: dict[str, Any], related_entities: list[str]) -> dict[str, Any]:
    return {
        "node_id": clean_text(node.get("node_id")),
        "doc_id": clean_text(node.get("doc_id")),
        "page": node.get("page", ""),
        "node_type": clean_text(node.get("node_type")),
        "structure_type": clean_text(node.get("structure_type")),
        "product_category": clean_text(node.get("product_category")),
        "service_intents": clean_text(node.get("service_intents")),
        "section": clean_text(node.get("section")),
        "source_ref": preview(node.get("source_ref", ""), 260),
        "content": clean_text(node.get("content")),
        "searchable_text": clean_text(node.get("searchable_text")) or node_text(node),
        "related_entities": related_entities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight GraphRAG-style KG and visual index for DataFountain.")
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--kg-dir", default=DEFAULT_KG_DIR)
    parser.add_argument("--visual-dir", default=DEFAULT_VISUAL_DIR)
    parser.add_argument("--text-dir", default=DEFAULT_TEXT_DIR)
    parser.add_argument("--top-items", type=int, default=30)
    args = parser.parse_args()

    nodes = read_jsonl(args.nodes)
    entities: dict[str, dict[str, Any]] = {}
    relations: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    product_profiles: dict[str, dict[str, Any]] = {}
    visual_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []

    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if not node_id:
            continue
        text = node_text(node)
        products = products_for_node(node)
        parts = match_catalog(text, PART_TERMS)
        actions = match_catalog(text, ACTION_TERMS)
        faults = match_catalog(text, FAULT_TERMS)
        node_entity_ids: list[str] = []

        product_ids = [add_entity(entities, "product", name, aliases, node) for name, aliases in products]
        part_ids = [add_entity(entities, "part", name, aliases, node) for name, aliases in parts]
        action_ids = [add_entity(entities, "action", name, aliases, node) for name, aliases in actions]
        fault_ids = [add_entity(entities, "fault", name, aliases, node) for name, aliases in faults]
        node_entity_ids.extend(product_ids + part_ids + action_ids + fault_ids)

        policy_ids: list[str] = []
        for intent in split_intents(node.get("service_intents")):
            if intent in AFTER_SALES_INTENT_TERMS:
                policy_name = INTENT_LABELS.get(intent, intent)
                policy_ids.append(add_entity(entities, "policy", policy_name, AFTER_SALES_INTENT_TERMS[intent], node))
        node_entity_ids.extend(policy_ids)

        image_id = ""
        if clean_text(node.get("node_type")) in VISUAL_NODE_TYPES and clean_text(node.get("crop_image_path")):
            image_id = add_entity(
                entities,
                "image",
                clean_text(node.get("image_id")) or node_id,
                (clean_text(node.get("image_id")) or node_id,),
                node,
            )
            node_entity_ids.append(image_id)
            visual_rows.append(build_visual_row(node, sorted(set(node_entity_ids))))
        else:
            text_rows.append(build_text_row(node, sorted(set(node_entity_ids))))

        for product_id in product_ids:
            profile = product_profiles.setdefault(
                product_id,
                {
                    "product_id": product_id,
                    "product": entities[product_id]["name"],
                    "aliases": set(entities[product_id]["aliases"]),
                    "doc_ids": set(),
                    "node_ids": set(),
                    "parts": Counter(),
                    "actions": Counter(),
                    "faults": Counter(),
                    "policies": Counter(),
                    "image_ids": set(),
                },
            )
            if clean_text(node.get("doc_id")):
                profile["doc_ids"].add(clean_text(node.get("doc_id")))
            profile["node_ids"].add(node_id)
            for part_name, _ in parts:
                profile["parts"][part_name] += 1
            for action_name, _ in actions:
                profile["actions"][action_name] += 1
            for fault_name, _ in faults:
                profile["faults"][fault_name] += 1
            for policy_id in policy_ids:
                profile["policies"][entities[policy_id]["name"]] += 1
            if clean_text(node.get("image_id")):
                profile["image_ids"].add(clean_text(node.get("image_id")))

        for product_id in product_ids:
            for part_id in part_ids:
                add_relation(relations, product_id, part_id, "product_has_part", node, 1.0)
            for action_id in action_ids:
                add_relation(relations, product_id, action_id, "product_supports_action", node, 0.85)
            for fault_id in fault_ids:
                add_relation(relations, product_id, fault_id, "product_has_fault", node, 0.75)
            for policy_id in policy_ids:
                add_relation(relations, policy_id, product_id, "policy_applies_to_product", node, 0.5)
            if image_id:
                add_relation(relations, product_id, image_id, "product_has_image", node, 0.8)
        for action_id in action_ids:
            for part_id in part_ids:
                add_relation(relations, action_id, part_id, "action_targets_part", node, 0.9)
            for fault_id in fault_ids:
                add_relation(relations, fault_id, action_id, "fault_solved_by_action", node, 0.8)
            if image_id:
                add_relation(relations, image_id, action_id, "image_illustrates_action", node, 0.95)
        for part_id in part_ids:
            if image_id:
                add_relation(relations, image_id, part_id, "image_depicts_part", node, 0.95)

    product_rows: list[dict[str, Any]] = []
    for profile in product_profiles.values():
        row = {
            "product_id": profile["product_id"],
            "product": profile["product"],
            "aliases": sorted(profile["aliases"]),
            "doc_ids": sorted(profile["doc_ids"]),
            "node_ids": sorted(profile["node_ids"])[: args.top_items],
            "parts": [name for name, _ in profile["parts"].most_common(args.top_items)],
            "actions": [name for name, _ in profile["actions"].most_common(args.top_items)],
            "faults": [name for name, _ in profile["faults"].most_common(args.top_items)],
            "policies": [name for name, _ in profile["policies"].most_common(args.top_items)],
            "image_ids": sorted(profile["image_ids"])[: args.top_items],
        }
        product_rows.append(row)

    entity_rows = serializable_entities(entities)
    relation_rows = sorted(relations.values(), key=lambda row: (row["relation_type"], row["source_id"], row["target_id"]))
    product_rows.sort(key=lambda row: row["product"])
    visual_rows.sort(key=lambda row: (row["doc_id"], row["image_id"]))
    text_rows.sort(key=lambda row: (row["doc_id"], row["node_id"]))

    write_jsonl(f"{args.kg_dir}/entities.jsonl", entity_rows)
    write_jsonl(f"{args.kg_dir}/relations.jsonl", relation_rows)
    write_jsonl(f"{args.kg_dir}/product_profiles.jsonl", product_rows)
    write_jsonl(f"{args.visual_dir}/images.jsonl", visual_rows)
    write_jsonl(f"{args.text_dir}/nodes.jsonl", text_rows)

    print(
        f"Built KG: entities={len(entity_rows)}, relations={len(relation_rows)}, "
        f"profiles={len(product_rows)}, text_nodes={len(text_rows)}, images={len(visual_rows)}"
    )
    print(f"KG dir: {args.kg_dir}; text index dir: {args.text_dir}; visual index dir: {args.visual_dir}")


if __name__ == "__main__":
    main()
