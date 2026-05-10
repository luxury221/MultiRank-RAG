from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline_common import clean_text, normalize_doc_id, read_csv, read_jsonl, resolve_path, write_csv, write_jsonl
from rerank_lib import AFTER_SALES_INTENT_TERMS


DEFAULT_NODES = "outputs/datafountain_1165_full/nodes.jsonl"
DEFAULT_EDGES = "outputs/datafountain_1165_full/edges.jsonl"
DEFAULT_QUESTIONS = "outputs/datafountain_1165_full/questions.csv"
DEFAULT_OUTPUT_DIR = "outputs/after_sales_kb"

POLICY_DOC_ID = "售后通用政策"

SERVICE_POLICIES = [
    {
        "intent": "return_refund",
        "title": "退换货与退款",
        "content": (
            "售后退换货与退款需要以商品页面、订单状态和平台政策为准。用户申请时应提供订单号、商品型号、"
            "问题描述、商品与外包装照片。若属于质量问题、错发或漏发，售后应优先核实并协助退换或退款；"
            "若涉及7天无理由、运费承担或退款到账时间，应以购买渠道规则和实际审核结果为准。"
        ),
    },
    {
        "intent": "invoice",
        "title": "发票与开票",
        "content": (
            "发票支持情况、发票类型、开票时间和收票方式以订单页或店铺规则为准。用户应提供发票抬头、税号、"
            "订单号和收票信息；若订单页面未显示开票入口，应联系售后客服进一步核实。"
        ),
    },
    {
        "intent": "shipping_damage",
        "title": "物流与包装破损",
        "content": (
            "收到商品时如果发现外包装破损、物流异常或商品受损，应先拍照保留外包装、运单号、商品状态和签收信息，"
            "再联系售后客服。售后需要结合物流记录、商品状态和平台规则判断是否补发、换货、维修或退款。"
        ),
    },
    {
        "intent": "warranty_repair",
        "title": "保修与维修",
        "content": (
            "保修范围通常面向保修期内的产品制造故障。人为损坏、误用、私自拆修、进水、摔落或超出保修期的问题，"
            "一般需要检测后确认是否收费维修。用户应提供订单号、型号、故障现象、照片或视频，以及已有送修记录。"
        ),
    },
    {
        "intent": "troubleshooting",
        "title": "故障排查",
        "content": (
            "故障类问题应先确认商品型号、使用场景、电源或连接状态、错误提示和故障发生步骤。售后回答应优先引用"
            "对应产品手册中的安装、设置、清洁、复位、维护和安全说明；证据不足时，应建议用户补充图片或视频。"
        ),
    },
    {
        "intent": "usage_operation",
        "title": "安装使用与维护",
        "content": (
            "安装、设置、连接、清洁和维护类问题应优先根据产品手册步骤回答。若步骤涉及图示、部件位置或安全警告，"
            "应结合图示节点、相邻正文和标题节点形成证据链，避免只给出孤立文本片段。"
        ),
    },
    {
        "intent": "spec_parts",
        "title": "规格参数与配件",
        "content": (
            "型号、尺寸、电压、容量、部件、配件清单和适配范围等问题应引用手册中的参数表、规格段落或部件图示。"
            "当不同产品手册存在相似术语时，应先确认具体商品型号，再回答对应产品的信息。"
        ),
    },
    {
        "intent": "safety",
        "title": "安全警告",
        "content": (
            "涉及儿童使用、触电、火灾、烫伤、运动风险、压力、燃油或高温部件的问题，应优先返回手册安全警告。"
            "回答应提醒用户停止危险操作，并在证据不足或风险较高时建议联系售后或专业人员处理。"
        ),
    },
]

PRODUCT_CATEGORY_HINTS = {
    "家电清洁": ("空调", "冰箱", "洗碗机", "烤箱", "空气净化器", "吹风机", "蒸汽清洁机"),
    "工具设备": ("电钻", "发电机", "水泵"),
    "数码外设": ("相机", "键盘", "鼠标", "VR", "头显", "健身追踪器"),
    "运动出行": ("健身单车", "摩托艇", "儿童电动摩托车"),
    "家具家居": ("人体工学椅", "温控器"),
}


def detect_intents(text: Any) -> list[str]:
    blob = clean_text(text).casefold()
    intents: list[str] = []
    for intent, terms in AFTER_SALES_INTENT_TERMS.items():
        if any(term.casefold() in blob for term in terms):
            intents.append(intent)
    return intents or ["manual_general"]


def infer_category(doc_id: str) -> str:
    for category, terms in PRODUCT_CATEGORY_HINTS.items():
        if any(term.casefold() in doc_id.casefold() for term in terms):
            return category
    return "通用商品"


def intent_label(intent: str) -> str:
    return {
        "return_refund": "退换货退款",
        "invoice": "发票开票",
        "shipping_damage": "物流破损",
        "warranty_repair": "保修维修",
        "troubleshooting": "故障排查",
        "usage_operation": "安装使用",
        "spec_parts": "规格配件",
        "safety": "安全警告",
        "manual_general": "手册通用",
    }.get(intent, intent)


def enrich_manual_node(node: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(node)
    content_blob = " ".join(
        [
            str(node.get("doc_id", "")),
            str(node.get("section", "")),
            str(node.get("source_ref", "")),
            str(node.get("content", "")),
            str(node.get("visual_summary", "")),
            str(node.get("previous_chunk_preview", "")),
            str(node.get("next_chunk_preview", "")),
        ]
    )
    intents = detect_intents(content_blob)
    doc_id = clean_text(node.get("doc_id")) or "未知商品"
    enriched["knowledge_base"] = "after_sales_kb"
    enriched["kb_domain"] = "customer_after_sales"
    enriched["kb_doc_type"] = "product_manual"
    enriched["product_category"] = infer_category(doc_id)
    enriched["service_intents"] = ";".join(intents)
    enriched.setdefault("paper_domain", "datafountain_customer_service")
    enriched.setdefault("chunk_template", "manual_qa")
    enriched["searchable_text"] = clean_text(content_blob)
    return enriched


def policy_nodes() -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for index, item in enumerate(SERVICE_POLICIES, start=1):
        intent = item["intent"]
        nodes.append(
            {
                "node_id": f"AS_POLICY_{intent.upper()}",
                "doc_id": POLICY_DOC_ID,
                "page": index,
                "node_type": "text",
                "content": item["content"],
                "source_ref": f"{POLICY_DOC_ID} / {item['title']}",
                "section": item["title"],
                "paper_domain": "after_sales_knowledge_base",
                "knowledge_base": "after_sales_kb",
                "kb_domain": "customer_after_sales",
                "kb_doc_type": "service_policy",
                "product_category": "通用售后",
                "service_intents": intent,
                "structure_type": "after_sales_policy",
                "chunk_template": "after_sales_policy",
                "searchable_text": f"{item['title']} {item['content']}",
            }
        )
    return nodes


def profile_nodes(manual_nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in manual_nodes:
        by_doc[clean_text(node.get("doc_id")) or "未知商品"].append(node)

    profiles: list[dict[str, Any]] = []
    profile_edges: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    for doc_id, doc_nodes in sorted(by_doc.items()):
        category = infer_category(doc_id)
        type_counts = Counter(clean_text(node.get("node_type")) or "text" for node in doc_nodes)
        intent_counts: Counter[str] = Counter()
        for node in doc_nodes:
            for intent in clean_text(node.get("service_intents")).split(";"):
                if intent:
                    intent_counts[intent] += 1
        top_intents = [intent for intent, _ in intent_counts.most_common(5)]
        profile_id = f"AS_PROFILE_{normalize_doc_id(doc_id)}"
        content = (
            f"{doc_id} 售后知识条目。产品类别：{category}。"
            f"包含文本块 {type_counts.get('text', 0)} 个、标题 {type_counts.get('title', 0)} 个、"
            f"图示 {type_counts.get('figure', 0)} 个。常见售后意图："
            f"{'、'.join(intent_label(intent) for intent in top_intents) or '手册通用'}。"
            "回答时应优先结合该商品手册节点、相邻图示和通用售后政策形成证据链。"
        )
        profiles.append(
            {
                "node_id": profile_id,
                "doc_id": doc_id,
                "page": 0,
                "node_type": "title",
                "content": content,
                "source_ref": f"{doc_id} / 售后知识库档案",
                "section": "售后知识库档案",
                "paper_domain": "after_sales_knowledge_base",
                "knowledge_base": "after_sales_kb",
                "kb_domain": "customer_after_sales",
                "kb_doc_type": "manual_profile",
                "product_category": category,
                "service_intents": ";".join(top_intents or ["manual_general"]),
                "structure_type": "manual_profile",
                "chunk_template": "after_sales_profile",
                "searchable_text": content,
            }
        )
        for node in doc_nodes:
            profile_edges.append(
                {
                    "source_id": profile_id,
                    "target_id": node["node_id"],
                    "edge_type": "profile_contains",
                    "weight": 0.3,
                }
            )
        catalog_rows.append(
            {
                "doc_id": doc_id,
                "product_category": category,
                "node_count": len(doc_nodes),
                "text_nodes": type_counts.get("text", 0),
                "title_nodes": type_counts.get("title", 0),
                "figure_nodes": type_counts.get("figure", 0),
                "service_intents": ";".join(top_intents or ["manual_general"]),
                "profile_node_id": profile_id,
            }
        )
    return profiles, profile_edges, catalog_rows


def policy_edges(policy_nodes_: list[dict[str, Any]], manual_nodes: list[dict[str, Any]], profile_nodes_: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    policy_by_intent = {node["service_intents"]: node["node_id"] for node in policy_nodes_}
    for policy in policy_nodes_:
        for profile in profile_nodes_:
            edges.append(
                {
                    "source_id": policy["node_id"],
                    "target_id": profile["node_id"],
                    "edge_type": "policy_applies_to_profile",
                    "weight": 0.18,
                }
            )
    for node in manual_nodes:
        for intent in clean_text(node.get("service_intents")).split(";"):
            policy_id = policy_by_intent.get(intent)
            if not policy_id:
                continue
            edges.append(
                {
                    "source_id": policy_id,
                    "target_id": node["node_id"],
                    "edge_type": "policy_supports_intent",
                    "weight": 0.35,
                }
            )
    return edges


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        source = clean_text(edge.get("source_id"))
        target = clean_text(edge.get("target_id"))
        edge_type = clean_text(edge.get("edge_type")) or "related"
        if not source or not target or source == target:
            continue
        key = (source, target, edge_type)
        weight = float(edge.get("weight") or 1.0)
        if key in dedup:
            dedup[key]["weight"] = round(float(dedup[key]["weight"]) + weight, 4)
        else:
            dedup[key] = {"source_id": source, "target_id": target, "edge_type": edge_type, "weight": round(weight, 4)}
    return list(dedup.values())


def convert_questions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        question = clean_text(row.get("question"))
        if not question:
            continue
        qid = clean_text(row.get("question_id")) or clean_text(row.get("id"))
        if qid and not qid.startswith("AS_Q"):
            qid = f"AS_Q{qid}"
        intents = detect_intents(question)
        converted.append(
            {
                "question_id": qid or f"AS_Q{len(converted) + 1:04d}",
                "doc_id": "",
                "question": question,
                "answer": "",
                "question_type": f"售后问答/{intent_label(intents[0])}",
                "gold_node_ids": "",
                "gold_pages": "",
                "gold_modalities": "text;figure",
                "evidence_note": "售后知识库自动导入问题，答案由 RAG 证据链生成。",
            }
        )
    return converted


def write_markdown(output_dir: Path, catalog_rows: list[dict[str, Any]], policies: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# 售后知识库",
        "",
        "本目录由 `scripts/19_build_after_sales_kb.py` 自动生成，面向售后问答、商品手册检索和证据链展示。",
        "",
        "## 文件",
        "",
        "- `nodes.jsonl`：商品手册节点、通用售后政策节点、商品档案节点。",
        "- `edges.jsonl`：手册原始结构边、商品档案边、政策到售后意图的支撑边。",
        "- `questions.csv`：售后测试问题。",
        "- `catalog.csv`：商品手册目录和统计。",
        "- `policies.json`：通用售后政策。",
        "- `manifest.json`：构建统计。",
        "",
        "## 构建统计",
        "",
        f"- 节点数：{manifest['nodes']}",
        f"- 边数：{manifest['edges']}",
        f"- 问题数：{manifest['questions']}",
        f"- 商品手册数：{manifest['manuals']}",
        "",
        "## 通用售后政策",
        "",
    ]
    for policy in policies:
        lines.append(f"### {policy['section']}")
        lines.append("")
        lines.append(policy["content"])
        lines.append("")
    lines.extend(["## 商品目录示例", ""])
    for row in catalog_rows[:20]:
        lines.append(
            f"- {row['doc_id']}：{row['product_category']}，节点 {row['node_count']}，意图 {row['service_intents']}"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def activate(output_dir: Path) -> None:
    parsed = resolve_path("outputs/parsed")
    parsed.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / "nodes.jsonl", parsed / "nodes.jsonl")
    shutil.copy2(output_dir / "edges.jsonl", parsed / "edges.jsonl")
    shutil.copy2(output_dir / "questions.csv", resolve_path("data/questions.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an after-sales RAG knowledge base from imported manual nodes.")
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--activate", action="store_true", help="Copy the generated KB into outputs/parsed and data/questions.csv.")
    args = parser.parse_args()

    source_nodes = [node for node in read_jsonl(args.nodes) if clean_text(node.get("content"))]
    source_edges = read_jsonl(args.edges)
    source_questions = read_csv(args.questions)
    if not source_nodes:
        raise SystemExit(f"No nodes found: {resolve_path(args.nodes)}")

    manual_nodes = [enrich_manual_node(node) for node in source_nodes]
    policies = policy_nodes()
    profiles, profile_edges, catalog_rows = profile_nodes(manual_nodes)
    nodes = policies + profiles + manual_nodes
    edges = dedupe_edges(source_edges + profile_edges + policy_edges(policies, manual_nodes, profiles))
    questions = convert_questions(source_questions)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "nodes.jsonl", nodes)
    write_jsonl(output_dir / "edges.jsonl", edges)
    write_csv(output_dir / "questions.csv", questions)
    write_csv(output_dir / "catalog.csv", catalog_rows)
    (output_dir / "policies.json").write_text(json.dumps(policies, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "name": "after_sales_kb",
        "nodes": len(nodes),
        "edges": len(edges),
        "questions": len(questions),
        "manuals": len(catalog_rows),
        "policies": len(policies),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_dir, catalog_rows, policies, manifest)

    if args.activate:
        activate(output_dir)
    print(f"Wrote after-sales KB to {output_dir}")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
