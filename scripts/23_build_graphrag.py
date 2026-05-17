from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter, defaultdict
from typing import Any

import networkx as nx

from pipeline_common import (
    DEFAULT_EDGES,
    DEFAULT_NODES,
    clean_text,
    ensure_project_dirs,
    normalize_doc_id,
    preview,
    read_jsonl,
    resolve_path,
    write_jsonl,
)


TEXT_FIELDS = (
    "doc_id",
    "section",
    "source_ref",
    "content",
    "visual_title",
    "key_objects",
    "ocr_text",
    "data_or_trends",
    "qa_evidence",
    "visual_caption",
    "visual_summary",
)

VISUAL_TYPES = {"figure", "table", "caption"}

REF_RE = re.compile(
    r"\b(fig\.?|figure|table)\s*([Ss]?\d+(?:[\.\-]\d+)?[A-Za-z]?)"
    r"|([\u56fe\u5716\u8868])\s*([Ss]?\d+(?:[\.\-]\d+)?[A-Za-z]?)",
    re.I,
)

EN_TERM_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9+\-]{2,}|[a-z][a-z0-9+\-]{3,})"
    r"(?:\s+(?:[A-Z][A-Za-z0-9+\-]{2,}|[a-z][a-z0-9+\-]{3,})){0,2}\b"
)

CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,12}")

STOP_TERMS = {
    "this",
    "that",
    "with",
    "from",
    "into",
    "were",
    "been",
    "have",
    "will",
    "should",
    "using",
    "based",
    "figure",
    "table",
    "page",
    "section",
    "system",
    "method",
    "result",
    "results",
    "the",
    "and",
    "for",
    "are",
    "not",
}


def node_text(node: dict[str, Any]) -> str:
    return clean_text(" ".join(clean_text(node.get(field)) for field in TEXT_FIELDS))


def entity_id(entity_type: str, name: str) -> str:
    return f"{entity_type}:{normalize_doc_id(name).lower()}"


def add_entity(
    entities: dict[str, dict[str, Any]],
    entity_type: str,
    name: str,
    node: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> str:
    name = clean_text(name)
    if not name:
        return ""
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
    for alias in aliases or []:
        alias = clean_text(alias)
        if alias and alias.casefold() != name.casefold():
            item["aliases"].add(alias)
    if node:
        item["evidence_count"] += 1
        doc_id = clean_text(node.get("doc_id"))
        node_id = clean_text(node.get("node_id"))
        if doc_id:
            item["doc_ids"].add(doc_id)
        if node_id:
            item["node_ids"].add(node_id)
    return eid


def add_relation(
    relations: dict[tuple[str, str, str, str], dict[str, Any]],
    source_id: str,
    target_id: str,
    relation_type: str,
    node: dict[str, Any],
    weight: float,
) -> None:
    if not source_id or not target_id or source_id == target_id:
        return
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
            "source_ref": preview(node.get("source_ref", ""), 220),
        },
    )
    item["weight"] = round(float(item["weight"]) + weight, 4)


def extract_refs(text: str) -> list[str]:
    refs: list[str] = []
    for match in REF_RE.finditer(text):
        latin_kind = clean_text(match.group(1)).lower()
        latin_no = clean_text(match.group(2))
        zh_kind = clean_text(match.group(3))
        zh_no = clean_text(match.group(4))
        if latin_kind:
            kind = "table" if latin_kind.startswith("table") else "figure"
            refs.append(f"{kind} {latin_no}")
        elif zh_kind and zh_no:
            kind = "table" if zh_kind == "\u8868" else "figure"
            refs.append(f"{kind} {zh_no}")
    return refs


def keep_term(term: str) -> bool:
    term = clean_text(term).strip(".,:;()[]{}<>/\\|\"'")
    if not term:
        return False
    folded = term.casefold()
    if folded in STOP_TERMS:
        return False
    if folded.isdigit():
        return False
    if len(folded) < 3 and not any("\u4e00" <= ch <= "\u9fff" for ch in term):
        return False
    return True


def extract_terms(text: str, limit: int = 12) -> list[str]:
    text = clean_text(text)
    counts: Counter[str] = Counter()
    for ref in extract_refs(text):
        counts[ref] += 4
    for match in EN_TERM_RE.finditer(text):
        term = match.group(0).strip()
        if keep_term(term):
            counts[term] += 2 if any(ch.isupper() for ch in term) else 1
    for match in CJK_RE.finditer(text):
        chunk = match.group(0)
        if len(chunk) <= 6 and keep_term(chunk):
            counts[chunk] += 2
        else:
            for size in (2, 3, 4):
                for index in range(0, max(0, len(chunk) - size + 1)):
                    term = chunk[index : index + size]
                    if keep_term(term):
                        counts[term] += 1
    ranked = sorted(counts.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
    return [term for term, _ in ranked[:limit]]


def graph_from_edges(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if node_id:
            graph.add_node(node_id)
    for edge in edges:
        source = clean_text(edge.get("source_id"))
        target = clean_text(edge.get("target_id"))
        if not source or not target or source == target:
            continue
        weight = max(float(edge.get("weight") or 0.0), 0.01)
        edge_type = clean_text(edge.get("edge_type")) or "related"
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
            graph[source][target]["edge_types"].add(edge_type)
        else:
            graph.add_edge(source, target, weight=weight, edge_types={edge_type})
    return graph


def community_key(node: dict[str, Any]) -> tuple[str, str]:
    doc_id = clean_text(node.get("doc_id")) or "unknown_doc"
    section = clean_text(node.get("section"))
    if section:
        return doc_id, section
    page = clean_text(node.get("page"))
    if page:
        return doc_id, f"page:{page}"
    return doc_id, "document"


def serializable_entity_rows(entities: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in entities.values():
        row = dict(item)
        row["aliases"] = sorted(row["aliases"])
        row["doc_ids"] = sorted(row["doc_ids"])
        row["node_ids"] = sorted(row["node_ids"])
        rows.append(row)
    return sorted(rows, key=lambda row: (row["entity_type"], row["name"]))


def build_graphrag(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    max_terms_per_node: int,
) -> dict[str, list[dict[str, Any]]]:
    graph = graph_from_edges(nodes, edges)
    entities: dict[str, dict[str, Any]] = {}
    relations: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    node_index: list[dict[str, Any]] = []
    entity_links: list[dict[str, Any]] = []
    community_nodes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    node_entities: dict[str, list[str]] = defaultdict(list)

    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if not node_id:
            continue
        doc_id = clean_text(node.get("doc_id")) or "unknown_doc"
        node_type = clean_text(node.get("node_type")) or "text"
        section = clean_text(node.get("section"))
        page = clean_text(node.get("page"))

        doc_entity = add_entity(entities, "product", doc_id, node)
        node_entities[node_id].append(doc_entity)

        if section:
            section_entity = add_entity(entities, "part", section, node)
            node_entities[node_id].append(section_entity)
            add_relation(relations, doc_entity, section_entity, "product_has_part", node, 0.75)

        page_name = f"{doc_id} page {page}" if page else ""
        page_entity = add_entity(entities, "part", page_name, node) if page_name else ""
        if page_entity:
            node_entities[node_id].append(page_entity)
            add_relation(relations, doc_entity, page_entity, "product_has_part", node, 0.25)

        text = node_text(node)
        terms = extract_terms(text, limit=max_terms_per_node)
        term_entities: list[str] = []
        for term in terms:
            eid = add_entity(entities, "part", term, node)
            if not eid:
                continue
            term_entities.append(eid)
            node_entities[node_id].append(eid)
            add_relation(relations, doc_entity, eid, "product_has_part", node, 0.25)
            if section:
                add_relation(relations, section_entity, eid, "product_has_part", node, 0.35)

        image_entity = ""
        if node_type in VISUAL_TYPES and clean_text(node.get("crop_image_path") or node.get("page_image_path")):
            image_name = clean_text(node.get("image_id")) or node_id
            image_entity = add_entity(entities, "image", image_name, node, aliases=[node_id])
            node_entities[node_id].append(image_entity)
            add_relation(relations, doc_entity, image_entity, "product_has_image", node, 0.65)
            for eid in term_entities[:6]:
                add_relation(relations, image_entity, eid, "image_depicts_part", node, 0.85)

        for left, right in itertools.combinations(term_entities[:6], 2):
            add_relation(relations, left, right, "concept_cooccurs", node, 0.08)

        unique_entities = list(dict.fromkeys(eid for eid in node_entities[node_id] if eid))
        for eid in unique_entities:
            entity_links.append(
                {
                    "node_id": node_id,
                    "entity_id": eid,
                    "doc_id": doc_id,
                    "page": page,
                    "node_type": node_type,
                }
            )

        community_nodes[community_key(node)].append(node)
        node_index.append(
            {
                "node_id": node_id,
                "doc_id": doc_id,
                "page": page,
                "node_type": node_type,
                "section": section,
                "degree": graph.degree(node_id) if node_id in graph else 0,
                "entity_ids": unique_entities,
                "source_ref": preview(node.get("source_ref", ""), 220),
                "content_preview": preview(node.get("content", ""), 260),
                "crop_image_path": clean_text(node.get("crop_image_path")),
                "page_image_path": clean_text(node.get("page_image_path")),
            }
        )

    communities: list[dict[str, Any]] = []
    community_id_by_key: dict[tuple[str, str], str] = {}
    for index, (key, group) in enumerate(sorted(community_nodes.items()), start=1):
        doc_id, label = key
        community_id = f"C{index:04d}_{normalize_doc_id(doc_id)}"
        community_id_by_key[key] = community_id
        pages = sorted({clean_text(node.get("page")) for node in group if clean_text(node.get("page"))})
        modalities = Counter(clean_text(node.get("node_type")) or "text" for node in group)
        entity_ids = sorted(
            {
                eid
                for node in group
                for eid in node_entities.get(clean_text(node.get("node_id")), [])
                if eid
            }
        )
        terms = Counter()
        for node in group:
            for term in extract_terms(node_text(node), limit=8):
                terms[term] += 1
        top_terms = [term for term, _ in terms.most_common(10)]
        communities.append(
            {
                "community_id": community_id,
                "doc_id": doc_id,
                "section": "" if label.startswith("page:") else label,
                "label": label,
                "pages": pages,
                "node_ids": [clean_text(node.get("node_id")) for node in group if clean_text(node.get("node_id"))],
                "entity_ids": entity_ids,
                "node_count": len(group),
                "modalities": dict(modalities),
                "top_terms": top_terms,
                "summary": (
                    f"Document {doc_id}, {label}; pages={','.join(pages[:8])}; "
                    f"modalities={','.join(f'{k}:{v}' for k, v in modalities.most_common())}; "
                    f"key_terms={', '.join(top_terms[:6])}"
                ),
            }
        )

    for row in node_index:
        key = community_key(row)
        row["community_id"] = community_id_by_key.get(key, "")

    graph_edges = []
    for edge in edges:
        source = clean_text(edge.get("source_id"))
        target = clean_text(edge.get("target_id"))
        if source and target:
            graph_edges.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "edge_type": clean_text(edge.get("edge_type")) or "related",
                    "weight": float(edge.get("weight") or 0.0),
                    "layer": "document_structure",
                }
            )

    summary = {
        "node_count": len(node_index),
        "edge_count": len(graph_edges),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "community_count": len(communities),
        "visual_node_count": sum(1 for row in node_index if row["node_type"] in VISUAL_TYPES),
        "graph_layers": [
            "document_structure_graph",
            "semantic_entity_graph",
            "community_summary_graph",
        ],
    }
    return {
        "nodes": node_index,
        "edges": graph_edges,
        "entities": serializable_entity_rows(entities),
        "relations": sorted(relations.values(), key=lambda row: (row["relation_type"], row["source_id"], row["target_id"])),
        "entity_links": entity_links,
        "communities": communities,
        "summary": [summary],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a lightweight GraphRAG index from evidence nodes and document graph edges."
    )
    parser.add_argument("--nodes", default=str(DEFAULT_NODES.relative_to(DEFAULT_NODES.parents[2])))
    parser.add_argument("--edges", default=str(DEFAULT_EDGES.relative_to(DEFAULT_EDGES.parents[2])))
    parser.add_argument("--output-dir", default="outputs/graphrag")
    parser.add_argument("--max-terms-per-node", type=int, default=12)
    args = parser.parse_args()

    ensure_project_dirs()
    nodes = read_jsonl(args.nodes)
    edges = read_jsonl(args.edges)
    outputs = build_graphrag(nodes, edges, max_terms_per_node=args.max_terms_per_node)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    summary = outputs["summary"][0]
    print(
        "Built GraphRAG index: "
        f"nodes={summary['node_count']}, edges={summary['edge_count']}, "
        f"entities={summary['entity_count']}, relations={summary['relation_count']}, "
        f"communities={summary['community_count']}"
    )
    print(f"GraphRAG dir: {output_dir}")


if __name__ == "__main__":
    main()
