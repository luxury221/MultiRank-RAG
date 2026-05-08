from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from typing import Any

import networkx as nx

from embedding_index import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingIndex,
)
from pipeline_common import as_float, clean_text, preview, split_multi


TYPE_WEIGHTS = {
    "text": 1.0,
    "table": 1.2,
    "figure": 1.2,
    "caption": 1.0,
    "page": 0.5,
    "title": 0.6,
    "equation": 0.9,
}

DEFAULT_MODALITIES = ["text", "table", "figure", "page", "caption"]
RETRIEVER_CHOICES = ["hybrid", "embedding", "lexical"]

BASE_EDGE_TYPE_WEIGHTS = {
    "same_page": 0.05,
    "belongs_to_page": 0.05,
    "text_ref_table": 1.0,
    "text_ref_figure": 1.0,
    "table_caption": 1.2,
    "figure_caption": 1.2,
    "related": 0.2,
}

DOCUMENT_REF_RE = re.compile(
    r"\b(fig\.?|figure|table)\s*([Ss]?\d+(?:[\.\-]\d+)?[A-Za-z]?)"
    r"|([\u56fe\u5716\u8868])\s*([Ss]?\d+(?:[\.\-]\d+)?[A-Za-z]?)",
    re.I,
)

VISUAL_NODE_TYPES = {"table", "figure", "caption"}
VISUAL_TEXT_FIELDS = (
    "visual_title",
    "visual_type",
    "key_objects",
    "data_or_trends",
    "qa_evidence",
    "limitations",
    "visual_caption",
    "visual_summary",
)

TABLE_TERMS = ("table", "tabular", "\u8868", "\u8868\u683c", "\u8868\u9898")
FIGURE_TERMS = (
    "fig.",
    "figure",
    "chart",
    "plot",
    "image",
    "diagram",
    "curve",
    "\u56fe",
    "\u5716",
    "\u56fe\u7247",
    "\u56fe\u50cf",
    "\u56fe\u8868",
    "\u56fe\u6587",
    "\u66f2\u7ebf",
    "\u8d8b\u52bf",
)
CROSS_MODAL_TERMS = ("cross-modal", "multimodal", "\u8de8\u6a21\u6001", "\u591a\u6a21\u6001", "\u7ed3\u5408")
LOCATION_TERMS = ("bbox", "bounding-box", "grounding", "\u5b9a\u4f4d", "\u8bc1\u636e\u5b9a\u4f4d", "\u9875")
TEXT_FACT_TERMS = ("\u6587\u672c\u4e8b\u5b9e",)
VISUAL_QUESTION_TYPE_TERMS = (
    "\u8868\u683c\u95ee\u7b54",
    "\u56fe\u6587\u4e00\u81f4\u6027",
    "\u56fe\u8868\u7406\u89e3",
    "\u8de8\u6a21\u6001\u7efc\u5408",
)


def lexical_similarity(query: str, texts: list[str]) -> list[float]:
    query = clean_text(query)
    texts = [clean_text(text) for text in texts]
    if not texts:
        return []
    if not query:
        return [0.0 for _ in texts]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [query] + texts
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        return [float(max(0.0, score)) for score in sims]
    except Exception:
        return [_char_jaccard(query, text) for text in texts]


def similarity_scores(
    question: dict[str, Any],
    nodes: list[dict[str, Any]],
    retriever: str = "lexical",
    embedding_index: EmbeddingIndex | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: str = str(DEFAULT_EMBEDDING_CACHE_DIR),
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    hybrid_alpha: float = 0.7,
) -> dict[str, float]:
    nodes = [node for node in nodes if clean_text(node.get("node_id")) and clean_text(node.get("content"))]
    if not nodes:
        return {}

    retriever = (retriever or "lexical").lower()
    if retriever in {"embedding", "hybrid"}:
        index = embedding_index or EmbeddingIndex.from_nodes(
            nodes,
            model_name=embedding_model,
            cache_dir=embedding_cache,
            device=embedding_device,
            batch_size=embedding_batch_size,
        )
        embedding_scores = index.score(question.get("question", ""), nodes)
        if retriever == "embedding":
            return embedding_scores
        lexical_values = lexical_similarity(question.get("question", ""), [node.get("content", "") for node in nodes])
        lexical_scores = {node["node_id"]: score for node, score in zip(nodes, lexical_values)}
        node_ids = [node["node_id"] for node in nodes]
        embedding_norm = normalize_scores(embedding_scores, node_ids)
        lexical_norm = normalize_scores(lexical_scores, node_ids)
        alpha = min(1.0, max(0.0, hybrid_alpha))
        return {
            node_id: alpha * embedding_norm.get(node_id, 0.0) + (1.0 - alpha) * lexical_norm.get(node_id, 0.0)
            for node_id in node_ids
        }
    if retriever != "lexical":
        raise ValueError(f"Unknown retriever: {retriever}. Expected one of {', '.join(RETRIEVER_CHOICES)}.")

    texts = [node.get("content", "") for node in nodes]
    scores = lexical_similarity(question.get("question", ""), texts)
    return {node["node_id"]: score for node, score in zip(nodes, scores)}


def _char_jaccard(a: str, b: str) -> float:
    a_set = {a[i : i + 2] for i in range(max(1, len(a) - 1))}
    b_set = {b[i : i + 2] for i in range(max(1, len(b) - 1))}
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def normalize_scores(scores: dict[str, float], keys: list[str] | None = None) -> dict[str, float]:
    if keys is None:
        keys = list(scores.keys())
    values = [scores.get(key, 0.0) for key in keys]
    if not values:
        return {}
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        fill = 0.0 if math.isclose(high, 0.0) else 1.0
        return {key: fill for key in keys}
    return {key: (scores.get(key, 0.0) - low) / (high - low) for key in keys}


def build_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        node_id = node.get("node_id")
        if node_id:
            graph.add_node(node_id)
    for edge in edges:
        source = edge.get("source_id")
        target = edge.get("target_id")
        if not source or not target or source == target:
            continue
        weight = max(as_float(edge.get("weight"), 1.0), 0.01)
        edge_type = clean_text(edge.get("edge_type")) or "related"
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
            edge_types = set(graph[source][target].get("edge_types", []))
            edge_types.add(edge_type)
            graph[source][target]["edge_types"] = sorted(edge_types)
        else:
            graph.add_edge(source, target, weight=weight, edge_type=edge_type, edge_types=[edge_type])
    return graph


def _prepare_ref_text(text: Any) -> str:
    text = clean_text(text)
    text = re.sub(r"F\s*i\s*g\s*u\s*r\s*e", "Figure", text, flags=re.I)
    text = re.sub(r"T\s*a\s*b\s*l\s*e", "Table", text, flags=re.I)
    return text


def _normalize_ref_no(raw: str) -> str:
    raw = clean_text(raw).lower().strip(".:-")
    if raw.startswith("s"):
        match = re.match(r"s\s*(\d+)", raw)
        return f"s{match.group(1)}" if match else raw
    match = re.match(r"(\d+)", raw)
    return match.group(1) if match else raw


def extract_document_refs(text: Any) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = set()
    text = _prepare_ref_text(text)
    for match in DOCUMENT_REF_RE.finditer(text):
        latin_kind = clean_text(match.group(1)).lower()
        latin_no = clean_text(match.group(2))
        zh_kind = clean_text(match.group(3))
        zh_no = clean_text(match.group(4))
        if latin_kind:
            kind = "table" if latin_kind.startswith("table") else "figure"
            refs.add((kind, _normalize_ref_no(latin_no)))
        elif zh_kind:
            kind = "table" if zh_kind == "表" else "figure"
            refs.add((kind, _normalize_ref_no(zh_no)))
    return refs


def _looks_like_toc_entry(text: Any) -> bool:
    text = clean_text(text)
    if "···" in text:
        return True
    return bool(re.search(r"\.{4,}\s*\d+\s*$", text))


def _question_blob(question: dict[str, Any]) -> str:
    return clean_text(f"{question.get('question_type', '')} {question.get('question', '')}").casefold()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.casefold() in text for term in terms)


def question_intent(question: dict[str, Any]) -> dict[str, bool]:
    blob = _question_blob(question)
    qtype = clean_text(question.get("question_type")).casefold()
    wants_table = _contains_any(blob, TABLE_TERMS)
    wants_figure = _contains_any(blob, FIGURE_TERMS)
    wants_cross = _contains_any(blob, CROSS_MODAL_TERMS)
    wants_location = _contains_any(blob, LOCATION_TERMS)
    text_fact = _contains_any(qtype, TEXT_FACT_TERMS)
    visual_type = _contains_any(qtype, VISUAL_QUESTION_TYPE_TERMS)
    wants_visual = visual_type or wants_table or wants_figure or wants_cross or wants_location
    return {
        "table": wants_table,
        "figure": wants_figure,
        "cross": wants_cross,
        "location": wants_location,
        "text_fact": text_fact,
        "visual": wants_visual,
    }


def visual_text_for_node(node: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in VISUAL_TEXT_FIELDS:
        value = clean_text(node.get(field))
        if value:
            parts.append(value)
    return " ".join(parts)


def node_has_visual_crop(node: dict[str, Any]) -> bool:
    return bool(clean_text(node.get("crop_image_path")))


def node_has_visual_caption(node: dict[str, Any]) -> bool:
    return bool(clean_text(node.get("visual_caption")) or clean_text(node.get("qa_evidence")))


def visual_signal_weight(question: dict[str, Any], visual_raw: dict[str, float]) -> float:
    if not any(score > 0 for score in visual_raw.values()):
        return 0.0
    intent = question_intent(question)
    if intent["text_fact"] and not (intent["table"] or intent["figure"] or intent["location"]):
        return 0.02
    if intent["table"] or intent["figure"]:
        return 0.14
    if intent["cross"] or intent["location"]:
        return 0.08
    if intent["visual"]:
        return 0.06
    return 0.03


def visual_grounding_scores(
    question: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    graph: nx.Graph,
    candidate_ids: list[str],
) -> dict[str, float]:
    if not candidate_ids:
        return {}
    intent = question_intent(question)
    visual_texts = [visual_text_for_node(nodes_by_id.get(node_id, {})) for node_id in candidate_ids]
    visual_sims = lexical_similarity(question.get("question", ""), visual_texts)
    sim_by_id = {node_id: score for node_id, score in zip(candidate_ids, visual_sims)}
    scores: dict[str, float] = {}
    for node_id in candidate_ids:
        node = nodes_by_id.get(node_id, {})
        node_type = clean_text(node.get("node_type")) or "text"
        visual_text = visual_text_for_node(node)
        semantic = sim_by_id.get(node_id, 0.0) if visual_text else 0.0
        type_bonus = 0.0
        if intent["table"] and node_type == "table":
            type_bonus += 0.18
        if intent["figure"] and node_type in {"figure", "caption"}:
            type_bonus += 0.18
        if intent["visual"] and node_type in VISUAL_NODE_TYPES:
            type_bonus += 0.08

        evidence_bonus = 0.0
        if node_has_visual_crop(node):
            evidence_bonus += 0.03
        if node_has_visual_caption(node):
            evidence_bonus += 0.05
        if clean_text(node.get("qa_evidence")):
            evidence_bonus += 0.04

        neighbor_bonus = 0.0
        if node_id in graph:
            for neighbor in graph.neighbors(node_id):
                neighbor_node = nodes_by_id.get(neighbor, {})
                neighbor_type = clean_text(neighbor_node.get("node_type")) or "text"
                if neighbor_type not in VISUAL_NODE_TYPES:
                    continue
                edge_data = graph.get_edge_data(node_id, neighbor, default={})
                edge_types = set(edge_data.get("edge_types") or [edge_data.get("edge_type", "related")])
                relation_bonus = 0.08 if edge_types & {"table_caption", "figure_caption"} else 0.03
                if node_has_visual_crop(neighbor_node):
                    relation_bonus += 0.02
                if node_has_visual_caption(neighbor_node):
                    relation_bonus += 0.03
                neighbor_bonus = max(neighbor_bonus, relation_bonus)

        scores[node_id] = max(0.0, 0.65 * semantic + type_bonus + evidence_bonus + neighbor_bonus)
    return scores


def _reference_weight(node_type: str, refs: set[tuple[str, str]]) -> float:
    if not refs:
        return 0.0
    if any(kind == "table" for kind, _ in refs):
        return {"table": 0.9, "caption": 0.95, "text": 0.75, "page": 0.1, "figure": 0.1}.get(node_type, 0.2)
    return {"figure": 1.0, "caption": 0.95, "text": 0.75, "page": 0.1, "table": 0.1}.get(node_type, 0.2)


def reference_scores(
    question: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    graph: nx.Graph,
    candidate_ids: list[str],
) -> dict[str, float]:
    question_refs = extract_document_refs(question.get("question", ""))
    if not question_refs:
        return {node_id: 0.0 for node_id in candidate_ids}

    ordered_ids = list(nodes_by_id.keys())
    node_refs: dict[str, set[tuple[str, str]]] = {}
    scores = {node_id: 0.0 for node_id in candidate_ids}
    for node_id, node in nodes_by_id.items():
        content = node.get("content", "")
        text = f"{content} {node.get('source_ref', '')}"
        refs = set() if _looks_like_toc_entry(content) else extract_document_refs(text)
        node_refs[node_id] = refs
        hits = refs & question_refs
        if node_id in scores and hits:
            node_type = clean_text(node.get("node_type")) or "text"
            scores[node_id] = max(scores[node_id], _reference_weight(node_type, hits))

    candidate_set = set(candidate_ids)
    for idx, node_id in enumerate(ordered_ids):
        anchor_hits = node_refs.get(node_id, set()) & question_refs
        if not anchor_hits:
            continue
        anchor = nodes_by_id[node_id]
        anchor_doc = clean_text(anchor.get("doc_id"))
        anchor_page = str(anchor.get("page", ""))
        anchor_score = _reference_weight(clean_text(anchor.get("node_type")) or "text", anchor_hits)
        for step in range(1, 4):
            near_idx = idx + step
            if near_idx >= len(ordered_ids):
                break
            near_id = ordered_ids[near_idx]
            near_node = nodes_by_id[near_id]
            if clean_text(near_node.get("doc_id")) != anchor_doc or str(near_node.get("page", "")) != anchor_page:
                break
            near_type = clean_text(near_node.get("node_type")) or "text"
            if any(kind == "table" for kind, _ in anchor_hits) and near_type == "table":
                if near_id in candidate_set:
                    scores[near_id] = max(scores[near_id], 0.95 / math.sqrt(step))
                break
            if any(kind == "figure" for kind, _ in anchor_hits) and near_type in {"figure", "caption"}:
                if near_id in candidate_set:
                    scores[near_id] = max(scores[near_id], 0.95 / math.sqrt(step))
                break
        for step in range(1, 3):
            near_idx = idx - step
            if near_idx < 0:
                break
            near_id = ordered_ids[near_idx]
            near_node = nodes_by_id[near_id]
            if clean_text(near_node.get("doc_id")) != anchor_doc or str(near_node.get("page", "")) != anchor_page:
                break
            near_type = clean_text(near_node.get("node_type")) or "text"
            if any(kind == "figure" for kind, _ in anchor_hits) and near_type in {"figure", "caption"}:
                if near_id in candidate_set:
                    scores[near_id] = max(scores[near_id], 0.75 / math.sqrt(step))
                break

        if node_id in graph:
            for neighbor in graph.neighbors(node_id):
                if neighbor not in candidate_set:
                    continue
                edge_data = graph.get_edge_data(node_id, neighbor, default={})
                edge_types = set(edge_data.get("edge_types") or [edge_data.get("edge_type", "related")])
                if edge_types & {"table_caption", "figure_caption", "text_ref_table", "text_ref_figure"}:
                    relation_factor = 0.85
                elif "same_page" in edge_types:
                    relation_factor = 0.35
                else:
                    relation_factor = 0.2
                scores[neighbor] = max(scores[neighbor], anchor_score * relation_factor)

    return scores


def query_modality_profile(question: dict[str, Any]) -> dict[str, dict[str, float]]:
    qtype = clean_text(question.get("question_type"))
    query = clean_text(question.get("question")).lower()

    wants_table = "表格" in qtype or "table" in query or "表 " in query or "表" in query
    wants_figure = (
        "图表" in qtype
        or "图文" in qtype
        or "figure" in query
        or "fig." in query
        or "图 " in query
        or "图" in query
        or "chart" in query
    )
    wants_cross = "跨模态" in qtype or ("结合" in query and (wants_table or wants_figure))
    wants_location = "证据定位" in qtype or "定位" in query or "页" in query

    node_type_weights = {"text": 1.0, "caption": 0.8, "table": 0.5, "figure": 0.5, "page": 0.1}
    edge_type_weights = dict(BASE_EDGE_TYPE_WEIGHTS)

    if wants_table:
        node_type_weights.update({"table": 1.6, "caption": 1.1, "text": 0.9, "figure": 0.2, "page": 0.05})
        edge_type_weights.update({"text_ref_table": 1.5, "table_caption": 1.4, "same_page": 0.03})
    if wants_figure:
        node_type_weights.update({"figure": 1.6, "caption": 1.2, "text": 0.9, "table": 0.25, "page": 0.05})
        edge_type_weights.update({"text_ref_figure": 1.5, "figure_caption": 1.4, "same_page": 0.03})
    if wants_cross:
        node_type_weights.update({"text": 1.1, "table": 1.2, "figure": 1.2, "caption": 1.1, "page": 0.05})
        edge_type_weights.update(
            {
                "text_ref_table": 1.5,
                "text_ref_figure": 1.5,
                "table_caption": 1.4,
                "figure_caption": 1.4,
                "same_page": 0.04,
            }
        )
    if wants_location:
        node_type_weights.update({"text": 1.0, "table": 1.0, "figure": 1.0, "caption": 1.2, "page": 0.4})
        edge_type_weights.update({"belongs_to_page": 0.2, "same_page": 0.04})

    return {"node_type_weights": node_type_weights, "edge_type_weights": edge_type_weights}


def graph_signal_multipliers(question: dict[str, Any]) -> dict[str, float]:
    qtype = clean_text(question.get("question_type"))
    query = clean_text(question.get("question")).lower()

    if "文本事实" in qtype or "证据定位" in qtype:
        return {"ppr": 0.0, "bridge": 0.0}
    if "表格" in qtype or "table" in query or "表" in query:
        return {"ppr": 1.0, "bridge": 0.8}
    if "图表理解" in qtype or "chart" in query:
        return {"ppr": 1.0, "bridge": 1.0}
    if "图文一致性" in qtype:
        return {"ppr": 0.2, "bridge": 0.0}
    if "跨模态" in qtype:
        return {"ppr": 0.1, "bridge": 1.0}
    if "figure" in query or "fig." in query or "图" in query:
        return {"ppr": 0.5, "bridge": 0.6}
    return {"ppr": 0.0, "bridge": 0.0}


def reference_signal_multiplier(question: dict[str, Any]) -> float:
    qtype = clean_text(question.get("question_type"))
    query = clean_text(question.get("question")).lower()

    if "文本事实" in qtype or "证据定位" in qtype:
        return 0.0
    if "表格" in qtype or "table" in query or "表" in query:
        return 1.5
    if "图表理解" in qtype or "chart" in query:
        return 1.0
    if "图文一致性" in qtype:
        return 1.5
    if "跨模态" in qtype:
        return 0.2
    if "figure" in query or "fig." in query or "图" in query:
        return 1.0
    return 0.0


def retrieve_candidates(
    question: dict[str, Any],
    nodes: list[dict[str, Any]],
    top_k: int = 10,
    retriever: str = "lexical",
    embedding_index: EmbeddingIndex | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: str = str(DEFAULT_EMBEDDING_CACHE_DIR),
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    hybrid_alpha: float = 0.7,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    doc_id = clean_text(question.get("doc_id"))
    pool = [node for node in nodes if clean_text(node.get("content"))]
    if doc_id:
        filtered = [node for node in pool if clean_text(node.get("doc_id")) == doc_id]
        if filtered:
            pool = filtered
    score_by_id = similarity_scores(
        question,
        pool,
        retriever=retriever,
        embedding_index=embedding_index,
        embedding_model=embedding_model,
        embedding_cache=embedding_cache,
        embedding_device=embedding_device,
        embedding_batch_size=embedding_batch_size,
        hybrid_alpha=hybrid_alpha,
    )
    ranked = sorted(pool, key=lambda node: score_by_id.get(node["node_id"], 0.0), reverse=True)
    return ranked[:top_k], score_by_id


def ppr_scores(
    graph: nx.Graph,
    candidate_ids: list[str],
    sim_scores: dict[str, float],
) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {node_id: 0.0 for node_id in candidate_ids}
    seeds = [node_id for node_id in candidate_ids if node_id in graph]
    if not seeds:
        return {node_id: 0.0 for node_id in candidate_ids}
    raw_total = sum(max(sim_scores.get(node_id, 0.0), 0.0) for node_id in seeds)
    if raw_total <= 0:
        personalization = {node_id: 1.0 / len(seeds) for node_id in seeds}
    else:
        personalization = {
            node_id: max(sim_scores.get(node_id, 0.0), 0.0) / raw_total for node_id in seeds
        }
    full_personalization = {node_id: 0.0 for node_id in graph.nodes}
    full_personalization.update(personalization)
    try:
        ranks = nx.pagerank(
            graph,
            alpha=0.85,
            personalization=full_personalization,
            weight="weight",
            max_iter=200,
            tol=1e-08,
        )
    except nx.PowerIterationFailedConvergence:
        ranks = {node_id: 0.0 for node_id in graph.nodes}
    return {node_id: float(ranks.get(node_id, 0.0)) for node_id in candidate_ids}


def bridge_scores(
    question: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    graph: nx.Graph,
    candidate_ids: list[str],
    tau: float = 0.2,
    modalities: list[str] | None = None,
    neighbor_sim: dict[str, float] | None = None,
) -> dict[str, float]:
    modalities = modalities or DEFAULT_MODALITIES
    if neighbor_sim is None:
        all_node_ids = list(nodes_by_id.keys())
        all_texts = [nodes_by_id[node_id].get("content", "") for node_id in all_node_ids]
        sim_values = lexical_similarity(question.get("question", ""), all_texts)
        neighbor_sim = {node_id: score for node_id, score in zip(all_node_ids, sim_values)}

    profile = query_modality_profile(question)
    node_type_weights = profile["node_type_weights"]
    edge_type_weights = profile["edge_type_weights"]
    target_total = sum(node_type_weights.get(modality, 0.0) for modality in modalities)
    target_total = max(1.0, target_total)

    scores: dict[str, float] = {}
    for node_id in candidate_ids:
        node = nodes_by_id.get(node_id, {})
        if _looks_like_toc_entry(node.get("content", "")):
            scores[node_id] = 0.0
            continue
        node_type = clean_text(node.get("node_type")) or "text"
        candidate_type_score = node_type_weights.get(node_type, 0.0)
        degree = graph.degree(node_id) if node_id in graph else 0
        degree_norm = 1.0 / math.log(2.0 + min(degree, 12))
        covered_weight = candidate_type_score if node_type in modalities else 0.0
        neighbor_score = 0.0
        if node_id in graph:
            for neighbor in graph.neighbors(node_id):
                neighbor_node = nodes_by_id.get(neighbor, {})
                if _looks_like_toc_entry(neighbor_node.get("content", "")):
                    continue
                neighbor_type = clean_text(neighbor_node.get("node_type")) or "text"
                if neighbor_type not in modalities:
                    continue
                sim = max(neighbor_sim.get(neighbor, 0.0), 0.0)
                if sim < tau and neighbor_type not in {"table", "figure", "caption"}:
                    continue

                edge_data = graph.get_edge_data(node_id, neighbor, default={})
                edge_types = edge_data.get("edge_types") or [edge_data.get("edge_type", "related")]
                relation_weight = max(
                    edge_type_weights.get(edge_type, edge_type_weights.get("related", 0.2))
                    for edge_type in edge_types
                )
                modality_weight = node_type_weights.get(neighbor_type, 0.0)
                covered_weight += modality_weight
                neighbor_score += relation_weight * modality_weight * (0.5 + sim)

        coverage = min(1.0, covered_weight / target_total)
        scores[node_id] = (0.35 * candidate_type_score + 0.65 * neighbor_score) * coverage * degree_norm
    for node_id in candidate_ids:
        if _looks_like_toc_entry(nodes_by_id.get(node_id, {}).get("content", "")):
            scores[node_id] = 0.0
    return scores


def rank_question(
    question: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    top_k: int = 10,
    candidate_rows: list[dict[str, Any]] | None = None,
    alpha: float = 0.93,
    beta: float = 0.07,
    lambda_s: float = 0.85,
    lambda_p: float = 0.0,
    lambda_b: float = 0.15,
    lambda_r: float = 0.1,
    tau: float = 0.2,
    retriever: str = "lexical",
    embedding_index: EmbeddingIndex | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: str = str(DEFAULT_EMBEDDING_CACHE_DIR),
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    hybrid_alpha: float = 0.7,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    nodes_by_id = {node["node_id"]: node for node in nodes if node.get("node_id")}
    graph = build_graph(nodes, edges)
    doc_id = clean_text(question.get("doc_id"))
    similarity_pool = [node for node in nodes if clean_text(node.get("content"))]
    if doc_id:
        doc_nodes = [node for node in similarity_pool if clean_text(node.get("doc_id")) == doc_id]
        if doc_nodes:
            similarity_pool = doc_nodes
    sim_scores = similarity_scores(
        question,
        similarity_pool,
        retriever=retriever,
        embedding_index=embedding_index,
        embedding_model=embedding_model,
        embedding_cache=embedding_cache,
        embedding_device=embedding_device,
        embedding_batch_size=embedding_batch_size,
        hybrid_alpha=hybrid_alpha,
    )

    if candidate_rows:
        candidate_ids = [row["node_id"] for row in candidate_rows if row.get("node_id") in nodes_by_id]
        candidate_nodes = [nodes_by_id[node_id] for node_id in candidate_ids]
        original_scores = {
            row["node_id"]: as_float(row.get("score") or row.get("sim_score"), sim_scores.get(row["node_id"], 0.0))
            for row in candidate_rows
            if row.get("node_id") in nodes_by_id
        }
    else:
        candidate_nodes, candidate_scores = retrieve_candidates(
            question,
            nodes,
            top_k=top_k,
            retriever=retriever,
            embedding_index=embedding_index,
            embedding_model=embedding_model,
            embedding_cache=embedding_cache,
            embedding_device=embedding_device,
            embedding_batch_size=embedding_batch_size,
            hybrid_alpha=hybrid_alpha,
        )
        candidate_ids = [node["node_id"] for node in candidate_nodes]
        original_scores = {node_id: candidate_scores.get(node_id, sim_scores.get(node_id, 0.0)) for node_id in candidate_ids}

    candidate_ids = list(dict.fromkeys(node_id for node_id in candidate_ids if node_id in nodes_by_id))

    sim_norm = normalize_scores(sim_scores, candidate_ids)
    ppr_raw = ppr_scores(graph, candidate_ids, sim_scores)
    ppr_norm = normalize_scores(ppr_raw, candidate_ids)
    bridge_raw = bridge_scores(question, nodes_by_id, graph, candidate_ids, tau=tau, neighbor_sim=sim_scores)
    bridge_norm = normalize_scores(bridge_raw, candidate_ids)
    ref_raw = reference_scores(question, nodes_by_id, graph, candidate_ids)
    ref_norm = normalize_scores(ref_raw, candidate_ids)
    visual_raw = visual_grounding_scores(question, nodes_by_id, graph, candidate_ids)
    visual_norm = normalize_scores(visual_raw, candidate_ids)
    original_norm = normalize_scores(original_scores, candidate_ids)
    graph_mix = graph_signal_multipliers(question)
    ppr_multiplier = max(0.0, min(1.0, graph_mix.get("ppr", 0.0)))
    bridge_multiplier = max(0.0, min(1.0, graph_mix.get("bridge", 0.0)))
    g2_beta = beta * ppr_multiplier
    g2_alpha = alpha + beta * (1.0 - ppr_multiplier)
    g3_p = lambda_p * ppr_multiplier
    g3_b = lambda_b * bridge_multiplier
    g3_s = lambda_s + lambda_p * (1.0 - ppr_multiplier) + lambda_b * (1.0 - bridge_multiplier)
    ref_multiplier = max(0.0, reference_signal_multiplier(question))
    g3_r = min(lambda_r * ref_multiplier if any(score > 0 for score in ref_raw.values()) else 0.0, max(0.0, g3_s))
    g3_s = max(0.0, g3_s - g3_r)
    g3_scores = {
        node_id: (
            g3_s * sim_norm.get(node_id, 0.0)
            + g3_p * ppr_norm.get(node_id, 0.0)
            + g3_b * bridge_norm.get(node_id, 0.0)
            + g3_r * ref_norm.get(node_id, 0.0)
        )
        for node_id in candidate_ids
    }
    visual_weight = visual_signal_weight(question, visual_raw)
    g4_scores = {
        node_id: g3_scores.get(node_id, 0.0)
        + visual_weight * visual_norm.get(node_id, 0.0) * max(0.05, 1.0 - g3_scores.get(node_id, 0.0))
        for node_id in candidate_ids
    }

    method_rankings: dict[str, list[tuple[str, float]]] = {
        "G0": [(node_id, original_norm.get(node_id, 0.0)) for node_id in candidate_ids],
        "G1": sorted(
            [(node_id, sim_norm.get(node_id, 0.0)) for node_id in candidate_ids],
            key=lambda item: item[1],
            reverse=True,
        ),
        "G2": sorted(
            [
                (
                    node_id,
                    g2_alpha * sim_norm.get(node_id, 0.0) + g2_beta * ppr_norm.get(node_id, 0.0),
                )
                for node_id in candidate_ids
            ],
            key=lambda item: item[1],
            reverse=True,
        ),
        "G3": sorted(
            [(node_id, g3_scores.get(node_id, 0.0)) for node_id in candidate_ids],
            key=lambda item: item[1],
            reverse=True,
        ),
        "G4": sorted(
            [(node_id, g4_scores.get(node_id, 0.0)) for node_id in candidate_ids],
            key=lambda item: item[1],
            reverse=True,
        ),
    }

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    rows: list[dict[str, Any]] = []
    for method, ranking in method_rankings.items():
        for rank, (node_id, score) in enumerate(ranking[:top_k], start=1):
            node = nodes_by_id[node_id]
            rows.append(
                {
                    "question_id": question.get("question_id", ""),
                    "doc_id": node.get("doc_id", ""),
                    "question": question.get("question", ""),
                    "method": method,
                    "rank": rank,
                    "node_id": node_id,
                    "node_type": node.get("node_type", ""),
                    "page": node.get("page", ""),
                    "score": round(float(score), 6),
                    "sim_score": round(sim_norm.get(node_id, 0.0), 6),
                    "ppr_score": round(ppr_norm.get(node_id, 0.0), 6),
                    "bridge_score": round(bridge_norm.get(node_id, 0.0), 6),
                    "ref_score": round(ref_norm.get(node_id, 0.0), 6),
                    "visual_score": round(visual_norm.get(node_id, 0.0), 6),
                    "has_visual_crop": int(node_has_visual_crop(node)),
                    "has_visual_caption": int(node_has_visual_caption(node)),
                    "visual_title": preview(node.get("visual_title", ""), 80),
                    "qa_evidence": preview(node.get("qa_evidence", ""), 160),
                    "crop_image_path": node.get("crop_image_path", ""),
                    "page_image_path": node.get("page_image_path", ""),
                    "source_ref": node.get("source_ref", ""),
                    "content_preview": preview(node.get("content", "")),
                    "rerank_time_ms": round(elapsed_ms if method in {"G2", "G3", "G4"} else 0.0, 3),
                }
            )
    return rows


def group_rankings(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("question_id", ""), row.get("method", ""))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: int(row.get("rank", 0)))
    return grouped


def neighbor_relations(
    node_id: str,
    nodes_by_id: dict[str, dict[str, Any]],
    graph: nx.Graph,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if node_id not in graph:
        return []
    relations: list[dict[str, Any]] = []
    for neighbor in list(graph.neighbors(node_id))[:limit]:
        node = nodes_by_id.get(neighbor, {})
        edge_data = graph.get_edge_data(node_id, neighbor, default={})
        relations.append(
            {
                "node_id": neighbor,
                "node_type": node.get("node_type", ""),
                "page": node.get("page", ""),
                "edge_type": edge_data.get("edge_type", "related"),
                "content_preview": preview(node.get("content", ""), 100),
            }
        )
    return relations


def answer_for_question(question: dict[str, Any], top_rows: list[dict[str, Any]]) -> str:
    answer = clean_text(question.get("answer"))
    if answer:
        return answer
    if not top_rows:
        return "No answer available because no evidence was retrieved."
    first = top_rows[0]
    return (
        f"Likely answer evidence is on page {first.get('page')} "
        f"from {first.get('node_type')} node {first.get('node_id')}: "
        f"{first.get('content_preview')}"
    )
