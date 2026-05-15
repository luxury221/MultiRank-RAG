from __future__ import annotations

import json
import os
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from embedding_index import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingIndex,
    node_embedding_text,
)
from pipeline_common import as_float, clean_text, preview, read_jsonl, resolve_path, split_multi


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
RETRIEVER_CHOICES = ["fusion", "hybrid", "embedding", "lexical", "bm25", "kg"]
DEFAULT_KG_DIR = os.getenv("RAG_KG_DIR", "outputs/kg")

_CORPUS_FEATURE_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}
_GRAPH_BUILD_CACHE: dict[tuple[int, int, int, int], nx.Graph] = {}
_NODES_BY_ID_CACHE: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
_NODE_REFS_CACHE: dict[int, dict[str, set[tuple[str, str]]]] = {}

KG_CONCRETE_TYPES = {"product", "part", "fault", "image"}
KG_SUPPORT_TYPES = {"action", "policy"}
KG_POLICY_INTENT_NAMES = {
    "return_refund": "\u9000\u6362\u8d27\u9000\u6b3e",
    "invoice": "\u53d1\u7968\u5f00\u7968",
    "shipping_damage": "\u7269\u6d41\u5305\u88c5\u7834\u635f",
    "warranty_repair": "\u4fdd\u4fee\u7ef4\u4fee",
    "troubleshooting": "\u6545\u969c\u6392\u67e5",
    "usage_operation": "\u5b89\u88c5\u4f7f\u7528",
    "spec_parts": "\u89c4\u683c\u914d\u4ef6",
    "safety": "\u5b89\u5168\u8b66\u544a",
}
KG_GENERIC_POLICY_TERMS = {
    "\u552e\u540e",
    "\u5ba2\u670d",
    "\u5546\u54c1",
    "\u670d\u52a1",
    "\u652f\u6301",
    "\u8303\u56f4",
    "\u5982\u4f55",
    "\u600e\u4e48",
    "\u9700\u8981",
    "after-sales",
    "customer service",
    "service",
    "support",
}
KG_GENERIC_ACTION_TERMS = {"use", "operate", "\u4f7f\u7528", "\u64cd\u4f5c"}

BASE_EDGE_TYPE_WEIGHTS = {
    "same_page": 0.05,
    "belongs_to_page": 0.05,
    "text_ref_table": 1.0,
    "text_ref_figure": 1.0,
    "table_caption": 1.2,
    "figure_caption": 1.2,
    "section_title": 0.35,
    "same_section": 0.12,
    "parent_section": 0.45,
    "chunk_sequence": 0.08,
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
    "ocr_text",
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

AFTER_SALES_INTENT_TERMS = {
    "return_refund": (
        "7天",
        "七天",
        "无理由",
        "退货",
        "换货",
        "退换",
        "退款",
        "运费",
        "return",
        "refund",
        "exchange",
    ),
    "invoice": ("发票", "开票", "抬头", "税号", "invoice"),
    "shipping_damage": ("物流", "快递", "运输", "包装破损", "破损", "签收", "包裹", "shipping", "package"),
    "warranty_repair": ("售后", "保修", "维修", "送修", "人为损坏", "质保", "客服", "warranty", "repair", "service"),
    "troubleshooting": (
        "故障",
        "异常",
        "无法",
        "不能",
        "不工作",
        "没反应",
        "报错",
        "重启",
        "漏水",
        "噪音",
        "充电",
        "开机",
        "troubleshoot",
        "error",
        "fault",
    ),
    "usage_operation": ("如何", "怎么", "步骤", "安装", "设置", "清洁", "使用", "连接", "校准", "operation", "setup"),
    "spec_parts": ("规格", "参数", "尺寸", "电压", "容量", "配件", "清单", "型号", "spec", "parts"),
    "safety": ("安全", "警告", "注意", "危险", "儿童", "火灾", "触电", "烫伤", "safety", "warning"),
}

AFTER_SALES_MANUAL_HINTS = (
    "售后",
    "客服",
    "保修",
    "维修",
    "手册",
    "说明书",
    "manual",
    "after-sales",
    "customer service",
    "datafountain_customer_service",
    "manual_qa",
)

DATAFOUNTAIN_PRODUCT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("空调", ("空调", "air conditioner", "air conditioning", " ac ")),
    ("电钻", ("电钻", "drill", "electric drill")),
    ("空气净化器", ("空气净化器", "air purifier")),
    ("吹风机", ("吹风机", "hair dryer", "blow dryer")),
    ("洗碗机", ("洗碗机", "dishwasher", "dish washer")),
    ("健身单车", ("健身单车", "exercise bike", "fitness bike", "stationary bike")),
    ("蒸汽清洁机", ("蒸汽清洁机", "steam cleaner", "steam mop")),
    ("儿童电动摩托车", ("儿童电动摩托车", "kids electric motorcycle", "electric motorcycle", "toy motorcycle")),
    ("冰箱", ("冰箱", "refrigerator", "fridge")),
    ("摩托艇", ("摩托艇", "jetski", "jet ski", "watercraft")),
    ("人体工学椅", ("人体工学椅", "ergonomic chair", "office chair")),
    ("功能键盘", ("功能键盘", "function keyboard", "keyboard")),
    ("烤箱", ("烤箱", "oven")),
    ("相机", ("相机", "camera")),
    ("可编程温控器", ("可编程温控器", "programmable thermostat", "thermostat")),
    ("健身追踪器", ("健身追踪器", "fitness tracker", "activity tracker")),
    ("水泵", ("水泵", "water pump", "pump")),
    ("发电机", ("发电机", "generator")),
    ("VR头显", ("VR头显", "vr headset", "virtual reality headset")),
    ("蓝牙激光鼠标", ("蓝牙激光鼠标", "bluetooth laser mouse", "laser mouse", "mouse")),
    ("耳机", ("earphones", "earbuds", "headphones", "耳机")),
    ("电子书阅读器", ("ereader", "e-reader", "ebook reader", "电子书", "阅读器")),
    ("传真机", ("fax", "fax machine", "传真")),
    ("烤架", ("grill", "barbecue", "bbq")),
    ("座机", ("landline", "handset", "base station", "telephone")),
    ("割草机", ("lawn mower", "mower")),
    ("微波炉", ("over-the-range microwave", "microwave")),
    ("主板", ("motherboard", "bios", "sata", "pci express", "cpu")),
    ("压力锅空气炸锅", ("multi-use pressure cooker", "pressure cooker", "air fryer")),
    ("扫地机/吸尘器", ("vacuum cleaner", "vacuum", "home base")),
    ("雪地摩托", ("snowmobile",)),
    ("电视/收音", ("television", "tv", "radio", "dvd player", "outdoor antenna")),
    ("电动牙刷", ("electric toothbrush", "toothbrush")),
)

DATAFOUNTAIN_EXTRA_PRODUCT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("coffee_machine", ("coffee machine", "coffee maker", "nespresso", "espresso", "lungo")),
    ("boat", ("boat", "anchor light", "sail", "stern light", "navigation light")),
    ("loudspeaker", ("loudspeaker", "speaker", "wireless speaker")),
    ("gps_navigation", ("gps", "navigator", "navigation", "nav", "route guidance")),
    ("robot_vacuum", ("vacuum", "robot vacuum", "roomba", "home base", "full bin sensor", "charging contacts")),
)

SECTION_QUERY_HINTS = {
    "method": ("method", "methods", "methodology", "approach", "\u65b9\u6cd5"),
    "experiment": ("experiment", "experiments", "evaluation", "results", "\u5b9e\u9a8c", "\u7ed3\u679c", "\u8bc4\u4f30"),
    "dataset": ("dataset", "benchmark", "data", "\u6570\u636e\u96c6", "\u6570\u636e"),
    "conclusion": ("conclusion", "discussion", "limitations", "\u7ed3\u8bba", "\u8ba8\u8bba", "\u5c40\u9650"),
    "theory": ("theorem", "lemma", "proof", "definition", "\u5b9a\u7406", "\u5f15\u7406", "\u8bc1\u660e", "\u5b9a\u4e49"),
    "finance": ("regression", "variables", "risk", "robustness", "\u56de\u5f52", "\u53d8\u91cf", "\u98ce\u9669", "\u7a33\u5065"),
    "medical": ("patient", "trial", "outcome", "adverse", "\u60a3\u8005", "\u8bd5\u9a8c", "\u7ed3\u5c40", "\u4e0d\u826f\u4e8b\u4ef6"),
}


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


def _corpus_key(nodes: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(clean_text(node.get("node_id")) for node in nodes)


def _corpus_feature(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    key = _corpus_key(nodes)
    feature = _CORPUS_FEATURE_CACHE.get(key)
    if feature is None:
        feature = {
            "node_ids": list(key),
            "texts": {},
            "lexical": {},
            "bm25": {},
            "node_refs": None,
            "scorable_nodes": None,
        }
        _CORPUS_FEATURE_CACHE[key] = feature
    return feature


def _scorable_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature = _corpus_feature(nodes)
    cached = feature.get("scorable_nodes")
    if cached is None:
        cached = [node for node in nodes if clean_text(node.get("node_id")) and node_embedding_text(node)]
        feature["scorable_nodes"] = cached
    return cached


def _cached_texts_for_nodes(nodes: list[dict[str, Any]], text_key: str) -> list[str]:
    feature = _corpus_feature(nodes)
    texts_by_key = feature["texts"]
    texts = texts_by_key.get(text_key)
    if texts is not None:
        return texts
    if text_key == "retrieval":
        texts = [node_retrieval_text(node) for node in nodes]
    elif text_key == "section":
        texts = [section_route_text(node) for node in nodes]
    elif text_key == "visual":
        texts = [
            clean_text(
                " ".join(
                    [
                        node.get("node_type", ""),
                        node.get("layout_role", ""),
                        node.get("bbox_source", ""),
                        visual_text_for_node(node),
                        preview(node.get("content", ""), 220),
                    ]
                )
            )
            for node in nodes
        ]
    elif text_key == "product":
        texts = [product_route_text(node) for node in nodes]
    else:
        raise ValueError(f"Unknown cached text key: {text_key}")
    texts_by_key[text_key] = texts
    return texts


def _build_lexical_corpus(texts: list[str]) -> dict[str, Any]:
    clean_texts = [clean_text(text) for text in texts]
    if not clean_texts:
        return {"texts": clean_texts, "vectorizer": None, "matrix": None}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        matrix = vectorizer.fit_transform(clean_texts)
        return {"texts": clean_texts, "vectorizer": vectorizer, "matrix": matrix}
    except Exception:
        return {"texts": clean_texts, "vectorizer": None, "matrix": None}


def _lexical_values_from_corpus(query: str, corpus: dict[str, Any]) -> list[float]:
    query = clean_text(query)
    texts = corpus.get("texts") or []
    if not texts:
        return []
    if not query:
        return [0.0 for _ in texts]
    vectorizer = corpus.get("vectorizer")
    matrix = corpus.get("matrix")
    if vectorizer is not None and matrix is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            query_matrix = vectorizer.transform([query])
            sims = cosine_similarity(query_matrix, matrix).ravel()
            return [float(max(0.0, score)) for score in sims]
        except Exception:
            pass
    return [_char_jaccard(query, text) for text in texts]


def _lexical_corpus_for_nodes(nodes: list[dict[str, Any]], text_key: str) -> dict[str, Any]:
    feature = _corpus_feature(nodes)
    corpus_by_key = feature["lexical"]
    corpus = corpus_by_key.get(text_key)
    if corpus is None:
        corpus = _build_lexical_corpus(_cached_texts_for_nodes(nodes, text_key))
        corpus_by_key[text_key] = corpus
    return corpus


def _scores_from_cached_texts(query: str, nodes: list[dict[str, Any]], text_key: str) -> dict[str, float]:
    feature = _corpus_feature(nodes)
    values = _lexical_values_from_corpus(query, _lexical_corpus_for_nodes(nodes, text_key))
    return {
        node_id: score
        for node_id, score in zip(feature["node_ids"], values)
        if node_id
    }


def bm25_tokenize(text: Any) -> list[str]:
    text = clean_text(text).casefold()
    if not text:
        return []
    tokens: list[str] = []
    for match in re.finditer(r"[a-z0-9][a-z0-9_+\-./]{1,}|[\u4e00-\u9fff]+", text):
        token = match.group(0).strip("._-/")
        if not token:
            continue
        if any("\u4e00" <= ch <= "\u9fff" for ch in token):
            chars = [ch for ch in token if "\u4e00" <= ch <= "\u9fff"]
            if len(chars) <= 4:
                tokens.append("".join(chars))
            tokens.extend(chars)
            tokens.extend("".join(chars[i : i + 2]) for i in range(max(0, len(chars) - 1)))
            tokens.extend("".join(chars[i : i + 3]) for i in range(max(0, len(chars) - 2)))
        else:
            tokens.append(token)
            if token.endswith("s") and len(token) > 4:
                tokens.append(token[:-1])
    return tokens


def _build_bm25_corpus(texts: list[str]) -> dict[str, Any]:
    doc_tokens = [bm25_tokenize(text) for text in texts]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    doc_freq: Counter[str] = Counter()
    term_freqs: list[Counter[str]] = []
    for tokens in doc_tokens:
        term_freq = Counter(tokens)
        term_freqs.append(term_freq)
        doc_freq.update(term_freq.keys())
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    return {
        "doc_tokens": doc_tokens,
        "doc_lengths": doc_lengths,
        "term_freqs": term_freqs,
        "doc_freq": doc_freq,
        "avg_doc_length": avg_doc_length,
        "total_docs": len(doc_tokens),
    }


def _bm25_values_from_corpus(
    query: str,
    corpus: dict[str, Any],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    doc_tokens = corpus.get("doc_tokens") or []
    query_terms = bm25_tokenize(query)
    if not doc_tokens:
        return []
    if not query_terms:
        return [0.0 for _ in doc_tokens]

    doc_lengths = corpus.get("doc_lengths") or []
    avg_doc_length = float(corpus.get("avg_doc_length") or 0.0)
    if avg_doc_length <= 0:
        return [0.0 for _ in doc_tokens]

    doc_freq: Counter[str] = corpus.get("doc_freq") or Counter()
    term_freqs = corpus.get("term_freqs") or [Counter(tokens) for tokens in doc_tokens]
    query_counts = Counter(query_terms)
    total_docs = int(corpus.get("total_docs") or len(doc_tokens))
    scores: list[float] = []
    for term_freq, doc_length in zip(term_freqs, doc_lengths):
        score = 0.0
        length_norm = k1 * (1.0 - b + b * doc_length / avg_doc_length)
        for term, query_count in query_counts.items():
            freq = term_freq.get(term, 0)
            if not freq:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            score += query_count * idf * (freq * (k1 + 1.0)) / (freq + length_norm)
        scores.append(float(max(0.0, score)))
    return scores


def _bm25_corpus_for_nodes(nodes: list[dict[str, Any]], text_key: str) -> dict[str, Any]:
    feature = _corpus_feature(nodes)
    corpus_by_key = feature["bm25"]
    corpus = corpus_by_key.get(text_key)
    if corpus is None:
        corpus = _build_bm25_corpus(_cached_texts_for_nodes(nodes, text_key))
        corpus_by_key[text_key] = corpus
    return corpus


def _bm25_scores_from_cached_texts(query: str, nodes: list[dict[str, Any]], text_key: str) -> dict[str, float]:
    feature = _corpus_feature(nodes)
    values = _bm25_values_from_corpus(query, _bm25_corpus_for_nodes(nodes, text_key))
    return {
        node_id: score
        for node_id, score in zip(feature["node_ids"], values)
        if node_id
    }


def bm25_similarity(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not texts:
        return []
    return _bm25_values_from_corpus(query, _build_bm25_corpus(texts), k1=k1, b=b)


def _bm25_scores_from_texts(query: str, nodes: list[dict[str, Any]], texts: list[str]) -> dict[str, float]:
    values = bm25_similarity(query, texts)
    return {
        clean_text(node.get("node_id")): score
        for node, score in zip(nodes, values)
        if clean_text(node.get("node_id"))
    }


def _safe_json_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(clean_text(item) for item in value if clean_text(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return clean_text(value)


def load_kg_index(kg_dir: str | Path = DEFAULT_KG_DIR) -> dict[str, Any]:
    if not clean_text(kg_dir):
        return {}
    root = resolve_path(kg_dir)
    entity_path = root / "entities.jsonl"
    relation_path = root / "relations.jsonl"
    profile_path = root / "product_profiles.jsonl"
    if not entity_path.exists() or not relation_path.exists():
        return {}

    entities = read_jsonl(entity_path)
    relations = read_jsonl(relation_path)
    profiles = read_jsonl(profile_path) if profile_path.exists() else []
    entity_by_id = {clean_text(row.get("entity_id")): row for row in entities if clean_text(row.get("entity_id"))}
    node_entities: dict[str, set[str]] = defaultdict(set)
    entity_terms: list[tuple[str, str, tuple[str, ...], str]] = []
    for row in entities:
        entity_id = clean_text(row.get("entity_id"))
        if not entity_id:
            continue
        name = clean_text(row.get("name"))
        aliases = tuple(alias for alias in split_multi(_safe_json_text(row.get("aliases"))) if alias)
        if isinstance(row.get("aliases"), list):
            aliases = tuple(clean_text(alias) for alias in row.get("aliases", []) if clean_text(alias))
        entity_terms.append((entity_id, name, aliases, clean_text(row.get("entity_type"))))
        node_ids = row.get("node_ids") if isinstance(row.get("node_ids"), list) else split_multi(row.get("node_ids"))
        for node_id in node_ids:
            if clean_text(node_id):
                node_entities[clean_text(node_id)].add(entity_id)

    node_relations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relations:
        node_id = clean_text(row.get("evidence_node_id"))
        if node_id:
            node_relations[node_id].append(row)

    product_profile_by_id = {
        clean_text(row.get("product_id")): row for row in profiles if clean_text(row.get("product_id"))
    }
    return {
        "root": str(root),
        "entities": entity_by_id,
        "entity_terms": entity_terms,
        "relations": relations,
        "node_entities": node_entities,
        "node_relations": node_relations,
        "product_profiles": product_profile_by_id,
    }


def kg_available(kg_index: dict[str, Any] | None) -> bool:
    return bool(kg_index and kg_index.get("entities"))


def _scores_from_texts(query: str, nodes: list[dict[str, Any]], texts: list[str]) -> dict[str, float]:
    values = lexical_similarity(query, texts)
    return {
        clean_text(node.get("node_id")): score
        for node, score in zip(nodes, values)
        if clean_text(node.get("node_id"))
    }


def _embedding_scores_safe(
    question: dict[str, Any],
    nodes: list[dict[str, Any]],
    embedding_index: EmbeddingIndex | None,
    embedding_model: str,
    embedding_cache: str,
    embedding_device: str,
    embedding_batch_size: int,
) -> dict[str, float]:
    if embedding_index is None:
        return {}
    try:
        return embedding_index.score(question.get("question", ""), nodes)
    except Exception:
        return {}


def section_route_text(node: dict[str, Any]) -> str:
    return clean_text(
        " ".join(
            [
                f"paper_domain:{node.get('paper_domain', '')}",
                f"section:{node.get('section', '')}",
                f"node_type:{node.get('node_type', '')}",
                f"structure_type:{node.get('structure_type', '')}",
                f"chunk_strategy:{node.get('chunk_strategy', '')}",
                f"layout_role:{node.get('layout_role', '')}",
                f"explicit_refs:{node.get('explicit_refs', '')}",
                preview(node.get("content", ""), 240),
            ]
        )
    )


def section_structure_scores(question: dict[str, Any], nodes: list[dict[str, Any]]) -> dict[str, float]:
    query = clean_text(question.get("question", ""))
    blob = _question_blob(question)
    expanded_terms: list[str] = [query]
    for terms in SECTION_QUERY_HINTS.values():
        if any(_term_in_text(term, blob) for term in terms):
            expanded_terms.extend(terms)
    expanded_query = " ".join(dict.fromkeys(term for term in expanded_terms if term))
    raw_scores = _scores_from_cached_texts(expanded_query, nodes, "section")
    scores: dict[str, float] = {}
    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if not node_id:
            continue
        bonus = 0.0
        node_type = clean_text(node.get("node_type"))
        structure_type = clean_text(node.get("structure_type"))
        if node_type == "title":
            bonus += 0.08
        if structure_type and structure_type != "section_title":
            bonus += 0.08
        section = clean_text(node.get("section"))
        if section and _term_in_text(section, blob):
            bonus += 0.12
        scores[node_id] = raw_scores.get(node_id, 0.0) + bonus
    return scores


def reference_route_scores(question: dict[str, Any], nodes: list[dict[str, Any]]) -> dict[str, float]:
    question_refs = extract_document_refs(question.get("question", ""))
    node_refs_by_id = _node_refs_for_nodes(nodes)
    scores: dict[str, float] = {}
    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if not node_id:
            continue
        node_refs = node_refs_by_id.get(node_id, set())
        hits = node_refs & question_refs if question_refs else set()
        node_type = clean_text(node.get("node_type")) or "text"
        score = 0.0
        if hits:
            score = _reference_weight(node_type, hits)
        elif question_refs and clean_text(node.get("explicit_refs")):
            score = 0.2
        scores[node_id] = score
    return scores


def visual_route_scores(question: dict[str, Any], nodes: list[dict[str, Any]]) -> dict[str, float]:
    intent = question_intent(question)
    if not intent["visual"]:
        return {clean_text(node.get("node_id")): 0.0 for node in nodes if clean_text(node.get("node_id"))}
    raw_scores = _scores_from_cached_texts(question.get("question", ""), nodes, "visual")
    scores: dict[str, float] = {}
    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if not node_id:
            continue
        node_type = clean_text(node.get("node_type"))
        bonus = 0.0
        if intent["table"] and node_type == "table":
            bonus += 0.35
        if intent["figure"] and node_type in {"figure", "caption"}:
            bonus += 0.35
        if intent["visual"] and node_type in VISUAL_NODE_TYPES:
            bonus += 0.15
        if clean_text(node.get("bbox")):
            bonus += 0.05
        if node_has_visual_caption(node):
            bonus += 0.08
        scores[node_id] = raw_scores.get(node_id, 0.0) + bonus
    return scores


def layout_route_scores(question: dict[str, Any], nodes: list[dict[str, Any]]) -> dict[str, float]:
    intent = question_intent(question)
    if not intent["location"]:
        return {clean_text(node.get("node_id")): 0.0 for node in nodes if clean_text(node.get("node_id"))}
    scores: dict[str, float] = {}
    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if not node_id:
            continue
        score = 0.0
        if clean_text(node.get("bbox")):
            score += 0.35
        if clean_text(node.get("page_image_path")):
            score += 0.15
        if clean_text(node.get("crop_image_path")):
            score += 0.2
        if clean_text(node.get("layout_parser")):
            score += 0.1
        scores[node_id] = score
    return scores


def question_route_profile(question: dict[str, Any]) -> dict[str, Any]:
    intent = question_intent(question)
    sales = after_sales_intents(question)
    products = matched_product_groups(question)
    primary = "general"
    if sales:
        strongest = max(sales.items(), key=lambda item: item[1])[0]
        if strongest in {"return_refund", "invoice", "shipping_damage", "warranty_repair"} and not products:
            primary = "policy"
        elif strongest in {"troubleshooting"}:
            primary = "troubleshooting"
        elif strongest in {"usage_operation", "spec_parts", "safety"}:
            primary = "manual_visual" if intent["visual"] else "manual"
        else:
            primary = "after_sales"
    elif intent["table"] or intent["figure"] or intent["cross"] or intent["location"]:
        primary = "visual"
    elif intent["text_fact"]:
        primary = "text_fact"
    return {
        "primary": primary,
        "after_sales": bool(sales),
        "sales_intents": sales,
        "has_product": bool(products),
        **intent,
    }


def kg_fusion_route_weight(question: dict[str, Any], kg_index: dict[str, Any] | None) -> float:
    context = kg_query_context(question, kg_index)
    if not context["entities"]:
        return 0.0
    if context["policies"] and not context["concrete"] and not context["actions"]:
        return 0.42
    if context["concrete"] and (context["actions"] or context["policies"]):
        return 0.72
    if context["concrete"]:
        return 0.45
    return 0.0


def route_weights_for_question(
    question: dict[str, Any],
    available_routes: dict[str, dict[str, float]],
    kg_index: dict[str, Any] | None = None,
) -> dict[str, float]:
    intent = question_intent(question)
    profile = question_route_profile(question)
    weights = {
        "lexical": 1.12,
        "bm25": 1.28,
        "embedding": 0.9,
        "product": 0.8 if matched_product_groups(question) else 0.0,
        "kg": min(0.22, kg_fusion_route_weight(question, kg_index)),
        "section": 0.55,
        "reference": 0.95 if extract_document_refs(question.get("question", "")) else 0.35,
        "visual": 0.18,
        "layout": 0.08,
    }
    primary = profile["primary"]
    if primary == "policy":
        weights["lexical"] = 1.05
        weights["bm25"] = 1.25
        weights["embedding"] = 0.85
        weights["product"] = 0.0
        weights["section"] = 0.9
        weights["visual"] = 0.08
        weights["layout"] = 0.05
    elif primary in {"manual", "troubleshooting"}:
        weights["lexical"] = 1.16
        weights["bm25"] = 1.34
        weights["embedding"] = 0.88
        weights["product"] = max(weights["product"], 0.55 if profile["has_product"] else 0.18)
        weights["section"] = 0.62
        weights["visual"] = max(weights["visual"], 0.28)
    elif primary == "manual_visual":
        weights["lexical"] = 1.08
        weights["bm25"] = 1.22
        weights["embedding"] = 0.86
        weights["visual"] = 0.55
        weights["layout"] = 0.22
        weights["reference"] = max(weights["reference"], 0.65)
        weights["section"] = 0.62
    elif primary == "visual":
        weights["visual"] = 0.7
        weights["layout"] = max(weights["layout"], 0.28)
        weights["reference"] = max(weights["reference"], 0.9)
    elif primary == "text_fact":
        weights["visual"] *= 0.25
        weights["layout"] *= 0.3
        weights["bm25"] = max(weights["bm25"], 1.18)
    if intent["table"] or intent["figure"]:
        weights["visual"] = max(weights["visual"], 0.62)
        weights["reference"] = max(weights["reference"], 0.8)
    if intent["cross"]:
        weights["visual"] = max(weights["visual"], 0.55)
        weights["section"] = max(weights["section"], 0.62)
    if intent["location"]:
        weights["layout"] = 0.4
    if intent["text_fact"]:
        weights["visual"] *= 0.35
        weights["layout"] *= 0.4
    return {route: weights.get(route, 0.5) for route in available_routes}


def node_answer_prior(question: dict[str, Any], node: dict[str, Any]) -> float:
    intent = question_intent(question)
    node_id = clean_text(node.get("node_id"))
    node_type = clean_text(node.get("node_type")) or "text"
    structure_type = clean_text(node.get("structure_type"))
    prior = 1.0
    if node_id.startswith("AS_PROFILE_") or structure_type == "manual_profile":
        prior *= 0.48
    if node_type == "title":
        prior *= 0.72
    elif node_type == "page":
        prior *= 0.42
    elif node_type == "text":
        prior *= 1.08
    elif node_type in VISUAL_NODE_TYPES:
        if intent["table"] or intent["figure"] or intent["location"] or intent["cross"]:
            prior *= 1.02 if node_has_visual_caption(node) else 0.92
        else:
            prior *= 0.72
    if is_after_sales_question(question) and node_type in VISUAL_NODE_TYPES:
        prior *= 0.55
    return prior


def apply_node_answer_priors(
    question: dict[str, Any],
    nodes: list[dict[str, Any]],
    scores: dict[str, float],
) -> dict[str, float]:
    adjusted: dict[str, float] = {}
    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if node_id:
            adjusted[node_id] = scores.get(node_id, 0.0) * node_answer_prior(question, node)
    return normalize_scores(adjusted, list(adjusted))


def rrf_fuse_scores(
    route_scores: dict[str, dict[str, float]],
    node_ids: list[str],
    route_weights: dict[str, float],
    rrf_k: int = 60,
) -> dict[str, float]:
    fused = {node_id: 0.0 for node_id in node_ids}
    for route, scores in route_scores.items():
        normalized = normalize_scores(scores, node_ids)
        ranking = [
            (node_id, normalized.get(node_id, 0.0))
            for node_id in node_ids
            if normalized.get(node_id, 0.0) > 0.0
        ]
        ranking.sort(key=lambda item: item[1], reverse=True)
        weight = route_weights.get(route, 0.5)
        for rank, (node_id, score) in enumerate(ranking, start=1):
            fused[node_id] += weight * (1.0 / (rrf_k + rank)) * (0.5 + 0.5 * score)
    return normalize_scores(fused, node_ids)


def fusion_scores(
    question: dict[str, Any],
    nodes: list[dict[str, Any]],
    embedding_index: EmbeddingIndex | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: str = str(DEFAULT_EMBEDDING_CACHE_DIR),
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    kg_index: dict[str, Any] | None = None,
) -> dict[str, float]:
    nodes = _scorable_nodes(nodes)
    node_ids = list(_corpus_feature(nodes)["node_ids"])
    if not nodes:
        return {}
    route_scores: dict[str, dict[str, float]] = {
        "lexical": _scores_from_cached_texts(question.get("question", ""), nodes, "retrieval"),
        "bm25": _bm25_scores_from_cached_texts(question.get("question", ""), nodes, "retrieval"),
        "product": product_route_scores(question, nodes),
        "kg": kg_route_scores(question, nodes, kg_index),
        "section": section_structure_scores(question, nodes),
        "reference": reference_route_scores(question, nodes),
        "visual": visual_route_scores(question, nodes),
        "layout": layout_route_scores(question, nodes),
    }
    embedding_scores = _embedding_scores_safe(
        question,
        nodes,
        embedding_index,
        embedding_model,
        embedding_cache,
        embedding_device,
        embedding_batch_size,
    )
    if embedding_scores:
        route_scores["embedding"] = embedding_scores
    route_scores = {
        route: scores
        for route, scores in route_scores.items()
        if any(score > 0 for score in scores.values())
    }
    if not route_scores:
        return {node_id: 0.0 for node_id in node_ids}
    fused = rrf_fuse_scores(route_scores, node_ids, route_weights_for_question(question, route_scores, kg_index))
    return apply_node_answer_priors(question, nodes, fused)


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
    kg_index: dict[str, Any] | None = None,
) -> dict[str, float]:
    nodes = _scorable_nodes(nodes)
    if not nodes:
        return {}

    retriever = (retriever or "lexical").lower()
    if retriever == "fusion":
        return fusion_scores(
            question,
            nodes,
            embedding_index=embedding_index,
            embedding_model=embedding_model,
            embedding_cache=embedding_cache,
            embedding_device=embedding_device,
            embedding_batch_size=embedding_batch_size,
            kg_index=kg_index,
        )
    if retriever == "bm25":
        return _bm25_scores_from_cached_texts(question.get("question", ""), nodes, "retrieval")
    if retriever == "kg":
        return kg_route_scores(question, nodes, kg_index)
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
        lexical_scores = _scores_from_cached_texts(question.get("question", ""), nodes, "retrieval")
        node_ids = list(_corpus_feature(nodes)["node_ids"])
        embedding_norm = normalize_scores(embedding_scores, node_ids)
        lexical_norm = normalize_scores(lexical_scores, node_ids)
        alpha = min(1.0, max(0.0, hybrid_alpha))
        return {
            node_id: alpha * embedding_norm.get(node_id, 0.0) + (1.0 - alpha) * lexical_norm.get(node_id, 0.0)
            for node_id in node_ids
        }
    if retriever != "lexical":
        raise ValueError(f"Unknown retriever: {retriever}. Expected one of {', '.join(RETRIEVER_CHOICES)}.")

    return _scores_from_cached_texts(question.get("question", ""), nodes, "retrieval")


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
    cache_key = (id(nodes), id(edges), len(nodes), len(edges))
    cached = _GRAPH_BUILD_CACHE.get(cache_key)
    if cached is not None:
        return cached
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
    _GRAPH_BUILD_CACHE[cache_key] = graph
    return graph


def nodes_by_id_cached(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    key = _corpus_key(nodes)
    cached = _NODES_BY_ID_CACHE.get(key)
    if cached is None:
        cached = {clean_text(node.get("node_id")): node for node in nodes if clean_text(node.get("node_id"))}
        _NODES_BY_ID_CACHE[key] = cached
    return cached


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


def _node_refs_for_nodes(nodes: list[dict[str, Any]]) -> dict[str, set[tuple[str, str]]]:
    feature = _corpus_feature(nodes)
    cached = feature.get("node_refs")
    if cached is not None:
        return cached
    refs_by_id: dict[str, set[tuple[str, str]]] = {}
    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if not node_id:
            continue
        content = f"{node.get('content', '')} {node.get('source_ref', '')} {node.get('explicit_refs', '')}"
        refs_by_id[node_id] = extract_document_refs(content)
    feature["node_refs"] = refs_by_id
    return refs_by_id


def _node_refs_for_nodes_by_id(
    nodes_by_id: dict[str, dict[str, Any]],
) -> dict[str, set[tuple[str, str]]]:
    cache_key = id(nodes_by_id)
    cached = _NODE_REFS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    refs_by_id: dict[str, set[tuple[str, str]]] = {}
    for node_id, node in nodes_by_id.items():
        content = node.get("content", "")
        text = f"{content} {node.get('source_ref', '')}"
        refs_by_id[node_id] = set() if _looks_like_toc_entry(content) else extract_document_refs(text)
    _NODE_REFS_CACHE[cache_key] = refs_by_id
    return refs_by_id


def _looks_like_toc_entry(text: Any) -> bool:
    text = clean_text(text)
    if "···" in text:
        return True
    return bool(re.search(r"\.{4,}\s*\d+\s*$", text))


def _question_blob(question: dict[str, Any]) -> str:
    return clean_text(f"{question.get('question_type', '')} {question.get('question', '')}").casefold()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_term_in_text(term, text) for term in terms)


def _term_in_text(term: str, text: str) -> bool:
    term = clean_text(term).casefold()
    text = clean_text(text).casefold()
    if not term or not text:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in term):
        return term in text
    pattern = r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"\s+") + r"s?(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _alias_in_text(alias: str, text: str) -> bool:
    alias = clean_text(alias).casefold()
    if not alias:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in alias):
        return alias in text
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None


def matched_product_groups(question: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    blob = f" {_question_blob(question)} "
    matches: list[tuple[str, tuple[str, ...]]] = []
    for canonical, aliases in (*DATAFOUNTAIN_PRODUCT_ALIASES, *DATAFOUNTAIN_EXTRA_PRODUCT_ALIASES):
        if _alias_in_text(canonical, blob) or any(_alias_in_text(alias, blob) for alias in aliases):
            matches.append((canonical, aliases))
    return matches


def node_retrieval_text(node: dict[str, Any]) -> str:
    return clean_text(
        " ".join(
            [
                node.get("doc_id", ""),
                node.get("product_category", ""),
                node.get("service_intents", ""),
                node.get("section", ""),
                node.get("source_ref", ""),
                node.get("previous_chunk_preview", ""),
                node.get("content", ""),
                node.get("next_chunk_preview", ""),
                node.get("searchable_text", ""),
                visual_text_for_node(node),
            ]
        )
    )


def product_route_text(node: dict[str, Any]) -> str:
    return clean_text(
        " ".join(
            [
                node.get("doc_id", ""),
                node.get("product_category", ""),
                node.get("section", ""),
                node.get("source_ref", ""),
                node.get("content", ""),
                node.get("visual_title", ""),
                node.get("key_objects", ""),
                node.get("qa_evidence", ""),
                node.get("visual_caption", ""),
            ]
        )
    ).casefold()


def product_route_scores(question: dict[str, Any], nodes: list[dict[str, Any]]) -> dict[str, float]:
    matches = matched_product_groups(question)
    scores: dict[str, float] = {}
    if not matches:
        return {clean_text(node.get("node_id")): 0.0 for node in nodes if clean_text(node.get("node_id"))}

    query_blob = f" {_question_blob(question)} "
    product_texts = _cached_texts_for_nodes(nodes, "product")
    for node, blob in zip(nodes, product_texts):
        node_id = clean_text(node.get("node_id"))
        if not node_id:
            continue
        doc = clean_text(node.get("doc_id")).casefold()
        section_source = clean_text(f"{node.get('section', '')} {node.get('source_ref', '')}").casefold()
        node_type = clean_text(node.get("node_type"))
        best = 0.0
        for canonical, aliases in matches:
            terms = (canonical, *aliases)
            query_terms = tuple(term for term in terms if _alias_in_text(term, query_blob))
            if not query_terms:
                continue
            score = 0.0
            if any(_alias_in_text(term, doc) for term in query_terms):
                score += 0.58
            if any(_alias_in_text(term, section_source) for term in query_terms):
                score += 0.42
            if any(_alias_in_text(term, blob) for term in query_terms):
                score += 0.26
            if node_type in VISUAL_NODE_TYPES and visual_text_for_node(node):
                score += 0.06
            best = max(best, min(1.0, score))
        scores[node_id] = best
    return scores


def kg_policy_names_for_question(question: dict[str, Any]) -> set[str]:
    blob = _question_blob(question)
    names = {
        KG_POLICY_INTENT_NAMES[intent]
        for intent, score in after_sales_intents(question).items()
        if score >= 0.65 and intent in KG_POLICY_INTENT_NAMES
    }
    for name in KG_POLICY_INTENT_NAMES.values():
        if _term_in_text(name, blob):
            names.add(name)
    return names


def kg_entity_type(kg_index: dict[str, Any], entity_id: str) -> str:
    return clean_text(kg_index.get("entities", {}).get(entity_id, {}).get("entity_type"))


def kg_query_context(question: dict[str, Any], kg_index: dict[str, Any] | None) -> dict[str, set[str]]:
    if not kg_available(kg_index):
        return {"entities": set(), "concrete": set(), "actions": set(), "policies": set(), "support": set()}
    blob = f" {_question_blob(question)} "
    strong_policy_names = kg_policy_names_for_question(question)
    candidates: list[tuple[str, str, str, list[str]]] = []
    for entity_id, name, aliases, entity_type in kg_index.get("entity_terms", []):
        terms = (name, *aliases)
        matched = [term for term in terms if _alias_in_text(term, blob)]
        if not matched:
            continue
        candidates.append((entity_id, clean_text(entity_type), clean_text(name), matched))

    hits: set[str] = set()
    concrete = {entity_id for entity_id, entity_type, _, _ in candidates if entity_type in KG_CONCRETE_TYPES}
    for entity_id, entity_type, name, matched in candidates:
        matched_terms = {clean_text(term).casefold() for term in matched if clean_text(term)}
        if entity_type == "image":
            if any(len(term) > 4 and _term_in_text(term, blob) for term in matched_terms):
                hits.add(entity_id)
            continue
        if entity_type == "policy":
            has_specific_policy_term = any(term not in KG_GENERIC_POLICY_TERMS for term in matched_terms)
            if name in strong_policy_names or has_specific_policy_term:
                hits.add(entity_id)
            continue
        if entity_type == "action":
            has_specific_action = any(term not in KG_GENERIC_ACTION_TERMS for term in matched_terms)
            if concrete and has_specific_action:
                hits.add(entity_id)
            continue
        hits.add(entity_id)

    for entity_id, entity in kg_index.get("entities", {}).items():
        if clean_text(entity.get("entity_type")) == "policy" and clean_text(entity.get("name")) in strong_policy_names:
            hits.add(entity_id)

    concrete = {entity_id for entity_id in hits if kg_entity_type(kg_index, entity_id) in KG_CONCRETE_TYPES}
    actions = {entity_id for entity_id in hits if kg_entity_type(kg_index, entity_id) == "action"}
    policies = {entity_id for entity_id in hits if kg_entity_type(kg_index, entity_id) == "policy"}
    return {
        "entities": hits,
        "concrete": concrete,
        "actions": actions,
        "policies": policies,
        "support": actions | policies,
    }


def kg_query_entities(question: dict[str, Any], kg_index: dict[str, Any] | None) -> set[str]:
    return set(kg_query_context(question, kg_index)["entities"])


def kg_route_scores(
    question: dict[str, Any],
    nodes: list[dict[str, Any]],
    kg_index: dict[str, Any] | None = None,
) -> dict[str, float]:
    node_ids = [clean_text(node.get("node_id")) for node in nodes if clean_text(node.get("node_id"))]
    if not kg_available(kg_index):
        return {node_id: 0.0 for node_id in node_ids}
    context = kg_query_context(question, kg_index)
    query_entities = context["entities"]
    if not query_entities:
        return {node_id: 0.0 for node_id in node_ids}

    concrete_entities = context["concrete"]
    action_entities = context["actions"]
    policy_entities = context["policies"]
    support_entities = context["support"]
    policy_only = bool(policy_entities) and not concrete_entities and not action_entities
    node_entities: dict[str, set[str]] = kg_index.get("node_entities", {})
    node_relations: dict[str, list[dict[str, Any]]] = kg_index.get("node_relations", {})
    scores: dict[str, float] = {}
    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if not node_id:
            continue
        direct_hits = node_entities.get(node_id, set()) & query_entities
        direct_concrete_hits = direct_hits & concrete_entities
        direct_action_hits = direct_hits & action_entities
        direct_policy_hits = direct_hits & policy_entities
        node_type = clean_text(node.get("node_type"))
        structure_type = clean_text(node.get("structure_type"))

        if policy_only:
            score = 0.0
            if direct_policy_hits:
                score = 0.55 + 0.12 * min(2, len(direct_policy_hits))
                if structure_type == "after_sales_policy":
                    score += 0.35
                elif structure_type == "manual_profile" or node_type == "title":
                    score = min(score, 0.18)
                elif node_type in VISUAL_NODE_TYPES:
                    score = min(score, 0.25)
                else:
                    score = min(score, 0.45)
            scores[node_id] = min(1.0, max(0.0, score))
            continue

        score = min(
            0.72,
            0.30 * len(direct_concrete_hits)
            + 0.12 * len(direct_action_hits)
            + 0.08 * len(direct_policy_hits),
        )

        relation_bonus = 0.0
        for relation in node_relations.get(node_id, []):
            source_id = clean_text(relation.get("source_id"))
            target_id = clean_text(relation.get("target_id"))
            relation_type = clean_text(relation.get("relation_type"))
            endpoints = {source_id, target_id}
            endpoint_hits = endpoints & query_entities
            if len(endpoint_hits) >= 2:
                relation_bonus = max(relation_bonus, 0.32)
            elif endpoint_hits and endpoints & concrete_entities:
                relation_bonus = max(relation_bonus, 0.16)
            if relation_type == "product_has_part" and len(endpoint_hits & concrete_entities) >= 2:
                relation_bonus = max(relation_bonus, 0.36)
            elif relation_type == "action_targets_part" and (endpoints & action_entities) and (endpoints & concrete_entities):
                relation_bonus = max(relation_bonus, 0.34)
            elif relation_type == "fault_solved_by_action" and (endpoints & action_entities) and (endpoints & concrete_entities):
                relation_bonus = max(relation_bonus, 0.32)
            elif relation_type in {"image_depicts_part", "image_illustrates_action"} and endpoints & (concrete_entities | action_entities):
                relation_bonus = max(relation_bonus, 0.26 if node_type in VISUAL_NODE_TYPES else 0.18)
            elif relation_type in {"product_supports_action", "policy_applies_to_product"} and (
                endpoints & concrete_entities
            ) and (endpoints & support_entities):
                relation_bonus = max(relation_bonus, 0.2)
        score += relation_bonus

        if node_type in VISUAL_NODE_TYPES and score > 0 and node_has_visual_caption(node):
            score += 0.08
        if structure_type == "after_sales_policy" and direct_policy_hits:
            score += 0.12
        if structure_type == "manual_profile" and not direct_concrete_hits:
            score *= 0.4
        scores[node_id] = min(1.0, max(0.0, score))
    return scores


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


def after_sales_intents(question: dict[str, Any]) -> dict[str, float]:
    blob = _question_blob(question)
    scores: dict[str, float] = {}
    for intent, terms in AFTER_SALES_INTENT_TERMS.items():
        hits = sum(1 for term in terms if _term_in_text(term, blob))
        if hits:
            scores[intent] = min(1.0, 0.35 + 0.18 * hits)
    if not scores and _contains_any(blob, AFTER_SALES_MANUAL_HINTS):
        scores["general_after_sales"] = 0.45
    return scores


def is_after_sales_question(question: dict[str, Any]) -> bool:
    return bool(after_sales_intents(question))


def node_after_sales_blob(node: dict[str, Any]) -> str:
    return clean_text(
        " ".join(
            [
                node.get("paper_domain", ""),
                node.get("chunk_template", ""),
                node.get("requested_chunk_template", ""),
                node.get("doc_id", ""),
                node.get("section", ""),
                node.get("node_type", ""),
                node.get("source_ref", ""),
                node.get("visual_summary", ""),
                node.get("previous_chunk_preview", ""),
                node.get("content", ""),
                node.get("next_chunk_preview", ""),
            ]
        )
    ).casefold()


def after_sales_domain_scores(
    question: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    candidate_ids: list[str],
) -> dict[str, float]:
    intent_scores = after_sales_intents(question)
    if not intent_scores:
        return {node_id: 0.0 for node_id in candidate_ids}

    query_terms = []
    for intent in intent_scores:
        query_terms.extend(AFTER_SALES_INTENT_TERMS.get(intent, ()))
    query_terms.append(question.get("question", ""))
    raw_query = " ".join(query_terms)
    raw_scores = lexical_similarity(raw_query, [node_after_sales_blob(nodes_by_id.get(node_id, {})) for node_id in candidate_ids])

    scores: dict[str, float] = {}
    for node_id, semantic in zip(candidate_ids, raw_scores):
        node = nodes_by_id.get(node_id, {})
        blob = node_after_sales_blob(node)
        node_type = clean_text(node.get("node_type")) or "text"
        matched_intents = [
            intent
            for intent, terms in AFTER_SALES_INTENT_TERMS.items()
            if any(_term_in_text(term, blob) for term in terms)
        ]
        policy_or_manual_bonus = 0.0
        if _contains_any(blob, AFTER_SALES_MANUAL_HINTS):
            policy_or_manual_bonus += 0.12
        if clean_text(node.get("paper_domain")) == "after_sales_knowledge_base":
            policy_or_manual_bonus += 0.18
        if clean_text(node.get("structure_type")) in {"after_sales_policy", "manual_profile"}:
            policy_or_manual_bonus += 0.12
        if node_type == "title":
            policy_or_manual_bonus += 0.04
        if matched_intents:
            strongest = max(intent_scores.get(intent, 0.0) for intent in matched_intents)
            policy_or_manual_bonus += 0.25 * strongest
        scores[node_id] = max(0.0, 0.65 * semantic + policy_or_manual_bonus)
    return scores


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
        return 0.08
    if intent["cross"] or intent["location"]:
        return 0.05
    if intent["visual"]:
        return 0.035
    return 0.015


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


def chain_coherence_scores(
    question: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    graph: nx.Graph,
    candidate_ids: list[str],
    sim_scores: dict[str, float],
) -> dict[str, float]:
    if not candidate_ids:
        return {}
    intent = question_intent(question)
    after_sales = after_sales_intents(question)
    scores: dict[str, float] = {}
    strong_edges = {
        "text_ref_table",
        "text_ref_figure",
        "table_caption",
        "figure_caption",
        "section_title",
        "parent_section",
    }
    useful_sequence_edges = {"chunk_sequence", "same_section", "same_page"}

    for node_id in candidate_ids:
        node = nodes_by_id.get(node_id, {})
        if _looks_like_toc_entry(node.get("content", "")):
            scores[node_id] = 0.0
            continue
        if node_id not in graph:
            scores[node_id] = 0.0
            continue

        node_doc = clean_text(node.get("doc_id"))
        node_section = clean_text(node.get("section"))
        node_type = clean_text(node.get("node_type")) or "text"
        neighbor_types: set[str] = set()
        relation_score = 0.0
        semantic_support = 0.0
        same_context_support = 0.0

        for neighbor in graph.neighbors(node_id):
            neighbor_node = nodes_by_id.get(neighbor, {})
            if not neighbor_node or _looks_like_toc_entry(neighbor_node.get("content", "")):
                continue
            neighbor_type = clean_text(neighbor_node.get("node_type")) or "text"
            neighbor_types.add(neighbor_type)
            edge_data = graph.get_edge_data(node_id, neighbor, default={})
            edge_types = set(edge_data.get("edge_types") or [edge_data.get("edge_type", "related")])
            edge_weight = as_float(edge_data.get("weight"), 1.0)
            if edge_types & strong_edges:
                relation_score += 0.22 * min(edge_weight, 2.0)
            elif edge_types & useful_sequence_edges:
                relation_score += 0.08 * min(edge_weight, 2.0)
            semantic_support = max(semantic_support, max(sim_scores.get(neighbor, 0.0), 0.0))
            if clean_text(neighbor_node.get("doc_id")) == node_doc:
                same_context_support += 0.03
            if node_section and clean_text(neighbor_node.get("section")) == node_section:
                same_context_support += 0.04

        modality_coverage = 0.0
        if intent["table"] and "table" in neighbor_types:
            modality_coverage += 0.22
        if intent["figure"] and ({"figure", "caption"} & neighbor_types):
            modality_coverage += 0.22
        if intent["cross"] and len(neighbor_types & {"text", "table", "figure", "caption"}) >= 2:
            modality_coverage += 0.25
        if after_sales and ({"title", "text", "figure"} & neighbor_types):
            modality_coverage += 0.12
        if node_type in {"page", "title"} and not after_sales:
            modality_coverage *= 0.65

        scores[node_id] = max(
            0.0,
            min(
                1.0,
                0.45 * min(relation_score, 1.0)
                + 0.3 * semantic_support
                + 0.15 * min(same_context_support, 1.0)
                + modality_coverage,
            ),
        )
    return scores


def chain_signal_weight(question: dict[str, Any], chain_raw: dict[str, float]) -> float:
    if not any(score > 0 for score in chain_raw.values()):
        return 0.0
    intent = question_intent(question)
    if is_after_sales_question(question):
        return 0.09
    if intent["cross"] or intent["table"] or intent["figure"]:
        return 0.08
    if intent["location"]:
        return 0.06
    if intent["text_fact"]:
        return 0.025
    return 0.04


def after_sales_signal_weight(question: dict[str, Any], domain_raw: dict[str, float]) -> float:
    if not after_sales_intents(question) or not any(score > 0 for score in domain_raw.values()):
        return 0.0
    return 0.11


def product_signal_weight(question: dict[str, Any], product_raw: dict[str, float]) -> float:
    if not matched_product_groups(question) or not any(score > 0 for score in product_raw.values()):
        return 0.0
    if is_after_sales_question(question):
        return 0.04
    return 0.08


def kg_signal_weight(question: dict[str, Any], kg_raw: dict[str, float], kg_index: dict[str, Any] | None) -> float:
    if not kg_available(kg_index) or not any(score > 0 for score in kg_raw.values()):
        return 0.0
    context = kg_query_context(question, kg_index)
    if not context["entities"]:
        return 0.0
    if context["policies"] and not context["concrete"] and not context["actions"]:
        return 0.025
    if context["concrete"] and (context["actions"] or context["policies"]):
        return 0.045 if is_after_sales_question(question) else 0.055
    if context["concrete"]:
        return 0.025 if is_after_sales_question(question) else 0.04
    return 0.0


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
    node_refs = _node_refs_for_nodes_by_id(nodes_by_id)
    scores = {node_id: 0.0 for node_id in candidate_ids}
    for node_id, node in nodes_by_id.items():
        refs = node_refs.get(node_id, set())
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
    kg_index: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    doc_id = clean_text(question.get("doc_id"))
    pool = _scorable_nodes(nodes)
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
        kg_index=kg_index,
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
    kg_index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    nodes_by_id = nodes_by_id_cached(nodes)
    graph = build_graph(nodes, edges)
    doc_id = clean_text(question.get("doc_id"))
    similarity_pool = _scorable_nodes(nodes)
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
        kg_index=kg_index,
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
            kg_index=kg_index,
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
    chain_raw = chain_coherence_scores(question, nodes_by_id, graph, candidate_ids, sim_scores)
    chain_norm = normalize_scores(chain_raw, candidate_ids)
    domain_raw = after_sales_domain_scores(question, nodes_by_id, candidate_ids)
    domain_norm = normalize_scores(domain_raw, candidate_ids)
    product_raw = product_route_scores(question, [nodes_by_id[node_id] for node_id in candidate_ids])
    product_norm = normalize_scores(product_raw, candidate_ids)
    kg_raw = kg_route_scores(question, [nodes_by_id[node_id] for node_id in candidate_ids], kg_index)
    kg_norm = normalize_scores(kg_raw, candidate_ids)
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
    chain_weight = chain_signal_weight(question, chain_raw)
    domain_weight = after_sales_signal_weight(question, domain_raw)
    product_weight = product_signal_weight(question, product_raw)
    kg_weight = kg_signal_weight(question, kg_raw, kg_index)
    g4_scores: dict[str, float] = {}
    domain_blend = min(0.22, domain_weight * 2.0) if domain_weight > 0 else 0.0
    for node_id in candidate_ids:
        base = g3_scores.get(node_id, 0.0)
        remaining = max(0.05, 1.0 - base)
        raw_score = (
            base
            + visual_weight * visual_norm.get(node_id, 0.0) * remaining
            + chain_weight * chain_norm.get(node_id, 0.0) * remaining
            + domain_weight * domain_norm.get(node_id, 0.0) * remaining
            + product_weight * product_norm.get(node_id, 0.0) * remaining
            + kg_weight * kg_norm.get(node_id, 0.0) * remaining
        )
        if domain_blend:
            raw_score = (1.0 - domain_blend) * raw_score + domain_blend * domain_norm.get(node_id, 0.0)
        g4_scores[node_id] = min(
            1.0,
            max(0.0, raw_score * node_answer_prior(question, nodes_by_id.get(node_id, {}))),
        )
    g4_scores = normalize_scores(g4_scores, candidate_ids)

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
                    "chain_score": round(chain_norm.get(node_id, 0.0), 6),
                    "domain_score": round(domain_norm.get(node_id, 0.0), 6),
                    "kg_score": round(kg_norm.get(node_id, 0.0), 6),
                    "rerank_profile": (
                        f"visual={visual_weight:.3f};chain={chain_weight:.3f};"
                        f"after_sales={domain_weight:.3f};product={product_weight:.3f};kg={kg_weight:.3f}"
                    ),
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


def _evidence_for_answer(top_rows: list[dict[str, Any]], max_items: int = 6) -> str:
    blocks: list[str] = []
    for index, row in enumerate(top_rows[:max_items], start=1):
        text = clean_text(row.get("content_preview") or row.get("content") or "")
        visual = clean_text(row.get("visual_caption") or row.get("visual_summary") or row.get("qa_evidence") or "")
        reason = clean_text(row.get("reason"))
        parts = [
            f"[{index}] page={row.get('page', '')} type={row.get('node_type', '')} "
            f"score={row.get('score', '')} source={row.get('source_ref', '')}",
        ]
        if text:
            parts.append(f"text: {text}")
        if visual:
            parts.append(f"visual: {visual}")
        if reason:
            parts.append(f"reason: {reason}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def is_mostly_english(text: Any) -> bool:
    text = clean_text(text)
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return letters > max(20, cjk * 2)


def _fallback_answer_for_question(question: dict[str, Any], top_rows: list[dict[str, Any]]) -> str:
    if not top_rows:
        return "No answer available because no evidence was retrieved."
    first = top_rows[0]
    return (
        f"Likely answer evidence is on page {first.get('page')} "
        f"from {first.get('node_type')} node {first.get('node_id')}: "
        f"{first.get('content_preview') or first.get('content') or ''}"
    )


def answer_for_question(question: dict[str, Any], top_rows: list[dict[str, Any]]) -> str:
    answer = clean_text(question.get("answer"))
    if answer:
        return answer
    if not top_rows:
        return _fallback_answer_for_question(question, top_rows)

    try:
        from ark_clients import ArkChatClient, ArkError, get_env

        provider = (os.getenv("RAG_ANSWER_PROVIDER", "").strip() or get_env("RAG_ANSWER_PROVIDER", "ark")).lower()
        if provider in {"", "off", "none", "local", "fallback"}:
            return _fallback_answer_for_question(question, top_rows)
        model = (
            get_env("RAG_ANSWER_MODEL", "")
            or get_env("ARK_TEXT_MODEL_PRO")
            or get_env("ARK_TEXT_MODEL")
            or get_env("ARK_MODEL")
        )
        if not model:
            return _fallback_answer_for_question(question, top_rows)
        chat = ArkChatClient(model=model)
        language_rule = "Answer in English." if is_mostly_english(question.get("question", "")) else "请用中文回答。"
        system_prompt = (
            "You are the final answer generator in a multimodal RAG evidence system. "
            "Answer directly from the supplied evidence. Prefer concrete steps, parts, "
            "conditions, numbers, and visual/table cues. Do not mention internal rank IDs, "
            "G4, rerank, or that evidence is insufficient unless the evidence truly cannot answer."
        )
        user_prompt = f"""Question:
{question.get('question', '')}

Evidence:
{_evidence_for_answer(top_rows)}

Requirements:
1. {language_rule}
2. Give a concise but complete final answer for the user.
3. If the question asks how to operate, use clear steps.
4. If visual or table evidence is relevant, explicitly say what the image/table shows.
5. Keep Chinese answers around 120-300 characters and English answers around 80-180 words."""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return clean_text(
                    chat.complete(
                        system_prompt,
                        user_prompt,
                        temperature=float(os.getenv("RAG_ANSWER_TEMPERATURE", "0.12")),
                        max_tokens=int(os.getenv("RAG_ANSWER_MAX_TOKENS", "700")),
                    )
                )
            except (ArkError, Exception) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        if last_error:
            return _fallback_answer_for_question(question, top_rows)
    except Exception:
        return _fallback_answer_for_question(question, top_rows)
    return _fallback_answer_for_question(question, top_rows)
