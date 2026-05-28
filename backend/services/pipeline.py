from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.config import env_bool, env_int, env_value
from backend.jobs.store import append_log, job_dir, write_status
from backend.services.frontend import normalize_ranking_for_frontend, normalize_step_for_frontend, rel_file_url
from backend.services.questions import infer_question_type, write_single_question
from backend.services.retrieval import (
    build_embedding_index_for_backend,
    build_kg_index_for_backend,
    candidate_rows_for_question,
)
from backend.services.script_modules import (
    build_graph_edges,
    card_builder,
    chain_builder,
    chunk_reporter,
    parse_pdf,
    visual_evidence,
)
from backend.services.visual import build_backend_captioner
from backend.utils.files import write_csv_dicts

from multirank_rag.common import clean_text, normalize_doc_id, preview, write_jsonl
from multirank_rag.rerank import answer_for_question, rank_question
from multirank_rag.retrieval import DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_EMBEDDING_DEVICE, DEFAULT_EMBEDDING_MODEL, build_graph


def _effect_priority_enabled(mode: str) -> bool:
    return mode.strip().lower() not in {"0", "false", "no", "off", "fast", "speed", "standard", "baseline", "lite"}


LIVE_PROFILES = {"live", "demo", "presentation", "live_fullchain", "fullchain", "onsite"}


def _is_live_profile(profile: str) -> bool:
    return clean_text(profile).strip().lower() in LIVE_PROFILES


def _profile_env_int(profile: str, live_name: str, backend_name: str, default: int) -> int:
    if _is_live_profile(profile):
        return env_int(live_name, env_int(backend_name, default))
    return env_int(backend_name, default)


def backend_pipeline_options(profile: str = "") -> dict[str, Any]:
    live_profile = _is_live_profile(profile)
    mode_default = "live" if live_profile else "quality"
    mode = env_value("RAG_BACKEND_PIPELINE_MODE", mode_default).strip().lower() or mode_default
    effect_priority = _effect_priority_enabled(mode)
    variant = env_value(
        "RAG_BACKEND_PIPELINE_VARIANT",
        "V5-live-fullchain" if live_profile else ("V5-online-quality" if effect_priority else "V4-online-standard"),
    )
    default_retriever = "lexical" if live_profile else ("multiroute" if effect_priority else "fusion")
    live_candidate_default = env_value("RAG_LIVE_CANDIDATE_RETRIEVER", default_retriever).strip().lower()
    live_rerank_default = env_value("RAG_LIVE_RERANK_RETRIEVER", live_candidate_default).strip().lower()
    candidate_retriever = env_value("RAG_BACKEND_CANDIDATE_RETRIEVER", default_retriever).strip().lower()
    rerank_retriever = env_value("RAG_BACKEND_RERANK_RETRIEVER", candidate_retriever).strip().lower()
    if live_profile:
        candidate_retriever = env_value("RAG_BACKEND_CANDIDATE_RETRIEVER", live_candidate_default).strip().lower()
        rerank_retriever = env_value("RAG_BACKEND_RERANK_RETRIEVER", live_rerank_default).strip().lower()
    return {
        "profile": "live_fullchain" if live_profile else (profile or "default"),
        "live_profile": live_profile,
        "pipeline_mode": mode,
        "pipeline_variant": variant,
        "effect_priority": effect_priority,
        "candidate_retriever": candidate_retriever,
        "rerank_retriever": rerank_retriever,
        "candidate_k": _profile_env_int(profile, "RAG_LIVE_CANDIDATE_K", "RAG_BACKEND_CANDIDATE_K", 24 if live_profile else 50),
        "rerank_k": _profile_env_int(profile, "RAG_LIVE_RERANK_K", "RAG_BACKEND_RERANK_K", 6 if live_profile else 10),
        "max_steps": _profile_env_int(profile, "RAG_LIVE_MAX_STEPS", "RAG_BACKEND_MAX_STEPS", 4 if live_profile else 5),
        "visual_dpi": _profile_env_int(profile, "RAG_LIVE_VISUAL_DPI", "RAG_BACKEND_VISUAL_DPI", 100 if live_profile else 120),
        "visual_max_captions": _profile_env_int(profile, "RAG_LIVE_VISUAL_MAX_CAPTIONS", "RAG_BACKEND_VISUAL_MAX_CAPTIONS", 2 if live_profile else 0),
        "context_expansion": env_bool("RAG_BACKEND_CONTEXT_EXPANSION", effect_priority),
        "adaptive_rerank_boost": env_bool("RAG_BACKEND_ADAPTIVE_RERANK_BOOST", effect_priority),
        "graph_context_boost": env_bool("RAG_BACKEND_GRAPH_CONTEXT_BOOST", effect_priority),
        "evidence_guard": env_bool("RAG_BACKEND_EVIDENCE_GUARD", effect_priority),
        "enhanced_context_edges": env_bool("RAG_BACKEND_ENHANCED_CONTEXT_EDGES", effect_priority),
    }


def _terms(text: Any) -> set[str]:
    value = clean_text(text).lower()
    return {term for term in re.findall(r"[0-9a-zA-Z\u4e00-\u9fff]{2,}", value) if term}


def _lexical_overlap(question_text: str, content: Any) -> float:
    query = clean_text(question_text).lower()
    body = clean_text(content).lower()
    if not query or not body:
        return 0.0
    q_terms = _terms(query)
    b_terms = _terms(body)
    word_score = len(q_terms & b_terms) / max(1, len(q_terms)) if q_terms else 0.0
    q_chars = {char for char in query if "\u4e00" <= char <= "\u9fff"}
    b_chars = {char for char in body if "\u4e00" <= char <= "\u9fff"}
    char_score = len(q_chars & b_chars) / max(1, len(q_chars)) if q_chars else 0.0
    return min(1.0, 0.65 * word_score + 0.35 * char_score)


def _fallback_candidate_rows(
    question: dict[str, str],
    nodes: list[dict[str, Any]],
    top_k: int,
    retriever: str,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for node in nodes:
        node_type = clean_text(node.get("node_type"))
        content = clean_text(node.get("content"))
        if not content or node_type == "page":
            continue
        modality_boost = 0.08 if node_type in {"table", "figure", "caption"} else 0.0
        score = _lexical_overlap(question.get("question", ""), content) + modality_boost
        scored.append((score, node))
    scored.sort(key=lambda item: (item[0], clean_text(item[1].get("node_type")) != "page"), reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, (score, node) in enumerate(scored[: max(1, top_k)], start=1):
        rows.append(
            {
                "question_id": question["question_id"],
                "doc_id": question["doc_id"],
                "question": question["question"],
                "rank": rank,
                "node_id": clean_text(node.get("node_id")),
                "node_type": node.get("node_type", ""),
                "page": node.get("page", ""),
                "score": round(max(score, 1.0 / (rank + 1)), 6),
                "retriever": retriever,
                "embedding_model": "",
                "query_plan": "fallback=lexical_safety_net",
                "query_plan_strategy": "live_fullchain_fallback",
                "required_modalities": "",
                "source_routes": "fallback",
                "route_ranks": str(rank),
                "source_ref": node.get("source_ref", ""),
                "content_preview": preview(node.get("content", ""), 520),
            }
        )
    return rows


def _fallback_ranking_rows(
    question: dict[str, str],
    candidate_rows: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(candidate_rows[: max(1, top_k)], start=1):
        node = nodes_by_id.get(clean_text(row.get("node_id")), {})
        rows.append(
            {
                "question_id": question["question_id"],
                "doc_id": question["doc_id"],
                "question": question["question"],
                "method": "G4",
                "rank": rank,
                "node_id": row.get("node_id", ""),
                "node_type": row.get("node_type", node.get("node_type", "")),
                "page": row.get("page", node.get("page", "")),
                "score": row.get("score", round(1.0 / (rank + 1), 6)),
                "sim_score": row.get("score", 0.0),
                "bridge_score": 0.0,
                "ref_score": 0.0,
                "visual_score": 1.0 if clean_text(node.get("crop_image_path")) else 0.0,
                "chain_score": 0.0,
                "domain_score": 0.0,
                "kg_score": 0.0,
                "model_rerank_score": 0.0,
                "adaptive_route": "live_fullchain_fallback",
                "query_plan": row.get("query_plan", ""),
                "query_plan_strategy": row.get("query_plan_strategy", ""),
                "required_modalities": row.get("required_modalities", ""),
                "answer_requirements": "",
                "rerank_profile": "fallback=rank_from_candidates",
                "source_routes": row.get("source_routes", ""),
                "route_ranks": row.get("route_ranks", ""),
                "has_visual_crop": int(bool(clean_text(node.get("crop_image_path")))),
                "has_visual_caption": int(bool(clean_text(node.get("visual_caption")))),
                "visual_title": preview(node.get("visual_title", ""), 80),
                "qa_evidence": preview(node.get("qa_evidence", ""), 160),
                "crop_image_path": node.get("crop_image_path", ""),
                "page_image_path": node.get("page_image_path", ""),
                "source_ref": row.get("source_ref", node.get("source_ref", "")),
                "content_preview": row.get("content_preview") or preview(node.get("content", ""), 520),
                "rerank_time_ms": 0.0,
            }
        )
    return rows


def _fallback_steps_from_rows(
    question: dict[str, str],
    ranking_rows: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    max_steps: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranking_rows:
        if len(steps) >= max_steps:
            break
        node_id = clean_text(row.get("node_id"))
        node = nodes_by_id.get(node_id)
        if not node or node_id in seen:
            continue
        seen.add(node_id)
        node_type = clean_text(node.get("node_type"))
        role = "table_or_figure" if node_type in {"table", "figure"} else ("caption" if node_type == "caption" else "main_evidence")
        steps.append(
            {
                "question_id": question.get("question_id", ""),
                "doc_id": question.get("doc_id", ""),
                "question_type": question.get("question_type", ""),
                "question": question.get("question", ""),
                "chain_step": len(steps) + 1,
                "role": role,
                "node_id": node_id,
                "node_type": node.get("node_type", ""),
                "page": node.get("page", ""),
                "relation": "live_fullchain_fallback",
                "score": row.get("score", round(1.0 / (len(steps) + 2), 6)),
                "sim_score": row.get("sim_score", row.get("score", 0.0)),
                "bridge_score": row.get("bridge_score", 0.0),
                "ref_score": row.get("ref_score", 0.0),
                "visual_score": row.get("visual_score", 0.0),
                "guard_score": "",
                "source_ref": node.get("source_ref", ""),
                "page_image_path": node.get("page_image_path", ""),
                "crop_image_path": node.get("crop_image_path", ""),
                "bbox": node.get("bbox", ""),
                "bbox_source": node.get("bbox_source", ""),
                "visual_summary": node.get("visual_summary", ""),
                "visual_caption": node.get("visual_caption", ""),
                "reason": "现场全链路兜底：基于召回与重排结果生成可展示证据步骤。",
                "content_preview": preview(node.get("content", ""), 360),
            }
        )
    return steps


def _looks_chinese(text: Any) -> bool:
    value = clean_text(text)
    return sum(1 for char in value if "\u4e00" <= char <= "\u9fff") >= 2


def _local_live_answer(question: dict[str, str], steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "未检索到可用于回答该问题的证据。"
    chinese = _looks_chinese(question.get("question"))
    snippets: list[str] = []
    for index, step in enumerate(steps[:3], start=1):
        text = clean_text(step.get("content_preview") or step.get("visual_caption") or step.get("visual_summary"))
        if text:
            if chinese:
                snippets.append(f"[E{index}] 第{step.get('page', '?')}页{step.get('node_type', '')}: {preview(text, 110)}")
            else:
                snippets.append(f"[E{index}] page {step.get('page', '?')} {step.get('node_type', '')}: {preview(text, 110)}")
    main_text = preview(
        clean_text(steps[0].get("content_preview") or steps[0].get("visual_caption") or steps[0].get("visual_summary")),
        180,
    )
    if chinese:
        evidence_text = "；".join(snippets)
        return f"根据上传文档的证据链，可以直接依据主证据回答：{main_text}。支撑证据包括{evidence_text}。"
    evidence_text = "; ".join(snippets)
    return f"The answer is grounded in the top evidence: {main_text}. Supporting evidence: {evidence_text}."


def run_upload_job(
    job_id: str,
    question_text: str,
    pdf_path: Path,
    chunk_template: str = "auto",
    profile: str = "live_fullchain",
) -> None:
    job = job_dir(job_id)
    try:
        options = backend_pipeline_options(profile)
        write_status(
            job_id,
            status="running",
            stage="parse",
            progress=8,
            profile=options["profile"],
            live_profile=int(options["live_profile"]),
            message="正在解析 PDF 并生成结构化节点",
            pipeline_mode=options["pipeline_mode"],
            pipeline_variant=options["pipeline_variant"],
            effect_priority=int(options["effect_priority"]),
            candidate_retriever=options["candidate_retriever"],
            rerank_retriever=options["rerank_retriever"],
            context_expansion=int(options["context_expansion"]),
            adaptive_rerank_boost=int(options["adaptive_rerank_boost"]),
            graph_context_boost=int(options["graph_context_boost"]),
            evidence_guard=int(options["evidence_guard"]),
            enhanced_context_edges=int(options["enhanced_context_edges"]),
        )
        append_log(
            job_id,
            "Online pipeline: "
            f"{options['pipeline_variant']} "
            f"(candidate={options['candidate_retriever']}, rerank={options['rerank_retriever']}, "
            f"context={int(options['context_expansion'])}, graph_boost={int(options['graph_context_boost'])}, "
            f"guard={int(options['evidence_guard'])}).",
        )
        append_log(job_id, f"正在解析 PDF 并生成模板化 chunk，模板：{chunk_template}")

        doc_id = normalize_doc_id(pdf_path.name)
        question = {
            "question_id": "CUSTOM",
            "doc_id": doc_id,
            "question": question_text,
            "answer": "",
            "question_type": infer_question_type(question_text),
            "gold_node_ids": "",
            "gold_pages": "",
            "gold_modalities": "",
            "evidence_note": "用户上传 PDF 的自定义问题",
        }
        write_single_question(job, question)

        parser_backend = env_value("RAG_PDF_PARSER", "mineru").strip().lower()
        if parser_backend == "native":
            nodes = parse_pdf.pdf_to_nodes(
                pdf_path,
                chunk_size=int(env_value("RAG_BACKEND_CHUNK_SIZE", "900")),
                chunk_template=chunk_template or env_value("RAG_BACKEND_CHUNK_TEMPLATE", "auto"),
            )
        else:
            nodes = parse_pdf.mineru_pdf_to_nodes(
                pdf_path,
                output_dir=job / "mineru",
                chunk_size=int(env_value("RAG_BACKEND_CHUNK_SIZE", "900")),
                chunk_template=chunk_template or env_value("RAG_BACKEND_CHUNK_TEMPLATE", "auto"),
                backend=env_value("MINERU_BACKEND", "pipeline"),
                method=env_value("MINERU_METHOD", "auto"),
                lang=env_value("MINERU_LANG", ""),
                api_url=env_value("MINERU_API_URL", ""),
            )
        write_jsonl(job / "nodes.raw.jsonl", nodes)
        chunk_report_rows = chunk_reporter.build_report(nodes)
        write_csv_dicts(job / "chunk_quality.csv", chunk_report_rows)
        chunk_report = chunk_report_rows[0] if chunk_report_rows else {}
        write_status(job_id, chunk_template=chunk_template, chunk_report=chunk_report)

        write_status(job_id, stage="visual", progress=24, message="正在生成页面截图、视觉裁剪与多模态线索")
        captioner, visual_caption_provider = build_backend_captioner(job_id)
        visual_max_captions = env_int(
            "RAG_BACKEND_VISUAL_MAX_CAPTIONS",
            env_int("RAG_VISUAL_MAX_CAPTIONS", int(options["visual_max_captions"])),
        )
        if options["live_profile"]:
            visual_max_captions = env_int("RAG_LIVE_VISUAL_MAX_CAPTIONS", int(options["visual_max_captions"]))
        append_log(
            job_id,
            f"Generating visual crops and {visual_caption_provider} captions before embedding.",
        )
        try:
            visual_caption_count = visual_evidence.process_document(
                pdf_path,
                nodes,
                job / "visual",
                int(options["visual_dpi"]),
                captioner,
                visual_max_captions,
                0,
                set(),
                True,
                "",
            )
        except Exception as exc:
            visual_caption_count = 0
            append_log(job_id, f"Visual evidence generation failed, continuing with text evidence: {exc}")
        write_status(
            job_id,
            visual_caption_provider=visual_caption_provider,
            visual_caption_model=getattr(captioner, "model_name", ""),
            visual_caption_count=visual_caption_count,
        )
        write_jsonl(job / "nodes.jsonl", nodes)

        write_status(job_id, stage="graph", progress=38, message="正在构建页面、正文、表格、图片之间的图关系")
        append_log(job_id, "正在构建页面、正文、表格、图片之间的图关系")
        edges = build_graph_edges.build_edges(nodes, enhanced_context_edges=bool(options["enhanced_context_edges"]))
        write_jsonl(job / "edges.jsonl", edges)

        write_status(job_id, stage="kg", progress=46, message="正在构建 GraphRAG 实体、关系与社区索引")
        append_log(job_id, "Building GraphRAG entity/relation/community index.")
        kg_index = build_kg_index_for_backend(job_id, job, job / "nodes.jsonl", job / "edges.jsonl")
        write_status(
            job_id,
            kg_enabled=bool(kg_index),
            kg_entity_count=len(kg_index.get("entities", {})),
            kg_relation_count=len(kg_index.get("relations", [])),
        )

        write_status(job_id, stage="retrieve", progress=56, message="正在进行候选证据召回")
        append_log(job_id, "正在召回候选证据")
        embedding_index = build_embedding_index_for_backend(
            nodes,
            job,
            [str(options["candidate_retriever"]), str(options["rerank_retriever"])],
        )
        candidate_rows = candidate_rows_for_question(
            question,
            nodes,
            job,
            embedding_index,
            kg_index,
            candidate_k=int(options["candidate_k"]),
            retriever=str(options["candidate_retriever"]),
            context_expansion=bool(options["context_expansion"]),
        )
        if not candidate_rows:
            append_log(job_id, "Candidate retrieval returned no rows; using live fallback retrieval.")
            candidate_rows = _fallback_candidate_rows(
                question,
                nodes,
                int(options["candidate_k"]),
                str(options["candidate_retriever"]),
            )
        write_csv_dicts(job / "candidates.csv", candidate_rows)

        write_status(job_id, stage="rerank", progress=70, message="正在重排序候选证据")
        append_log(job_id, f"Running {options['pipeline_variant']} evidence rerank.")
        nodes_by_id = {node.get("node_id", ""): node for node in nodes if node.get("node_id")}
        try:
            ranking_rows = rank_question(
                question,
                nodes,
                edges,
                top_k=int(options["rerank_k"]),
                candidate_rows=candidate_rows,
                retriever=str(options["rerank_retriever"]),
                embedding_index=embedding_index,
                embedding_model=env_value("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                embedding_cache=env_value("RAG_BACKEND_EMBEDDING_CACHE", str(job / "embeddings")),
                embedding_device=env_value("RAG_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE),
                embedding_batch_size=int(env_value("RAG_EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))),
                hybrid_alpha=float(env_value("RAG_BACKEND_HYBRID_ALPHA", "0.7")),
                kg_index=kg_index,
                context_expansion=bool(options["context_expansion"]),
                adaptive_rerank_boost=bool(options["adaptive_rerank_boost"]),
                graph_context_boost=bool(options["graph_context_boost"]),
            )
        except Exception as exc:
            append_log(job_id, f"Rerank failed, continuing with candidate-ranked fallback: {exc}")
            ranking_rows = _fallback_ranking_rows(question, candidate_rows, nodes_by_id, int(options["rerank_k"]))
        if not [row for row in ranking_rows if row.get("method") == "G4"]:
            append_log(job_id, "No G4 rows found after rerank; adding live fallback G4 rows.")
            ranking_rows.extend(_fallback_ranking_rows(question, candidate_rows, nodes_by_id, int(options["rerank_k"])))
        write_csv_dicts(job / "reranked.csv", ranking_rows)

        write_status(job_id, stage="chain", progress=84, message="正在组织证据链并生成答案")
        append_log(job_id, "正在生成证据链")
        g4_rows = [row for row in ranking_rows if row.get("method") == "G4"]
        graph = build_graph(nodes, edges)
        steps = chain_builder.build_chain_for_question(
            question,
            g4_rows,
            nodes_by_id,
            graph,
            max_steps=int(options["max_steps"]),
            evidence_guard=bool(options["evidence_guard"]),
        )
        if not steps:
            append_log(job_id, "Evidence chain builder returned no steps; using live fallback steps.")
            steps = _fallback_steps_from_rows(question, g4_rows, nodes_by_id, int(options["max_steps"]))
        write_csv_dicts(job / "chain_steps.csv", steps)
        question["answer"] = answer_for_question(question, steps or g4_rows)
        answer_provider = env_value("RAG_ANSWER_PROVIDER", env_value("RAG_MODEL_PROVIDER", "ark")).strip().lower()
        if answer_provider in {"", "off", "none", "local", "fallback"} and steps:
            question["answer"] = _local_live_answer(question, steps)

        write_status(job_id, stage="card", progress=93, message="正在生成证据卡片")
        append_log(job_id, "正在生成证据卡片")
        card_path = job / "evidence_card.png"
        if steps:
            try:
                card_builder.build_card(question, steps, card_path, max_steps=int(options["max_steps"]))
            except Exception as exc:
                append_log(job_id, f"Evidence card generation failed; returning text evidence instead: {exc}")

        normalized_steps = [normalize_step_for_frontend(job_id, step) for step in steps]
        card_url = rel_file_url(job_id, card_path) if card_path.exists() else ""
        result = {
            "question": {
                "question_id": "CUSTOM",
                "doc_id": doc_id,
                "question": question["question"],
                "answer": question["answer"],
                "question_type": question["question_type"],
                "gold_node_ids": [],
                "gold_pages": [],
                "gold_modalities": [],
                "evidence_note": question["evidence_note"],
                "profile": options["profile"],
                "pipeline_mode": options["pipeline_mode"],
                "pipeline_variant": options["pipeline_variant"],
                "effect_priority": int(options["effect_priority"]),
                "candidate_retriever": options["candidate_retriever"],
                "rerank_retriever": options["rerank_retriever"],
                "candidate_k": int(options["candidate_k"]),
                "rerank_k": int(options["rerank_k"]),
                "context_expansion": int(options["context_expansion"]),
                "adaptive_rerank_boost": int(options["adaptive_rerank_boost"]),
                "graph_context_boost": int(options["graph_context_boost"]),
                "evidence_guard": int(options["evidence_guard"]),
                "enhanced_context_edges": int(options["enhanced_context_edges"]),
                "embedding_provider": env_value("RAG_EMBEDDING_PROVIDER", env_value("RAG_MODEL_PROVIDER", "auto")),
                "embedding_model": env_value("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                "graph_edge_count": len(edges),
                "card_url": card_url,
                "num_steps": len(normalized_steps),
                "quality_status": "pass" if card_url and normalized_steps else "warn",
                "quality_issues": [] if card_url and normalized_steps else ["未能生成完整证据卡片或证据链"],
                "visual_required": 0,
                "visual_node_steps": len(
                    [step for step in normalized_steps if step.get("node_type") in {"table", "figure", "caption"}]
                ),
                "crop_steps": len([step for step in normalized_steps if step.get("crop_url")]),
                "existing_crop_steps": len([step for step in normalized_steps if step.get("crop_url")]),
                "qwen_caption_steps": sum(1 for step in normalized_steps if clean_text(step.get("visual_caption"))),
                "visual_caption_provider": visual_caption_provider,
                "visual_caption_model": getattr(captioner, "model_name", ""),
                "visual_caption_count": visual_caption_count,
                "kg_enabled": bool(kg_index),
                "kg_entity_count": len(kg_index.get("entities", {})),
                "kg_relation_count": len(kg_index.get("relations", [])),
                "source_pages": sorted({str(step.get("page")) for step in normalized_steps if step.get("page")}),
            },
            "steps": normalized_steps,
            "rankings": normalize_ranking_for_frontend(job_id, ranking_rows),
            "chunk_report": chunk_report,
        }
        (job / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        write_status(job_id, status="succeeded", stage="done", progress=100, result=result)
        append_log(job_id, "完成")
    except Exception as exc:
        write_status(job_id, status="failed", stage="failed", error=str(exc), progress=100)
