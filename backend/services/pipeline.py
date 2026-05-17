from __future__ import annotations

import json
from pathlib import Path

from backend.config import env_int, env_value
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

from embedding_index import DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_EMBEDDING_DEVICE, DEFAULT_EMBEDDING_MODEL
from pipeline_common import clean_text, normalize_doc_id, write_jsonl
from rerank_lib import answer_for_question, build_graph, rank_question


def run_upload_job(job_id: str, question_text: str, pdf_path: Path, chunk_template: str = "auto") -> None:
    job = job_dir(job_id)
    try:
        write_status(job_id, status="running", stage="parse", progress=8)
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

        write_status(job_id, stage="visual", progress=24)
        captioner, visual_caption_provider = build_backend_captioner(job_id)
        visual_max_captions = env_int(
            "RAG_BACKEND_VISUAL_MAX_CAPTIONS",
            env_int("RAG_VISUAL_MAX_CAPTIONS", 0),
        )
        append_log(
            job_id,
            f"Generating visual crops and {visual_caption_provider} captions before embedding.",
        )
        visual_caption_count = visual_evidence.process_document(
            pdf_path,
            nodes,
            job / "visual",
            env_int("RAG_BACKEND_VISUAL_DPI", 120),
            captioner,
            visual_max_captions,
            0,
            set(),
            True,
            "",
        )
        write_status(
            job_id,
            visual_caption_provider=visual_caption_provider,
            visual_caption_model=getattr(captioner, "model_name", ""),
            visual_caption_count=visual_caption_count,
        )
        write_jsonl(job / "nodes.jsonl", nodes)

        write_status(job_id, stage="graph", progress=38)
        append_log(job_id, "正在构建页面、正文、表格、图片之间的图关系")
        edges = build_graph_edges.build_edges(nodes)
        write_jsonl(job / "edges.jsonl", edges)

        write_status(job_id, stage="kg", progress=46)
        append_log(job_id, "Building GraphRAG entity/relation/community index.")
        kg_index = build_kg_index_for_backend(job_id, job, job / "nodes.jsonl", job / "edges.jsonl")
        write_status(
            job_id,
            kg_enabled=bool(kg_index),
            kg_entity_count=len(kg_index.get("entities", {})),
            kg_relation_count=len(kg_index.get("relations", [])),
        )

        write_status(job_id, stage="retrieve", progress=56)
        append_log(job_id, "正在召回候选证据")
        embedding_index = build_embedding_index_for_backend(nodes, job)
        candidate_rows = candidate_rows_for_question(question, nodes, job, embedding_index, kg_index)
        write_csv_dicts(job / "candidates.csv", candidate_rows)

        write_status(job_id, stage="rerank", progress=70)
        append_log(job_id, "正在进行 G4 证据重排序")
        ranking_rows = rank_question(
            question,
            nodes,
            edges,
            top_k=int(env_value("RAG_BACKEND_RERANK_K", "10")),
            candidate_rows=candidate_rows,
            retriever=env_value("RAG_BACKEND_RERANK_RETRIEVER", "fusion"),
            embedding_index=embedding_index,
            embedding_model=env_value("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_cache=env_value("RAG_BACKEND_EMBEDDING_CACHE", str(job / "embeddings")),
            embedding_device=env_value("RAG_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE),
            embedding_batch_size=int(env_value("RAG_EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))),
            hybrid_alpha=float(env_value("RAG_BACKEND_HYBRID_ALPHA", "0.7")),
            kg_index=kg_index,
        )
        write_csv_dicts(job / "reranked.csv", ranking_rows)

        write_status(job_id, stage="chain", progress=84)
        append_log(job_id, "正在生成证据链")
        g4_rows = [row for row in ranking_rows if row.get("method") == "G4"]
        graph = build_graph(nodes, edges)
        nodes_by_id = {node.get("node_id", ""): node for node in nodes if node.get("node_id")}
        steps = chain_builder.build_chain_for_question(
            question,
            g4_rows,
            nodes_by_id,
            graph,
            max_steps=int(env_value("RAG_BACKEND_MAX_STEPS", "5")),
        )
        write_csv_dicts(job / "chain_steps.csv", steps)
        question["answer"] = answer_for_question(question, steps or g4_rows)

        write_status(job_id, stage="card", progress=93)
        append_log(job_id, "正在生成证据卡片")
        card_path = job / "evidence_card.png"
        if steps:
            card_builder.build_card(question, steps, card_path, max_steps=int(env_value("RAG_BACKEND_MAX_STEPS", "5")))

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
