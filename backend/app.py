from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
JOB_ROOT = ROOT / "outputs" / "upload_jobs"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from embedding_index import (  # noqa: E402
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingIndex,
)
from pipeline_common import clean_text, normalize_doc_id, preview, write_csv, write_jsonl  # noqa: E402
from rerank_lib import answer_for_question, build_graph, load_kg_index, rank_question, retrieve_candidates  # noqa: E402


app = FastAPI(title="Multimodal RAG Evidence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_LOCK = threading.Lock()


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parse_pdf = load_script_module("parse_pdf_script", SCRIPTS_DIR / "01_parse_pdf.py")
build_graph_edges = load_script_module("build_graph_script", SCRIPTS_DIR / "02_build_graph.py")
visual_evidence = load_script_module("visual_evidence_script", SCRIPTS_DIR / "10_build_visual_evidence.py")
chain_builder = load_script_module("chain_builder_script", SCRIPTS_DIR / "09_build_evidence_chains.py")
card_builder = load_script_module("card_builder_script", SCRIPTS_DIR / "11_build_evidence_cards.py")
chunk_reporter = load_script_module("chunk_reporter_script", SCRIPTS_DIR / "14_chunk_quality_report.py")


def safe_filename(filename: str) -> str:
    name = Path(filename or "upload.pdf").name
    stem = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", Path(name).stem).strip("_") or "upload"
    return f"{stem}.pdf"


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return JOB_ROOT / job_id


def write_status(job_id: str, **updates: Any) -> dict[str, Any]:
    path = job_dir(job_id) / "status.json"
    with JOBS_LOCK:
        status: dict[str, Any] = {}
        if path.exists():
            status = json.loads(path.read_text(encoding="utf-8"))
        status.setdefault("job_id", job_id)
        status.update(updates)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status


def read_status(job_id: str) -> dict[str, Any]:
    path = job_dir(job_id) / "status.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def append_log(job_id: str, message: str) -> None:
    status = read_status(job_id)
    logs = list(status.get("logs", []))
    logs.append(message)
    write_status(job_id, logs=logs[-80:], message=message)


def infer_question_type(question: str) -> str:
    text = question.lower()
    if any(term in text for term in ["table", "表格", "表 ", "表中"]):
        return "表格问答"
    if any(term in text for term in ["figure", "fig.", "图", "图片", "曲线", "趋势"]):
        return "图表理解"
    if any(term in text for term in ["跨模态", "结合", "图文", "多模态"]):
        return "跨模态综合"
    return "自定义问题"


def write_single_question(job: Path, question: dict[str, str]) -> Path:
    path = job / "questions.csv"
    fields = [
        "question_id",
        "doc_id",
        "question",
        "answer",
        "question_type",
        "gold_node_ids",
        "gold_pages",
        "gold_modalities",
        "evidence_note",
    ]
    write_csv(path, [question], fields)
    return path


def candidate_rows_for_question(
    question: dict[str, str],
    nodes: list[dict[str, Any]],
    job: Path | None = None,
    embedding_index: EmbeddingIndex | None = None,
    kg_index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidate_k = int(env_value("RAG_BACKEND_CANDIDATE_K", "50"))
    retriever = env_value("RAG_BACKEND_CANDIDATE_RETRIEVER", "fusion")
    candidates, scores = retrieve_candidates(
        question,
        nodes,
        top_k=candidate_k,
        retriever=retriever,
        embedding_index=embedding_index,
        embedding_model=env_value("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_cache=env_value("RAG_BACKEND_EMBEDDING_CACHE", str((job or JOB_ROOT) / "embeddings")),
        embedding_device=env_value("RAG_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE),
        embedding_batch_size=int(env_value("RAG_EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))),
        hybrid_alpha=float(env_value("RAG_BACKEND_HYBRID_ALPHA", "0.7")),
        kg_index=kg_index,
    )
    rows: list[dict[str, Any]] = []
    for rank, node in enumerate(candidates, start=1):
        node_id = clean_text(node.get("node_id"))
        rows.append(
            {
                "question_id": question["question_id"],
                "doc_id": question["doc_id"],
                "question": question["question"],
                "rank": rank,
                "node_id": node_id,
                "node_type": node.get("node_type", ""),
                "page": node.get("page", ""),
                "score": round(scores.get(node_id, 0.0), 6),
                "retriever": retriever,
                "embedding_model": env_value("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
                if retriever in {"embedding", "hybrid", "fusion"}
                else "",
                "source_ref": node.get("source_ref", ""),
                "content_preview": preview(node.get("content", "")),
            }
        )
    return rows


def build_embedding_index_for_backend(nodes: list[dict[str, Any]], job: Path) -> EmbeddingIndex | None:
    retriever = env_value("RAG_BACKEND_RERANK_RETRIEVER", "fusion")
    candidate_retriever = env_value("RAG_BACKEND_CANDIDATE_RETRIEVER", "fusion")
    if retriever not in {"embedding", "hybrid", "fusion"} and candidate_retriever not in {"embedding", "hybrid", "fusion"}:
        return None
    model = env_value("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    provider = env_value("RAG_EMBEDDING_PROVIDER", env_value("RAG_MODEL_PROVIDER", "auto"))
    device = env_value("RAG_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)
    batch_size = int(env_value("RAG_EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE)))
    cache_dir = env_value("RAG_BACKEND_EMBEDDING_CACHE", str(job / "embeddings"))
    return EmbeddingIndex.from_nodes(
        nodes,
        model_name=model,
        provider=provider,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
    )


def backend_kg_enabled() -> bool:
    value = env_value("RAG_BACKEND_ENABLE_KG", env_value("RAG_ENABLE_KG", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def build_kg_index_for_backend(job_id: str, job: Path, nodes_path: Path, edges_path: Path) -> dict[str, Any]:
    if not backend_kg_enabled():
        append_log(job_id, "GraphRAG is disabled for this backend job.")
        return {}
    kg_dir = job / "graphrag"
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "23_build_graphrag.py"),
        "--nodes",
        str(nodes_path),
        "--edges",
        str(edges_path),
        "--output-dir",
        str(kg_dir),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except Exception as exc:
        append_log(job_id, f"GraphRAG build failed: {exc}")
        return {}
    if result.returncode != 0:
        detail = clean_text(result.stderr or result.stdout)
        append_log(job_id, f"GraphRAG build failed: {preview(detail, 360)}")
        return {}
    kg_index = load_kg_index(kg_dir)
    append_log(
        job_id,
        f"GraphRAG enabled: {len(kg_index.get('entities', {}))} entities, "
        f"{len(kg_index.get('relations', []))} relations.",
    )
    return kg_index


def rel_file_url(job_id: str, path: str | Path) -> str:
    if not path:
        return ""
    target = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    base = job_dir(job_id).resolve()
    try:
        rel = target.relative_to(base)
    except ValueError:
        return ""
    return f"http://127.0.0.1:8765/api/jobs/{job_id}/files/{rel.as_posix()}"


def normalize_step_for_frontend(job_id: str, step: dict[str, Any]) -> dict[str, Any]:
    return {
        "chain_step": int(float(step.get("chain_step") or 0)),
        "role": step.get("role", ""),
        "node_id": step.get("node_id", ""),
        "node_type": step.get("node_type", ""),
        "page": int(float(step.get("page") or 0)),
        "relation": step.get("relation", ""),
        "score": float(step.get("score") or 0.0),
        "visual_score": float(step.get("visual_score") or 0.0),
        "source_ref": step.get("source_ref", ""),
        "crop_url": rel_file_url(job_id, step.get("crop_image_path", "")),
        "page_url": rel_file_url(job_id, step.get("page_image_path", "")),
        "visual_summary": preview(step.get("visual_summary", ""), 420),
        "visual_caption": preview(step.get("visual_caption", ""), 420),
        "reason": preview(step.get("reason", ""), 320),
        "content_preview": preview(step.get("content_preview", ""), 520),
    }


def normalize_ranking_for_frontend(job_id: str, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = clean_text(row.get("method"))
        grouped.setdefault(method, []).append(
            {
                "rank": int(float(row.get("rank") or 0)),
                "node_id": row.get("node_id", ""),
                "node_type": row.get("node_type", ""),
                "page": int(float(row.get("page") or 0)),
                "score": float(row.get("score") or 0.0),
                "sim_score": float(row.get("sim_score") or 0.0),
                "bridge_score": float(row.get("bridge_score") or 0.0),
                "ref_score": float(row.get("ref_score") or 0.0),
                "visual_score": float(row.get("visual_score") or 0.0),
                "kg_score": float(row.get("kg_score") or 0.0),
                "model_rerank_score": float(row.get("model_rerank_score") or 0.0),
                "rerank_profile": row.get("rerank_profile", ""),
                "has_visual_crop": int(row.get("has_visual_crop") or 0),
                "has_visual_caption": int(row.get("has_visual_caption") or 0),
                "source_ref": row.get("source_ref", ""),
                "content_preview": preview(row.get("content_preview", ""), 360),
                "crop_url": rel_file_url(job_id, row.get("crop_image_path", "")),
            }
        )
    for method in grouped:
        grouped[method].sort(key=lambda item: item["rank"])
    return grouped


def write_csv_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_chunk_template(value: str) -> str:
    value = clean_text(value).lower() or "auto"
    return value if value in {"auto", "general", "ai", "math", "finance", "medical"} else "auto"


VISUAL_CAPTION_PROVIDERS = {"local", "qwen", "doubao", "xinference", "openai_compatible"}


def env_value(name: str, default: str = "") -> str:
    try:
        from ark_clients import get_env

        return get_env(name, default)
    except Exception:
        return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    try:
        return int(env_value(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_value(name, str(default)))
    except (TypeError, ValueError):
        return default


def default_backend_visual_caption_provider() -> str:
    provider = clean_text(
        env_value("RAG_BACKEND_VISUAL_CAPTION_PROVIDER")
        or env_value("RAG_VISUAL_CAPTION_PROVIDER")
        or ""
    ).lower()
    if provider in VISUAL_CAPTION_PROVIDERS:
        return provider
    model_provider = clean_text(env_value("RAG_MODEL_PROVIDER", "")).lower()
    if model_provider == "xinference" and env_value("XINFERENCE_VISION_MODEL"):
        return "xinference"
    if model_provider in {"openai_compatible", "openai-compatible", "local_openai", "local-server"} and (
        env_value("OPENAI_COMPATIBLE_VISION_MODEL") or env_value("LOCAL_VISION_MODEL")
    ):
        return "openai_compatible"
    return "qwen" if env_value("DASHSCOPE_API_KEY") else "local"


def build_backend_captioner(job_id: str):
    provider = default_backend_visual_caption_provider()
    if provider == "qwen":
        captioner = visual_evidence.QwenVisionCaptioner(
            model_name=env_value("RAG_BACKEND_QWEN_MODEL")
            or env_value("RAG_QWEN_VL_MODEL")
            or visual_evidence.QWEN_DEFAULT_MODEL,
            base_url=env_value("RAG_QWEN_BASE_URL", visual_evidence.QWEN_DEFAULT_BASE_URL),
            api_key_env=env_value("RAG_QWEN_API_KEY_ENV", "DASHSCOPE_API_KEY"),
            timeout=env_float("RAG_QWEN_TIMEOUT", 60.0),
        )
        if captioner.available():
            return captioner, "qwen"
        append_log(job_id, "Qwen visual caption is enabled but the API key is unavailable; using crops only.")
        return visual_evidence.VisualCaptioner(""), "local"
    if provider == "doubao":
        captioner = visual_evidence.ArkVisionCaptioner(
            model_name=env_value("RAG_BACKEND_ARK_VISION_MODEL") or env_value("RAG_ARK_VISION_MODEL", ""),
            base_url=env_value("RAG_ARK_BASE_URL", visual_evidence.ARK_DEFAULT_BASE_URL),
            api_key_env=env_value("RAG_ARK_API_KEY_ENV", "ARK_API_KEY"),
            timeout=env_float("RAG_ARK_TIMEOUT", 60.0),
        )
        if captioner.available():
            return captioner, "doubao"
        append_log(job_id, "Doubao visual caption is enabled but ARK_API_KEY or model is unavailable; using crops only.")
        return visual_evidence.VisualCaptioner(""), "local"
    if provider == "xinference":
        captioner = visual_evidence.QwenVisionCaptioner(
            model_name=env_value("RAG_BACKEND_XINFERENCE_VISION_MODEL")
            or env_value("XINFERENCE_VISION_MODEL")
            or env_value("RAG_VISION_MODEL", ""),
            base_url=env_value("XINFERENCE_BASE_URL", visual_evidence.XINFERENCE_DEFAULT_BASE_URL),
            api_key_env=env_value("XINFERENCE_API_KEY_ENV", "XINFERENCE_API_KEY"),
            timeout=env_float("XINFERENCE_TIMEOUT", 60.0),
            allow_no_api_key=True,
        )
        if captioner.available() and captioner.model_name:
            return captioner, "xinference"
        append_log(job_id, "Xinference visual caption is enabled but no vision model is configured; using crops only.")
        return visual_evidence.VisualCaptioner(""), "local"
    if provider == "openai_compatible":
        captioner = visual_evidence.QwenVisionCaptioner(
            model_name=env_value("RAG_BACKEND_OPENAI_COMPATIBLE_VISION_MODEL")
            or env_value("OPENAI_COMPATIBLE_VISION_MODEL")
            or env_value("LOCAL_VISION_MODEL", ""),
            base_url=env_value(
                "OPENAI_COMPATIBLE_BASE_URL",
                env_value("LOCAL_MODEL_BASE_URL", visual_evidence.OPENAI_COMPATIBLE_DEFAULT_BASE_URL),
            ),
            api_key_env=env_value("OPENAI_COMPATIBLE_API_KEY_ENV", "OPENAI_COMPATIBLE_API_KEY"),
            timeout=env_float("OPENAI_COMPATIBLE_TIMEOUT", 60.0),
            allow_no_api_key=True,
        )
        if captioner.available() and captioner.model_name:
            return captioner, "openai_compatible"
        append_log(job_id, "OpenAI-compatible visual caption is enabled but no vision model is configured; using crops only.")
        return visual_evidence.VisualCaptioner(""), "local"
    caption_model = env_value("RAG_BACKEND_VISUAL_CAPTION_MODEL", "")
    caption_device = env_value("RAG_BACKEND_VISUAL_CAPTION_DEVICE", "auto")
    return visual_evidence.VisualCaptioner(caption_model, caption_device) if caption_model else visual_evidence.VisualCaptioner(""), "local"


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
        rerank_retriever = env_value("RAG_BACKEND_RERANK_RETRIEVER", "fusion")
        ranking_rows = rank_question(
            question,
            nodes,
            edges,
            top_k=int(env_value("RAG_BACKEND_RERANK_K", "10")),
            candidate_rows=candidate_rows,
            retriever=rerank_retriever,
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
                "visual_node_steps": len([step for step in normalized_steps if step.get("node_type") in {"table", "figure", "caption"}]),
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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_pdf(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    question: str = Form(...),
    chunk_template: str = Form("auto"),
) -> dict[str, Any]:
    question = clean_text(question)
    chunk_template = normalize_chunk_template(chunk_template)
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    job_id = uuid.uuid4().hex
    job = job_dir(job_id)
    pdf_dir = job / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / safe_filename(pdf.filename)

    with pdf_path.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)

    write_status(
        job_id,
        status="queued",
        stage="queued",
        progress=0,
        pdf_name=pdf_path.name,
        question=question,
        chunk_template=chunk_template,
        logs=["任务已创建"],
    )
    background_tasks.add_task(run_upload_job, job_id, question, pdf_path, chunk_template)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return read_status(job_id)


@app.get("/api/jobs/{job_id}/files/{path:path}")
def get_job_file(job_id: str, path: str):
    base = job_dir(job_id).resolve()
    target = (base / path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)
