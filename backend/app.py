from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import shutil
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
from rerank_lib import answer_for_question, build_graph, rank_question, retrieve_candidates  # noqa: E402


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
    spec.loader.exec_module(module)
    return module


parse_pdf = load_script_module("parse_pdf_script", SCRIPTS_DIR / "01_parse_pdf.py")
build_graph_edges = load_script_module("build_graph_script", SCRIPTS_DIR / "02_build_graph.py")
visual_evidence = load_script_module("visual_evidence_script", SCRIPTS_DIR / "10_build_visual_evidence.py")
chain_builder = load_script_module("chain_builder_script", SCRIPTS_DIR / "09_build_evidence_chains.py")
card_builder = load_script_module("card_builder_script", SCRIPTS_DIR / "11_build_evidence_cards.py")


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


def candidate_rows_for_question(question: dict[str, str], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_k = int(os.getenv("RAG_BACKEND_CANDIDATE_K", "50"))
    retriever = os.getenv("RAG_BACKEND_CANDIDATE_RETRIEVER", "lexical")
    candidates, scores = retrieve_candidates(question, nodes, top_k=candidate_k, retriever=retriever)
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
                "embedding_model": "",
                "source_ref": node.get("source_ref", ""),
                "content_preview": preview(node.get("content", "")),
            }
        )
    return rows


def build_embedding_index_for_backend(nodes: list[dict[str, Any]], job: Path) -> EmbeddingIndex | None:
    retriever = os.getenv("RAG_BACKEND_RERANK_RETRIEVER", "hybrid")
    if retriever not in {"embedding", "hybrid"}:
        return None
    model = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    device = os.getenv("RAG_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)
    batch_size = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE)))
    cache_dir = os.getenv("RAG_BACKEND_EMBEDDING_CACHE", str(job / "embeddings"))
    return EmbeddingIndex.from_nodes(
        nodes,
        model_name=model,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
    )


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


def run_upload_job(job_id: str, question_text: str, pdf_path: Path) -> None:
    job = job_dir(job_id)
    try:
        write_status(job_id, status="running", stage="parse", progress=8)
        append_log(job_id, "正在解析 PDF 并生成文本、表格、图注节点")

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

        nodes = parse_pdf.pdf_to_nodes(pdf_path, chunk_size=int(os.getenv("RAG_BACKEND_CHUNK_SIZE", "900")))
        write_jsonl(job / "nodes.raw.jsonl", nodes)

        write_status(job_id, stage="visual", progress=24)
        append_log(job_id, "正在为页面和证据节点生成视觉裁剪图")
        captioner = visual_evidence.VisualCaptioner("")
        visual_evidence.process_document(
            pdf_path,
            nodes,
            job / "visual",
            int(os.getenv("RAG_BACKEND_VISUAL_DPI", "120")),
            captioner,
            0,
            0,
            set(),
            True,
            "",
        )
        write_jsonl(job / "nodes.jsonl", nodes)

        write_status(job_id, stage="graph", progress=38)
        append_log(job_id, "正在构建页面、正文、表格、图片之间的图关系")
        edges = build_graph_edges.build_edges(nodes)
        write_jsonl(job / "edges.jsonl", edges)

        write_status(job_id, stage="retrieve", progress=52)
        append_log(job_id, "正在召回候选证据")
        candidate_rows = candidate_rows_for_question(question, nodes)
        write_csv_dicts(job / "candidates.csv", candidate_rows)

        write_status(job_id, stage="rerank", progress=68)
        append_log(job_id, "正在进行 G4 证据重排序")
        rerank_retriever = os.getenv("RAG_BACKEND_RERANK_RETRIEVER", "hybrid")
        embedding_index = build_embedding_index_for_backend(nodes, job)
        ranking_rows = rank_question(
            question,
            nodes,
            edges,
            top_k=int(os.getenv("RAG_BACKEND_RERANK_K", "10")),
            candidate_rows=candidate_rows,
            retriever=rerank_retriever,
            embedding_index=embedding_index,
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_cache=os.getenv("RAG_BACKEND_EMBEDDING_CACHE", str(job / "embeddings")),
            embedding_device=os.getenv("RAG_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE),
            embedding_batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))),
            hybrid_alpha=float(os.getenv("RAG_BACKEND_HYBRID_ALPHA", "0.7")),
        )
        write_csv_dicts(job / "reranked.csv", ranking_rows)

        write_status(job_id, stage="chain", progress=82)
        append_log(job_id, "正在生成证据链")
        g4_rows = [row for row in ranking_rows if row.get("method") == "G4"]
        question["answer"] = answer_for_question(question, g4_rows)
        graph = build_graph(nodes, edges)
        nodes_by_id = {node.get("node_id", ""): node for node in nodes if node.get("node_id")}
        steps = chain_builder.build_chain_for_question(
            question,
            g4_rows,
            nodes_by_id,
            graph,
            max_steps=int(os.getenv("RAG_BACKEND_MAX_STEPS", "5")),
        )
        write_csv_dicts(job / "chain_steps.csv", steps)

        write_status(job_id, stage="card", progress=92)
        append_log(job_id, "正在生成证据卡片")
        card_path = job / "evidence_card.png"
        if steps:
            card_builder.build_card(question, steps, card_path, max_steps=int(os.getenv("RAG_BACKEND_MAX_STEPS", "5")))

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
                "qwen_caption_steps": 0,
                "source_pages": sorted({str(step.get("page")) for step in normalized_steps if step.get("page")}),
            },
            "steps": normalized_steps,
            "rankings": normalize_ranking_for_frontend(job_id, ranking_rows),
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
) -> dict[str, Any]:
    question = clean_text(question)
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
        logs=["任务已创建"],
    )
    background_tasks.add_task(run_upload_job, job_id, question, pdf_path)
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
