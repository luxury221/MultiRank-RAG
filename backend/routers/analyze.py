from __future__ import annotations

import shutil
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from backend.config import env_int, normalize_chunk_template
from backend.jobs.store import job_dir, write_status
from backend.schemas.analyze import AnalyzeResponse
from backend.services.pipeline import run_upload_job
from backend.utils.files import safe_filename
from backend.utils.pdf import count_pdf_pages

from multirank_rag.common import clean_text


router = APIRouter()


@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_pdf(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    question: str = Form(...),
    chunk_template: str = Form("auto"),
    profile: str = Form("live_fullchain"),
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

    page_count = count_pdf_pages(pdf_path)
    max_pages = env_int("RAG_LIVE_MAX_PAGES", 8)
    max_mb = env_int("RAG_UPLOAD_MAX_MB", 20)
    file_size_mb = round(pdf_path.stat().st_size / (1024 * 1024), 2)
    if max_mb > 0 and file_size_mb > max_mb:
        raise HTTPException(
            status_code=413,
            detail=f"现场全链路模式建议上传 {max_mb}MB 以内的小 PDF，当前文件约 {file_size_mb}MB。",
        )
    if page_count and max_pages > 0 and page_count > max_pages:
        raise HTTPException(
            status_code=413,
            detail=f"现场全链路模式建议上传 {max_pages} 页以内的小 PDF，当前文件为 {page_count} 页。",
        )

    write_status(
        job_id,
        status="queued",
        stage="queued",
        progress=0,
        pdf_name=pdf_path.name,
        pdf_pages=page_count,
        file_size_mb=file_size_mb,
        question=question,
        chunk_template=chunk_template,
        profile=profile,
        logs=["任务已创建"],
    )
    background_tasks.add_task(run_upload_job, job_id, question, pdf_path, chunk_template, profile)
    return {"job_id": job_id, "status": "queued"}
