from __future__ import annotations

import shutil
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from backend.config import normalize_chunk_template
from backend.jobs.store import job_dir, write_status
from backend.schemas.analyze import AnalyzeResponse
from backend.services.pipeline import run_upload_job
from backend.utils.files import safe_filename

from multirank_rag.common import clean_text


router = APIRouter()


@router.post("/api/analyze", response_model=AnalyzeResponse)
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
