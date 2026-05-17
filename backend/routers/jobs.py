from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.jobs.store import job_dir, read_status


router = APIRouter()


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return read_status(job_id)


@router.get("/api/jobs/{job_id}/files/{path:path}")
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
