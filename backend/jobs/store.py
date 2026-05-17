from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend.config import JOB_ROOT


JOBS_LOCK = threading.Lock()


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
