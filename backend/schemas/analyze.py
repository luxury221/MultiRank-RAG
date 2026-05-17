from __future__ import annotations

from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str
