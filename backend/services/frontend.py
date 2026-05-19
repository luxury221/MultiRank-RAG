from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import ROOT
from backend.jobs.store import job_dir

from multirank_rag.common import clean_text, preview


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
