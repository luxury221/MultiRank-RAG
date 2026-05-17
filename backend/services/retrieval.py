from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from backend.config import ROOT, SCRIPTS_DIR, env_value
from backend.jobs.store import append_log

from embedding_index import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingIndex,
)
from pipeline_common import clean_text, preview
from rerank_lib import load_kg_index, retrieve_candidates


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
        embedding_cache=env_value("RAG_BACKEND_EMBEDDING_CACHE", str((job or ROOT / "outputs") / "embeddings")),
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
    if retriever not in {"embedding", "hybrid", "fusion"} and candidate_retriever not in {
        "embedding",
        "hybrid",
        "fusion",
    }:
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
