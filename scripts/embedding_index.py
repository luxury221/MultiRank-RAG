from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_common import OUTPUT_DIR, clean_text, resolve_path


DEFAULT_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
DEFAULT_EMBEDDING_DEVICE = os.environ.get("RAG_EMBEDDING_DEVICE", "auto")
DEFAULT_EMBEDDING_BATCH_SIZE = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "16"))
DEFAULT_EMBEDDING_CACHE_DIR = OUTPUT_DIR / "embeddings"


def node_embedding_text(node: dict[str, Any]) -> str:
    node_type = clean_text(node.get("node_type")) or "text"
    content = clean_text(node.get("content"))
    return f"{node_type}\n{content}" if content else ""


def _safe_model_name(model_name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", model_name).strip("_") or "embedding_model"


def _nodes_hash(nodes: list[dict[str, Any]], model_name: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    for node in nodes:
        node_id = clean_text(node.get("node_id"))
        if not node_id:
            continue
        hasher.update(b"\nNODE\n")
        hasher.update(node_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(clean_text(node.get("doc_id")).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(node_embedding_text(node).encode("utf-8"))
    return hasher.hexdigest()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class EmbeddingIndex:
    nodes: list[dict[str, Any]]
    node_ids: list[str]
    embeddings: np.ndarray
    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str = DEFAULT_EMBEDDING_DEVICE
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE

    _model: Any | None = None

    @classmethod
    def from_nodes(
        cls,
        nodes: list[dict[str, Any]],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str | Path = DEFAULT_EMBEDDING_CACHE_DIR,
        device: str = DEFAULT_EMBEDDING_DEVICE,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> "EmbeddingIndex":
        valid_nodes = [node for node in nodes if clean_text(node.get("node_id")) and node_embedding_text(node)]
        node_ids = [clean_text(node.get("node_id")) for node in valid_nodes]
        cache_dir = resolve_path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        content_hash = _nodes_hash(valid_nodes, model_name)
        cache_path = cache_dir / f"{_safe_model_name(model_name)}_{content_hash[:16]}.npz"

        embeddings = cls._load_cache(cache_path, model_name, content_hash, node_ids)
        if embeddings is None:
            texts = [node_embedding_text(node) for node in valid_nodes]
            model = cls._load_model(model_name, device)
            embeddings = cls._encode(model, texts, batch_size=batch_size)
            metadata = {
                "model_name": model_name,
                "content_hash": content_hash,
                "count": len(node_ids),
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            }
            np.savez_compressed(
                cache_path,
                embeddings=embeddings.astype("float32"),
                node_ids=np.asarray(node_ids, dtype=str),
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=str),
            )
        return cls(
            nodes=valid_nodes,
            node_ids=node_ids,
            embeddings=embeddings.astype("float32"),
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )

    @staticmethod
    def _load_cache(
        cache_path: Path,
        model_name: str,
        content_hash: str,
        node_ids: list[str],
    ) -> np.ndarray | None:
        if not cache_path.exists():
            return None
        try:
            data = np.load(cache_path, allow_pickle=False)
            metadata = json.loads(str(data["metadata"].item()))
            cached_ids = [str(item) for item in data["node_ids"].tolist()]
            if (
                metadata.get("model_name") == model_name
                and metadata.get("content_hash") == content_hash
                and cached_ids == node_ids
            ):
                return data["embeddings"].astype("float32")
        except Exception:
            return None
        return None

    @staticmethod
    def _load_model(model_name: str, device: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding retriever needs sentence-transformers. "
                "Install it in your environment or run with --retriever lexical."
            ) from exc
        return SentenceTransformer(model_name, device=_resolve_device(device))

    @staticmethod
    def _encode(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype="float32")

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model(self.model_name, self.device)
        return self._model

    def score(self, query: Any, nodes: list[dict[str, Any]] | None = None) -> dict[str, float]:
        query_text = clean_text(query)
        target_ids = self.node_ids if nodes is None else [clean_text(node.get("node_id")) for node in nodes]
        if not query_text or self.embeddings.size == 0:
            return {node_id: 0.0 for node_id in target_ids if node_id}

        query_embedding = self._encode(self.model, [query_text], batch_size=1)[0]
        all_scores = np.matmul(self.embeddings, query_embedding)
        score_by_id = {
            node_id: float(max(0.0, score))
            for node_id, score in zip(self.node_ids, all_scores.tolist())
        }
        return {node_id: score_by_id.get(node_id, 0.0) for node_id in target_ids if node_id}
