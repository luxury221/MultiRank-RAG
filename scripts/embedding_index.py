from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_common import OUTPUT_DIR, clean_text, resolve_path


def _env(name: str, default: str = "") -> str:
    try:
        from ark_clients import get_env

        return get_env(name, default)
    except Exception:
        return os.environ.get(name, default)


DOUBAO_EMBEDDING_VISION_MODEL = "doubao-embedding-vision-250615"
LOCAL_FALLBACK_EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_EMBEDDING_PROVIDER = (
    _env("RAG_EMBEDDING_PROVIDER", "").strip().lower()
    or _env("RAG_MODEL_PROVIDER", "auto").strip().lower()
    or "auto"
)
DEFAULT_EMBEDDING_MODEL = (
    _env("RAG_EMBEDDING_MODEL", "").strip()
    or _env("XINFERENCE_EMBEDDING_MODEL", "").strip()
    or _env("OPENAI_COMPATIBLE_EMBEDDING_MODEL", "").strip()
    or _env("LOCAL_EMBEDDING_MODEL", "").strip()
    or _env("ARK_EMBEDDING_MODEL", "").strip()
    or (DOUBAO_EMBEDDING_VISION_MODEL if _env("ARK_API_KEY", "").strip() else LOCAL_FALLBACK_EMBEDDING_MODEL)
)
DEFAULT_EMBEDDING_DEVICE = _env("RAG_EMBEDDING_DEVICE", "auto")
DEFAULT_EMBEDDING_BATCH_SIZE = int(_env("RAG_EMBEDDING_BATCH_SIZE", "16"))
DEFAULT_EMBEDDING_WORKERS = int(_env("RAG_EMBEDDING_WORKERS", "1"))
DEFAULT_EMBEDDING_CACHE_DIR = OUTPUT_DIR / "embeddings"

ARK_EMBEDDING_MODEL_HINTS = ("doubao-embedding", "embedding-vision", "multimodal")
OPENAI_COMPATIBLE_PROVIDERS = {"openai_compatible", "openai-compatible", "local_openai", "local-server"}
XINFERENCE_PROVIDERS = {"xinference"}
VISUAL_EMBED_NODE_TYPES = {"figure", "table", "caption"}
VISUAL_TEXT_FIELDS = (
    "visual_title",
    "visual_type",
    "key_objects",
    "ocr_text",
    "data_or_trends",
    "qa_evidence",
    "limitations",
    "visual_caption",
    "visual_summary",
)


def node_embedding_text(node: dict[str, Any]) -> str:
    node_type = clean_text(node.get("node_type")) or "text"
    paper_domain = clean_text(node.get("paper_domain"))
    doc_id = clean_text(node.get("doc_id"))
    source_ref = clean_text(node.get("source_ref"))
    section = clean_text(node.get("section"))
    structure_type = clean_text(node.get("structure_type"))
    chunk_strategy = clean_text(node.get("chunk_strategy"))
    product_category = clean_text(node.get("product_category"))
    service_intents = clean_text(node.get("service_intents"))
    searchable_text = clean_text(node.get("searchable_text"))
    previous_preview = clean_text(node.get("previous_chunk_preview"))
    next_preview = clean_text(node.get("next_chunk_preview"))
    explicit_refs = clean_text(node.get("explicit_refs"))
    content = clean_text(node.get("content"))
    visual_parts = [
        f"{field}: {clean_text(node.get(field))}"
        for field in VISUAL_TEXT_FIELDS
        if clean_text(node.get(field))
    ]
    if not content and not visual_parts:
        return ""
    parts = [f"type: {node_type}"]
    if doc_id:
        parts.append(f"doc_id: {doc_id}")
    if paper_domain:
        parts.append(f"paper_domain: {paper_domain}")
    if product_category:
        parts.append(f"product_category: {product_category}")
    if service_intents:
        parts.append(f"service_intents: {service_intents}")
    if section:
        parts.append(f"section: {section}")
    if source_ref:
        parts.append(f"source_ref: {source_ref}")
    if structure_type:
        parts.append(f"structure_type: {structure_type}")
    if chunk_strategy:
        parts.append(f"chunk_strategy: {chunk_strategy}")
    if explicit_refs:
        parts.append(f"explicit_refs: {explicit_refs}")
    if previous_preview:
        parts.append(f"previous_context: {previous_preview}")
    if next_preview:
        parts.append(f"next_context: {next_preview}")
    if visual_parts:
        parts.append("visual_evidence:\n" + "\n".join(visual_parts))
    if searchable_text and searchable_text != content:
        parts.append(f"searchable_text: {searchable_text}")
    if content:
        parts.append(content)
    return "\n".join(parts)


def node_embedding_image_path(node: dict[str, Any]) -> Path | None:
    node_type = clean_text(node.get("node_type"))
    if node_type not in VISUAL_EMBED_NODE_TYPES:
        return None
    for field in ("crop_image_path", "image_path"):
        path_text = clean_text(node.get(field))
        if not path_text or re.match(r"^(https?|data):", path_text, flags=re.I):
            continue
        path = resolve_path(path_text)
        if path.exists() and path.is_file():
            return path
    return None


def _embedding_provider(model_name: str, provider: str = DEFAULT_EMBEDDING_PROVIDER) -> str:
    provider = (provider or "auto").strip().lower()
    if provider in {"ark", "doubao", "volcengine"}:
        return "ark"
    if provider in XINFERENCE_PROVIDERS:
        return "xinference"
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return "openai_compatible"
    if provider in {"local", "sentence-transformers", "sentence_transformers", "st"}:
        return "local"
    lowered = model_name.lower()
    if _env("XINFERENCE_EMBEDDING_MODEL", "").strip() and model_name == _env("XINFERENCE_EMBEDDING_MODEL", "").strip():
        return "xinference"
    openai_model = _env("OPENAI_COMPATIBLE_EMBEDDING_MODEL", "").strip() or _env("LOCAL_EMBEDDING_MODEL", "").strip()
    if openai_model and model_name == openai_model:
        return "openai_compatible"
    if any(hint in lowered for hint in ARK_EMBEDDING_MODEL_HINTS):
        return "ark"
    return "local"


def _safe_model_name(model_name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", model_name).strip("_") or "embedding_model"


def _nodes_hash(nodes: list[dict[str, Any]], model_name: str, provider: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(provider.encode("utf-8"))
    hasher.update(b"\0")
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
        image_path = node_embedding_image_path(node) if provider == "ark" else None
        if image_path:
            try:
                stat = image_path.stat()
                hasher.update(b"\0IMAGE\0")
                hasher.update(str(image_path.resolve()).encode("utf-8"))
                hasher.update(b"\0")
                hasher.update(str(stat.st_size).encode("utf-8"))
                hasher.update(b"\0")
                hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
            except OSError:
                pass
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
    workers: int = DEFAULT_EMBEDDING_WORKERS
    provider: str = DEFAULT_EMBEDDING_PROVIDER
    cache_dir: str | Path = DEFAULT_EMBEDDING_CACHE_DIR

    _model: Any | None = None
    _ark_embedder: Any | None = None
    _ark_cache: Any | None = None
    _openai_embedder: Any | None = None
    _openai_cache: Any | None = None
    _query_embedding_cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_nodes(
        cls,
        nodes: list[dict[str, Any]],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        provider: str = DEFAULT_EMBEDDING_PROVIDER,
        cache_dir: str | Path = DEFAULT_EMBEDDING_CACHE_DIR,
        device: str = DEFAULT_EMBEDDING_DEVICE,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        workers: int = DEFAULT_EMBEDDING_WORKERS,
    ) -> "EmbeddingIndex":
        valid_nodes = [node for node in nodes if clean_text(node.get("node_id")) and node_embedding_text(node)]
        node_ids = [clean_text(node.get("node_id")) for node in valid_nodes]
        cache_dir = resolve_path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        provider = _embedding_provider(model_name, provider)
        content_hash = _nodes_hash(valid_nodes, model_name, provider)
        cache_path = cache_dir / f"{provider}_{_safe_model_name(model_name)}_{content_hash[:16]}.npz"

        embeddings = cls._load_cache(cache_path, model_name, content_hash, node_ids, provider)
        if embeddings is None:
            if provider == "ark":
                embeddings = cls._encode_ark_nodes(valid_nodes, model_name, cache_dir)
            elif provider in {"xinference", "openai_compatible"}:
                embeddings = cls._encode_openai_nodes(valid_nodes, model_name, cache_dir, provider)
            else:
                texts = [node_embedding_text(node) for node in valid_nodes]
                model = cls._load_model(model_name, device)
                embeddings = cls._encode(model, texts, batch_size=batch_size)
            metadata = {
                "provider": provider,
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
            workers=workers,
            provider=provider,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _load_cache(
        cache_path: Path,
        model_name: str,
        content_hash: str,
        node_ids: list[str],
        provider: str,
    ) -> np.ndarray | None:
        if not cache_path.exists():
            return None
        try:
            data = np.load(cache_path, allow_pickle=False)
            metadata = json.loads(str(data["metadata"].item()))
            cached_ids = [str(item) for item in data["node_ids"].tolist()]
            if (
                metadata.get("provider", "local") == provider
                and metadata.get("model_name") == model_name
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
        return EmbeddingIndex._normalize_rows(np.asarray(embeddings, dtype="float32"))

    @staticmethod
    def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings.astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (embeddings / norms).astype("float32")

    @staticmethod
    def _ark_cache_path(cache_dir: str | Path, model_name: str) -> Path:
        return resolve_path(cache_dir) / f"{_safe_model_name(model_name)}_items.jsonl"

    @staticmethod
    def _openai_cache_path(cache_dir: str | Path, model_name: str, provider: str) -> Path:
        return resolve_path(cache_dir) / f"{provider}_{_safe_model_name(model_name)}_items.jsonl"

    @staticmethod
    def _encode_ark_nodes(nodes: list[dict[str, Any]], model_name: str, cache_dir: str | Path) -> np.ndarray:
        try:
            from ark_clients import ArkMultimodalEmbedder, JsonlEmbeddingCache
        except ImportError as exc:
            raise RuntimeError("Ark embedding retriever needs scripts/ark_clients.py.") from exc

        embedder = ArkMultimodalEmbedder(model=model_name)
        cache = JsonlEmbeddingCache(EmbeddingIndex._ark_cache_path(cache_dir, model_name))
        total = len(nodes)
        workers = max(1, int(_env("RAG_EMBEDDING_WORKERS", "1")))
        vectors: list[list[float] | None] = [None] * total

        def encode_one(index: int, node: dict[str, Any]) -> tuple[int, list[float]]:
            text = node_embedding_text(node)
            image_path = node_embedding_image_path(node)
            try:
                if image_path:
                    try:
                        vector = embedder.embed_image_file(image_path, text=text[:1800], cache=cache)
                    except Exception as image_exc:
                        node_id = clean_text(node.get("node_id")) or f"#{index + 1}"
                        print(
                            f"Ark image embedding failed for node {node_id}; falling back to text embedding: {image_exc}"
                        )
                        vector = embedder.embed_text(text, cache=cache)
                else:
                    vector = embedder.embed_text(text, cache=cache)
            except Exception as exc:
                node_id = clean_text(node.get("node_id")) or f"#{index + 1}"
                raise RuntimeError(f"Ark embedding failed for node {node_id}: {exc}") from exc
            return index, vector

        if workers == 1:
            for index, node in enumerate(nodes):
                done_index, vector = encode_one(index, node)
                vectors[done_index] = vector
                completed = done_index + 1
                if completed == 1 or completed % 10 == 0 or completed == total:
                    print(f"Ark embedding-vision encoded {completed}/{total} nodes")
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(encode_one, index, node) for index, node in enumerate(nodes)]
                completed = 0
                for future in as_completed(futures):
                    done_index, vector = future.result()
                    vectors[done_index] = vector
                    completed += 1
                    if completed == 1 or completed % 10 == 0 or completed == total:
                        print(f"Ark embedding-vision encoded {completed}/{total} nodes")

        return EmbeddingIndex._normalize_rows(np.asarray(vectors, dtype="float32"))

    @staticmethod
    def _encode_openai_nodes(
        nodes: list[dict[str, Any]],
        model_name: str,
        cache_dir: str | Path,
        provider: str,
    ) -> np.ndarray:
        try:
            from ark_clients import JsonlEmbeddingCache, OpenAICompatibleEmbedder
        except ImportError as exc:
            raise RuntimeError("OpenAI-compatible embedding retriever needs scripts/ark_clients.py.") from exc

        embedder = OpenAICompatibleEmbedder(provider=provider, model=model_name)
        cache = JsonlEmbeddingCache(EmbeddingIndex._openai_cache_path(cache_dir, model_name, provider))
        total = len(nodes)
        workers = max(1, int(_env("RAG_EMBEDDING_WORKERS", "1")))
        vectors: list[list[float] | None] = [None] * total

        def encode_one(index: int, node: dict[str, Any]) -> tuple[int, list[float]]:
            text = node_embedding_text(node)
            try:
                vector = embedder.embed_text(text, cache=cache)
            except Exception as exc:
                node_id = clean_text(node.get("node_id")) or f"#{index + 1}"
                raise RuntimeError(f"{provider} embedding failed for node {node_id}: {exc}") from exc
            return index, vector

        if workers == 1:
            for index, node in enumerate(nodes):
                done_index, vector = encode_one(index, node)
                vectors[done_index] = vector
                completed = done_index + 1
                if completed == 1 or completed % 10 == 0 or completed == total:
                    print(f"{provider} embedding encoded {completed}/{total} nodes")
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(encode_one, index, node) for index, node in enumerate(nodes)]
                completed = 0
                for future in as_completed(futures):
                    done_index, vector = future.result()
                    vectors[done_index] = vector
                    completed += 1
                    if completed == 1 or completed % 10 == 0 or completed == total:
                        print(f"{provider} embedding encoded {completed}/{total} nodes")

        return EmbeddingIndex._normalize_rows(np.asarray(vectors, dtype="float32"))

    @property
    def model(self) -> Any:
        if self.provider == "ark":
            raise RuntimeError("Ark embedding index does not use a local sentence-transformers model.")
        if self.provider in {"xinference", "openai_compatible"}:
            raise RuntimeError("OpenAI-compatible embedding index does not use a local sentence-transformers model.")
        if self._model is None:
            self._model = self._load_model(self.model_name, self.device)
        return self._model

    @property
    def ark_embedder(self) -> Any:
        if self._ark_embedder is None:
            from ark_clients import ArkMultimodalEmbedder

            self._ark_embedder = ArkMultimodalEmbedder(model=self.model_name)
        return self._ark_embedder

    @property
    def ark_cache(self) -> Any:
        if self._ark_cache is None:
            from ark_clients import JsonlEmbeddingCache

            self._ark_cache = JsonlEmbeddingCache(self._ark_cache_path(self.cache_dir, self.model_name))
        return self._ark_cache

    @property
    def openai_embedder(self) -> Any:
        if self._openai_embedder is None:
            from ark_clients import OpenAICompatibleEmbedder

            self._openai_embedder = OpenAICompatibleEmbedder(provider=self.provider, model=self.model_name)
        return self._openai_embedder

    @property
    def openai_cache(self) -> Any:
        if self._openai_cache is None:
            from ark_clients import JsonlEmbeddingCache

            self._openai_cache = JsonlEmbeddingCache(
                self._openai_cache_path(self.cache_dir, self.model_name, self.provider)
            )
        return self._openai_cache

    def _encode_queries(self, queries: list[str]) -> dict[str, np.ndarray]:
        unique_queries = [query for query in dict.fromkeys(clean_text(query) for query in queries) if query]
        missing = [query for query in unique_queries if query not in self._query_embedding_cache]
        if missing:
            if self.provider == "ark":
                for query in missing:
                    vector = self.ark_embedder.embed_text(query, cache=self.ark_cache)
                    self._query_embedding_cache[query] = self._normalize_rows(
                        np.asarray([vector], dtype="float32")
                    )[0]
            elif self.provider in {"xinference", "openai_compatible"}:
                for query in missing:
                    vector = self.openai_embedder.embed_text(query, cache=self.openai_cache)
                    self._query_embedding_cache[query] = self._normalize_rows(
                        np.asarray([vector], dtype="float32")
                    )[0]
            else:
                embeddings = self._encode(self.model, missing, batch_size=max(1, self.batch_size))
                for query, embedding in zip(missing, embeddings):
                    self._query_embedding_cache[query] = embedding.astype("float32")
        return {
            query: self._query_embedding_cache[query]
            for query in unique_queries
            if query in self._query_embedding_cache
        }

    def score_many(self, queries: list[Any], nodes: list[dict[str, Any]] | None = None) -> list[dict[str, float]]:
        clean_queries = [clean_text(query) for query in queries]
        target_ids = self.node_ids if nodes is None else [clean_text(node.get("node_id")) for node in nodes]
        target_ids = [node_id for node_id in target_ids if node_id]
        empty = {node_id: 0.0 for node_id in target_ids}
        if self.embeddings.size == 0:
            return [dict(empty) for _ in clean_queries]

        query_embeddings = self._encode_queries(clean_queries)
        valid_queries = [query for query in dict.fromkeys(clean_queries) if query and query in query_embeddings]
        if not valid_queries:
            return [dict(empty) for _ in clean_queries]

        matrix = np.vstack([query_embeddings[query] for query in valid_queries]).astype("float32")
        if self.embeddings.ndim != 2 or matrix.ndim != 2 or matrix.shape[1] != self.embeddings.shape[1]:
            return [dict(empty) for _ in clean_queries]

        all_scores = np.matmul(self.embeddings, matrix.T)
        score_maps: dict[str, dict[str, float]] = {}
        target_set = set(target_ids)
        for column, query in enumerate(valid_queries):
            score_maps[query] = {
                node_id: float(max(0.0, score))
                for node_id, score in zip(self.node_ids, all_scores[:, column].tolist())
                if node_id in target_set
            }
        return [score_maps.get(query, dict(empty)) if query else dict(empty) for query in clean_queries]

    def score(self, query: Any, nodes: list[dict[str, Any]] | None = None) -> dict[str, float]:
        query_text = clean_text(query)
        target_ids = self.node_ids if nodes is None else [clean_text(node.get("node_id")) for node in nodes]
        if not query_text or self.embeddings.size == 0:
            return {node_id: 0.0 for node_id in target_ids if node_id}
        return self.score_many([query_text], nodes=nodes)[0]
