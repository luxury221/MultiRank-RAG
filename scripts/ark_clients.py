from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    import winreg
except ImportError:  # pragma: no cover - only used on Windows.
    winreg = None  # type: ignore[assignment]


class ArkError(RuntimeError):
    pass


def _windows_user_env(name: str) -> str:
    if winreg is None:
        return ""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value).strip()
    except OSError:
        return ""


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or _windows_user_env(name) or default


def ark_base_url() -> str:
    return get_env("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")


def ark_api_key() -> str:
    key = get_env("ARK_API_KEY")
    if not key:
        raise ArkError("ARK_API_KEY is not configured.")
    return key


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass
class JsonlEmbeddingCache:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, list[float]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = str(row.get("key", ""))
                    embedding = row.get("embedding")
                    if key and isinstance(embedding, list):
                        self._items[key] = [float(value) for value in embedding]

    @staticmethod
    def text_key(model: str, dimensions: int | None, text: str) -> str:
        raw = json.dumps(
            {"model": model, "dimensions": dimensions, "text": text},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def image_file_key(model: str, dimensions: int | None, image_path: str | Path, text: str = "") -> str:
        path = Path(image_path)
        stat = path.stat()
        raw = json.dumps(
            {
                "model": model,
                "dimensions": dimensions,
                "image_path": str(path.resolve()),
                "image_size": stat.st_size,
                "image_mtime_ns": stat.st_mtime_ns,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> list[float] | None:
        return self._items.get(key)

    def put(self, key: str, embedding: list[float]) -> None:
        if key in self._items:
            return
        self._items[key] = embedding
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"key": key, "embedding": embedding}, ensure_ascii=False) + "\n")


@dataclass
class ArkMultimodalEmbedder:
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    dimensions: int | None = None
    timeout_seconds: int = 90
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.api_key = self.api_key or ark_api_key()
        self.base_url = (self.base_url or ark_base_url()).rstrip("/")
        self.model = self.model or get_env("ARK_EMBEDDING_MODEL", "doubao-embedding-vision-250615")
        if self.dimensions is None:
            raw_dimensions = get_env("ARK_EMBEDDING_DIMENSIONS", "")
            self.dimensions = int(raw_dimensions) if raw_dimensions.isdigit() else 1024

    def embed_text(self, text: str, cache: JsonlEmbeddingCache | None = None) -> list[float]:
        text = str(text).strip()
        if not text:
            return []
        cache_key = JsonlEmbeddingCache.text_key(self.model or "", self.dimensions, text)
        if cache:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        embedding = self.embed_items([{"type": "text", "text": text}])
        if cache:
            cache.put(cache_key, embedding)
        return embedding

    def embed_image_url(self, image_url: str, text: str = "") -> list[float]:
        items: list[dict[str, Any]] = []
        if text.strip():
            items.append({"type": "text", "text": text.strip()})
        items.append({"type": "image_url", "image_url": {"url": image_url}})
        return self.embed_items(items)

    def embed_image_file(
        self,
        image_path: str | Path,
        text: str = "",
        cache: JsonlEmbeddingCache | None = None,
    ) -> list[float]:
        path = Path(image_path)
        cache_key = JsonlEmbeddingCache.image_file_key(self.model or "", self.dimensions, path, text)
        if cache:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        embedding = self.embed_image_url(f"data:{mime};base64,{data}", text=text)
        if cache:
            cache.put(cache_key, embedding)
        return embedding

    def embed_items(self, items: list[dict[str, Any]]) -> list[float]:
        if not items:
            return []
        body: dict[str, Any] = {"model": self.model, "input": items}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings/multimodal",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return _extract_embedding(payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = ArkError(f"Ark embedding HTTP {exc.code}: {detail[:500]}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))
        raise ArkError(str(last_error) if last_error else "Ark embedding request failed.")


@dataclass
class ArkChatClient:
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        self.api_key = self.api_key or ark_api_key()
        self.base_url = (self.base_url or ark_base_url()).rstrip("/")
        self.model = (
            self.model
            or get_env("RAG_ANSWER_MODEL")
            or get_env("ARK_TEXT_MODEL_PRO")
            or get_env("ARK_TEXT_MODEL")
            or get_env("ARK_MODEL")
        )
        if not self.model:
            raise ArkError("ARK_TEXT_MODEL or ARK_MODEL is not configured.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise ArkError("The openai package is required for Ark chat calls.") from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_seconds)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 420,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


def _extract_embedding(payload: dict[str, Any]) -> list[float]:
    data = payload.get("data")
    embedding: Any = None
    if isinstance(data, dict):
        embedding = data.get("embedding")
    elif isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            embedding = first.get("embedding")
    if not isinstance(embedding, list):
        raise ArkError(f"Ark embedding response does not contain an embedding: {json.dumps(payload)[:500]}")
    return [float(value) for value in embedding]
