from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
JOB_ROOT = ROOT / "outputs" / "upload_jobs"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


API_TITLE = "Multimodal RAG Evidence API"

CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://localhost:5173",
    "http://localhost:5174",
]

VISUAL_CAPTION_PROVIDERS = {"local", "qwen", "doubao", "xinference", "openai_compatible"}


def env_value(name: str, default: str = "") -> str:
    try:
        from ark_clients import get_env

        return get_env(name, default)
    except Exception:
        return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    try:
        return int(env_value(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_value(name, str(default)))
    except (TypeError, ValueError):
        return default


def normalize_chunk_template(value: str) -> str:
    value = " ".join(str(value or "").split()).lower() or "auto"
    return value if value in {"auto", "general", "ai", "math", "finance", "medical"} else "auto"
