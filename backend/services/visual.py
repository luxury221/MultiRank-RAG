from __future__ import annotations

from backend.config import VISUAL_CAPTION_PROVIDERS, env_float, env_value
from backend.jobs.store import append_log
from backend.services.script_modules import visual_evidence

from multirank_rag.common import clean_text


def default_backend_visual_caption_provider() -> str:
    provider = clean_text(
        env_value("RAG_BACKEND_VISUAL_CAPTION_PROVIDER")
        or env_value("RAG_VISUAL_CAPTION_PROVIDER")
        or ""
    ).lower()
    if provider in VISUAL_CAPTION_PROVIDERS:
        return provider
    model_provider = clean_text(env_value("RAG_MODEL_PROVIDER", "")).lower()
    if model_provider == "xinference" and env_value("XINFERENCE_VISION_MODEL"):
        return "xinference"
    if model_provider in {"openai_compatible", "openai-compatible", "local_openai", "local-server"} and (
        env_value("OPENAI_COMPATIBLE_VISION_MODEL") or env_value("LOCAL_VISION_MODEL")
    ):
        return "openai_compatible"
    return "qwen" if env_value("DASHSCOPE_API_KEY") else "local"


def build_backend_captioner(job_id: str):
    provider = default_backend_visual_caption_provider()
    if provider == "qwen":
        captioner = visual_evidence.QwenVisionCaptioner(
            model_name=env_value("RAG_BACKEND_QWEN_MODEL")
            or env_value("RAG_QWEN_VL_MODEL")
            or visual_evidence.QWEN_DEFAULT_MODEL,
            base_url=env_value("RAG_QWEN_BASE_URL", visual_evidence.QWEN_DEFAULT_BASE_URL),
            api_key_env=env_value("RAG_QWEN_API_KEY_ENV", "DASHSCOPE_API_KEY"),
            timeout=env_float("RAG_QWEN_TIMEOUT", 60.0),
        )
        if captioner.available():
            return captioner, "qwen"
        append_log(job_id, "Qwen visual caption is enabled but the API key is unavailable; using crops only.")
        return visual_evidence.VisualCaptioner(""), "local"
    if provider == "doubao":
        captioner = visual_evidence.ArkVisionCaptioner(
            model_name=env_value("RAG_BACKEND_ARK_VISION_MODEL") or env_value("RAG_ARK_VISION_MODEL", ""),
            base_url=env_value("RAG_ARK_BASE_URL", visual_evidence.ARK_DEFAULT_BASE_URL),
            api_key_env=env_value("RAG_ARK_API_KEY_ENV", "ARK_API_KEY"),
            timeout=env_float("RAG_ARK_TIMEOUT", 60.0),
        )
        if captioner.available():
            return captioner, "doubao"
        append_log(job_id, "Doubao visual caption is enabled but ARK_API_KEY or model is unavailable; using crops only.")
        return visual_evidence.VisualCaptioner(""), "local"
    if provider == "xinference":
        captioner = visual_evidence.QwenVisionCaptioner(
            model_name=env_value("RAG_BACKEND_XINFERENCE_VISION_MODEL")
            or env_value("XINFERENCE_VISION_MODEL")
            or env_value("RAG_VISION_MODEL", ""),
            base_url=env_value("XINFERENCE_BASE_URL", visual_evidence.XINFERENCE_DEFAULT_BASE_URL),
            api_key_env=env_value("XINFERENCE_API_KEY_ENV", "XINFERENCE_API_KEY"),
            timeout=env_float("XINFERENCE_TIMEOUT", 60.0),
            allow_no_api_key=True,
        )
        if captioner.available() and captioner.model_name:
            return captioner, "xinference"
        append_log(job_id, "Xinference visual caption is enabled but no vision model is configured; using crops only.")
        return visual_evidence.VisualCaptioner(""), "local"
    if provider == "openai_compatible":
        captioner = visual_evidence.QwenVisionCaptioner(
            model_name=env_value("RAG_BACKEND_OPENAI_COMPATIBLE_VISION_MODEL")
            or env_value("OPENAI_COMPATIBLE_VISION_MODEL")
            or env_value("LOCAL_VISION_MODEL", ""),
            base_url=env_value(
                "OPENAI_COMPATIBLE_BASE_URL",
                env_value("LOCAL_MODEL_BASE_URL", visual_evidence.OPENAI_COMPATIBLE_DEFAULT_BASE_URL),
            ),
            api_key_env=env_value("OPENAI_COMPATIBLE_API_KEY_ENV", "OPENAI_COMPATIBLE_API_KEY"),
            timeout=env_float("OPENAI_COMPATIBLE_TIMEOUT", 60.0),
            allow_no_api_key=True,
        )
        if captioner.available() and captioner.model_name:
            return captioner, "openai_compatible"
        append_log(job_id, "OpenAI-compatible visual caption is enabled but no vision model is configured; using crops only.")
        return visual_evidence.VisualCaptioner(""), "local"
    caption_model = env_value("RAG_BACKEND_VISUAL_CAPTION_MODEL", "")
    caption_device = env_value("RAG_BACKEND_VISUAL_CAPTION_DEVICE", "auto")
    return (
        visual_evidence.VisualCaptioner(caption_model, caption_device)
        if caption_model
        else visual_evidence.VisualCaptioner("")
    ), "local"
