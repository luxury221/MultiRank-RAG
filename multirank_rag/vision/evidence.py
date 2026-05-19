from __future__ import annotations

from multirank_rag.legacy import load_numbered_script


_legacy = load_numbered_script("10_build_visual_evidence.py", "multirank_rag_legacy_visual_evidence")

ARK_DEFAULT_BASE_URL = _legacy.ARK_DEFAULT_BASE_URL
CAPTION_PROVIDERS = _legacy.CAPTION_PROVIDERS
OPENAI_COMPATIBLE_DEFAULT_BASE_URL = _legacy.OPENAI_COMPATIBLE_DEFAULT_BASE_URL
QWEN_DEFAULT_BASE_URL = _legacy.QWEN_DEFAULT_BASE_URL
QWEN_DEFAULT_MODEL = _legacy.QWEN_DEFAULT_MODEL
VISUAL_NODE_TYPES = _legacy.VISUAL_NODE_TYPES
XINFERENCE_DEFAULT_BASE_URL = _legacy.XINFERENCE_DEFAULT_BASE_URL

ArkVisionCaptioner = _legacy.ArkVisionCaptioner
QwenVisionCaptioner = _legacy.QwenVisionCaptioner
VisualCaptioner = _legacy.VisualCaptioner

process_document = _legacy.process_document
build_visual_summary = _legacy.build_visual_summary
write_caption_fields = _legacy.write_caption_fields
