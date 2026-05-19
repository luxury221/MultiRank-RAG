from __future__ import annotations

from multirank_rag.legacy import load_numbered_script


_legacy = load_numbered_script("12_evaluate_evidence_chains.py", "multirank_rag_legacy_evaluate_chains")

evaluate_chain = _legacy.evaluate_chain
summarize = _legacy.summarize
step_has_visual_grounding = _legacy.step_has_visual_grounding
