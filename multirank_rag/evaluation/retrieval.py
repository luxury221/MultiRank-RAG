from __future__ import annotations

from multirank_rag.legacy import load_numbered_script


_legacy = load_numbered_script("05_evaluate.py", "multirank_rag_legacy_evaluate")

evaluate_question_method = _legacy.evaluate_question_method
summarize = _legacy.summarize
ndcg_at_k = _legacy.ndcg_at_k
