from __future__ import annotations

from multirank_rag.legacy import import_legacy_module


_rerank_lib = import_legacy_module("rerank_lib")

self_correct_answer = _rerank_lib.self_correct_answer
answer_for_question = _rerank_lib.answer_for_question
