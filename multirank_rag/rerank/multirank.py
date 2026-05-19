from __future__ import annotations

from multirank_rag.legacy import import_legacy_module


_rerank_lib = import_legacy_module("rerank_lib")

adaptive_rag_route = _rerank_lib.adaptive_rag_route
answer_for_question = _rerank_lib.answer_for_question
build_query_plan = _rerank_lib.build_query_plan
query_plan_summary = _rerank_lib.query_plan_summary
rank_question = _rerank_lib.rank_question
