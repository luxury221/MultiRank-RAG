from __future__ import annotations

from multirank_rag.legacy import load_numbered_script


_legacy = load_numbered_script("09_build_evidence_chains.py", "multirank_rag_legacy_evidence_chains")

CHAIN_FIELDS = _legacy.CHAIN_FIELDS
build_chain_for_question = _legacy.build_chain_for_question
chain_summary = _legacy.chain_summary
write_markdown = _legacy.write_markdown
