from __future__ import annotations

from multirank_rag.legacy import load_numbered_script


_legacy = load_numbered_script("11_build_evidence_cards.py", "multirank_rag_legacy_evidence_cards")

build_card = _legacy.build_card
grouped_steps = _legacy.grouped_steps
