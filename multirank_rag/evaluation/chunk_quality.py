from __future__ import annotations

from multirank_rag.legacy import load_numbered_script


_legacy = load_numbered_script("14_chunk_quality_report.py", "multirank_rag_legacy_chunk_quality")

build_report = _legacy.build_report
