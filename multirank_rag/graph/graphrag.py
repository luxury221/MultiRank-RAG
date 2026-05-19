from __future__ import annotations

from multirank_rag.legacy import load_numbered_script


_legacy = load_numbered_script("23_build_graphrag.py", "multirank_rag_legacy_graphrag")

build_graphrag = _legacy.build_graphrag
extract_terms = _legacy.extract_terms
extract_refs = _legacy.extract_refs
graph_from_edges = _legacy.graph_from_edges
