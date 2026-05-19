from __future__ import annotations

from multirank_rag.legacy import load_numbered_script


_legacy = load_numbered_script("02_build_graph.py", "multirank_rag_legacy_build_graph")

build_edges = _legacy.build_edges
add_edge = _legacy.add_edge
extract_refs = _legacy.extract_refs
