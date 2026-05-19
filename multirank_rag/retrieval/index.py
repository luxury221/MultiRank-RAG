from __future__ import annotations

from multirank_rag.legacy import import_legacy_module


_embedding_index = import_legacy_module("embedding_index")
_rerank_lib = import_legacy_module("rerank_lib")

DEFAULT_EMBEDDING_BATCH_SIZE = _embedding_index.DEFAULT_EMBEDDING_BATCH_SIZE
DEFAULT_EMBEDDING_DEVICE = _embedding_index.DEFAULT_EMBEDDING_DEVICE
DEFAULT_EMBEDDING_MODEL = _embedding_index.DEFAULT_EMBEDDING_MODEL
EmbeddingIndex = _embedding_index.EmbeddingIndex

build_graph = _rerank_lib.build_graph
load_kg_index = _rerank_lib.load_kg_index
retrieve_candidates = _rerank_lib.retrieve_candidates
