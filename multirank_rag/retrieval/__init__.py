"""Retrieval and embedding interfaces."""

from multirank_rag.retrieval.index import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingIndex,
    build_graph,
    load_kg_index,
    retrieve_candidates,
)

__all__ = [
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    "DEFAULT_EMBEDDING_DEVICE",
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingIndex",
    "build_graph",
    "load_kg_index",
    "retrieve_candidates",
]
