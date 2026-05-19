from __future__ import annotations

from multirank_rag.legacy import import_legacy_module


_ark_clients = import_legacy_module("ark_clients")

ArkChatClient = _ark_clients.ArkChatClient
ArkError = _ark_clients.ArkError
ArkMultimodalEmbedder = _ark_clients.ArkMultimodalEmbedder
JsonlEmbeddingCache = _ark_clients.JsonlEmbeddingCache
ModelClientError = _ark_clients.ModelClientError
OpenAICompatibleEmbedder = _ark_clients.OpenAICompatibleEmbedder
XinferenceRerankClient = _ark_clients.XinferenceRerankClient
answer_model_for_provider = _ark_clients.answer_model_for_provider
create_chat_client = _ark_clients.create_chat_client
get_env = _ark_clients.get_env
rerank_model_for_provider = _ark_clients.rerank_model_for_provider
