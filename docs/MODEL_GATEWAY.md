# Model Gateway

MultiRank-RAG now supports three model access modes behind one configuration layer:

```text
ark                 Direct cloud API calls, usually Doubao / Ark and DashScope Qwen-VL.
xinference          A local Xinference gateway at /v1 for LLM, embedding, VLM, and rerank.
openai_compatible   Any local OpenAI-compatible server, such as vLLM, LM Studio, or an Ollama proxy.
```

MinerU is still the PDF parser. The model gateway is only responsible for visual captioning, embedding, final answer generation, and optional model reranking.

## Ark / Cloud Mode

Use this when you call Doubao and Qwen services directly:

```text
RAG_MODEL_PROVIDER=ark
RAG_EMBEDDING_PROVIDER=ark
RAG_EMBEDDING_MODEL=doubao-embedding-vision-250615
ARK_API_KEY=your-ark-key

RAG_ANSWER_PROVIDER=ark
RAG_ANSWER_MODEL=your-doubao-2.0-pro-endpoint-id

RAG_VISUAL_CAPTION_PROVIDER=qwen
DASHSCOPE_API_KEY=your-dashscope-key
RAG_QWEN_VL_MODEL=qwen-vl-plus
```

This keeps the original high-quality cloud chain.

## Xinference Mode

Use this when your models are launched in Xinference and exposed through its OpenAI-compatible `/v1` API:

```text
RAG_MODEL_PROVIDER=xinference
XINFERENCE_BASE_URL=http://127.0.0.1:9997/v1
XINFERENCE_API_KEY=not-used

RAG_EMBEDDING_PROVIDER=xinference
XINFERENCE_EMBEDDING_MODEL=your-embedding-model-uid

RAG_ANSWER_PROVIDER=xinference
XINFERENCE_LLM_MODEL=your-llm-model-uid

RAG_VISUAL_CAPTION_PROVIDER=xinference
XINFERENCE_VISION_MODEL=your-vlm-model-uid

RAG_RERANK_PROVIDER=xinference
XINFERENCE_RERANK_MODEL=your-rerank-model-uid
RAG_ENABLE_MODEL_RERANK=1
```

The project does not automatically download or launch models in Xinference. It calls the model UIDs that you have already launched.

## Local OpenAI-Compatible Mode

Use this for a local service that follows the OpenAI API shape but is not Xinference:

```text
RAG_MODEL_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_COMPATIBLE_API_KEY=not-used

RAG_EMBEDDING_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_EMBEDDING_MODEL=your-local-embedding-model

RAG_ANSWER_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_LLM_MODEL=your-local-chat-model

RAG_VISUAL_CAPTION_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_VISION_MODEL=your-local-vlm-model

RAG_RERANK_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_RERANK_MODEL=your-local-rerank-model
```

## What Each Model Does

- Visual model: reads cropped figures/tables and writes `visual_caption`, `key_objects`, `qa_evidence`, and related fields.
- Embedding model: converts evidence-node text into vectors for semantic retrieval.
- Rerank model: optional cross-encoder/reranker signal blended into G4.
- Answer model: generates the final grounded answer from evidence-chain text.

The system remains usable without a configured VLM or reranker; it falls back to crops plus structural/lexical/embedding signals where possible.
