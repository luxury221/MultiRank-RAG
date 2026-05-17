# Architecture

MultiRank-RAG 的核心目标是把复杂 PDF 中的文本、图片、表格、公式和版面关系转换成可检索、可重排、可解释的证据链。系统不是简单的“PDF 转文本 + 向量检索”，而是把复杂文档拆成三类互补能力：

```text
Document RAG              复杂 PDF 结构理解
GraphRAG                  证据节点关系传播与子图解释
Multimodal Evidence RAG   图文表联合证据检索和展示
```

## End-to-End Flow

```text
PDF documents
  -> PDF parsing
  -> evidence node construction
  -> visual evidence enrichment
  -> GraphRAG construction
  -> hybrid retrieval
  -> MultiRank reranking
  -> evidence chain assembly
  -> grounded answer generation
  -> UI / API / CSV export
```

## Evidence Node Layer

解析后的文档会被统一转换为 evidence node。节点类型包括：

- `title`: 标题或章节。
- `text`: 正文 chunk。
- `table`: 表格内容。
- `figure`: 图片、图示、截图或页面裁剪图。
- `caption`: 图注和表注。
- `equation`: 公式。
- `page`: 页面级上下文。

常用字段：

```text
node_id, doc_id, page, node_type, content, source_ref,
section, bbox, parent_chunk_id, previous_node_id, next_node_id,
crop_image_path, page_image_path,
visual_caption, ocr_text, key_objects, qa_evidence, visual_summary
```

这一层是整个系统的“统一证据接口”。后续检索、GraphRAG、重排、证据链和前端展示都围绕 evidence node 工作。

## Document Graph

`scripts/02_build_graph.py` 会从节点中构建文档结构边：

- `belongs_to_page`: 节点属于某一页。
- `same_page`: 同页节点关系。
- `same_section`: 同章节顺序关系。
- `section_title`: 标题与正文关系。
- `chunk_sequence`: 前后 chunk 顺序。
- `figure_caption` / `table_caption`: 图表与图注关系。
- `text_ref_figure` / `text_ref_table`: 正文显式引用图表。

这部分解决复杂 PDF 中“证据并不总在同一段文本里”的问题。

## Visual Evidence Layer

视觉增强模块会为图片和表格补充结构化视觉字段：

- `visual_caption`: 图片内容说明。
- `ocr_text`: 图片中的文字。
- `key_objects`: 关键对象、按钮、部件、符号。
- `qa_evidence`: 更适合问答使用的视觉证据摘要。

视觉模型负责理解图片内容，embedding 模型负责把文本和图片证据映射到统一检索空间。两者职责不同，可以组合使用。

## Model Gateway Layer

模型调用被统一收敛到一个 provider 配置层：

- `ark`: 直接调用 Ark / Doubao / DashScope 等云端模型。
- `xinference`: 通过本地 Xinference 的 OpenAI-compatible `/v1` 接口调用 LLM、embedding、VLM 和 reranker。
- `openai_compatible`: 通过任意本地 OpenAI-compatible 服务调用模型。

这层不会替代 MinerU。MinerU 仍然负责 PDF 解析；模型网关负责视觉 caption、embedding、可选模型 rerank 和最终答案生成。这样同一套后端和离线 pipeline 可以在云端 API、Xinference、本地部署之间切换。

## GraphRAG Layer

GraphRAG 是当前结构升级的重点。它不是替代向量检索，而是把检索结果放回文档图和语义图里进行关系传播。

`scripts/23_build_graphrag.py` 会生成：

```text
outputs/graphrag/nodes.jsonl
outputs/graphrag/edges.jsonl
outputs/graphrag/entities.jsonl
outputs/graphrag/relations.jsonl
outputs/graphrag/entity_links.jsonl
outputs/graphrag/communities.jsonl
outputs/graphrag/summary.jsonl
```

GraphRAG 由三层组成：

- 文档结构图：页面、章节、图表、图注、正文引用关系。
- 语义实体图：文档主题、章节概念、关键术语、图像实体之间的关系。
- 社区摘要图：按文档章节或页面聚合节点，形成可解释的局部知识单元。

GraphRAG 输出兼容原有 rerank 的 KG 接口，因此 `outputs/graphrag/entities.jsonl` 和 `outputs/graphrag/relations.jsonl` 可以直接作为 `--kg-dir` 输入。

## Retrieval Layer

系统支持多路召回：

- `bm25`: 关键词和短语匹配。
- `lexical`: 字符级和词级相似度。
- `embedding`: 语义向量召回。
- `visual`: 图片 caption、OCR、视觉证据召回。
- `kg` / `graphrag`: 基于实体和关系的结构召回。
- `fusion`: 多路召回融合。

召回阶段负责覆盖率，重排阶段负责精度和证据结构质量。

## MultiRank Reranking

重排序分为 G0-G4：

- `G0`: 原始召回排序。
- `G1`: 语义相似度。
- `G2`: 语义相似度 + PPR 图传播。
- `G3`: G2 + bridge + reference。
- `G4`: G3 + visual grounding + evidence chain + GraphRAG/KG 信号 + 可选模型 rerank 信号。

G4 是当前系统的主策略，适合需要图文证据、上下文连续性、跨页面引用和实体关系约束的场景。

## Evidence Chain

答案生成前会先组织证据链：

- 主证据节点。
- 同页或同章节上下文。
- 图表、图片、图注补充证据。
- GraphRAG 中的实体、社区和关系证据。

最终输出既可以是普通问答结果，也可以是前端使用的证据链和证据卡片。
