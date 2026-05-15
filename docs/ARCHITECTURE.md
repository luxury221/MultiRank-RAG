# Architecture

MultiRank-RAG 的核心目标是把复杂 PDF 中的多模态信息转化为可检索、可重排、可追溯的证据链。系统不只关注“回答是什么”，也关注“回答依据来自哪里”。

## End-to-End Flow

```text
PDF documents
  -> PDF parsing
  -> evidence node construction
  -> visual evidence enrichment
  -> index and graph construction
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

这种统一 schema 是后续检索、重排、证据链和前端展示的基础。

## Visual Evidence Layer

视觉增强模块会对图片和页面裁剪图生成结构化视觉字段：

- `visual_caption`: 图片内容说明。
- `ocr_text`: 图片内文字。
- `key_objects`: 关键对象、按钮、部件、符号。
- `qa_evidence`: 适合问答使用的视觉证据摘要。

视觉模型负责理解图片内容，embedding 模型负责将文本和图片证据映射到统一检索空间。两者职责不同，可以组合使用。

## Retrieval Layer

系统支持多路召回：

- `bm25`: 关键词和短语匹配。
- `lexical`: 字符级和词级相似度。
- `embedding`: 语义向量召回。
- `visual`: 图像 caption、OCR、视觉证据召回。
- `kg`: 产品、部件、动作、故障和政策关系召回。
- `fusion`: 多路召回融合。

召回阶段负责覆盖率，重排阶段负责精度。

## GraphRAG Layer

轻量知识图谱从 evidence node 中抽取实体和关系：

```text
product, part, action, fault, policy, image
```

典型关系：

```text
product_has_part
product_supports_action
action_targets_part
fault_solved_by_action
image_depicts_part
image_illustrates_action
policy_applies_to_product
```

KG 不替代向量检索，而是给检索和重排提供结构约束，减少“语义相近但产品不对”“图片存在但不支撑答案”的问题。

## MultiRank Reranking

重排序分为 G0-G4：

- `G0`: 原始召回排序。
- `G1`: 语义相似度。
- `G2`: 语义相似度 + PPR 图传播。
- `G3`: G2 + bridge + reference。
- `G4`: G3 + visual grounding + evidence chain + domain/product/KG signals。

G4 是当前系统的主要重排策略，适合需要图文证据、上下文连续性和产品一致性的场景。

## Answer and Evidence Chain

答案生成前会先组织证据链：

- 主证据节点。
- 同页或同章节上下文。
- 图片、表格、图注补充证据。
- KG 中的部件、动作和故障关系。

最终输出既可以是普通问答结果，也可以是前端使用的证据链和证据卡片。

