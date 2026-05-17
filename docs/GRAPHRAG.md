# GraphRAG Design

GraphRAG 在本项目中的定位是“结构增强层”。它不替代 embedding，也不替代视觉模型，而是把 PDF 中分散的证据节点组织成图，让检索和重排能够利用页面、章节、图表、图注、实体和社区关系。

## Why GraphRAG Fits This Project

复杂 PDF 和图文知识问答常见问题：

- 答案文本、表格和图片经常不在同一个 chunk。
- 图注、正文引用和图片本体之间需要结构连接。
- 跨页问题需要从多个局部证据中合成答案。
- 向量相似度容易召回“语义相近但证据关系弱”的节点。

GraphRAG 可以让系统从“找相似文本”升级为“找相关证据子图”。

## Implementation

核心脚本：

```text
scripts/23_build_graphrag.py
```

输入：

```text
outputs/parsed/nodes.jsonl
outputs/parsed/edges.jsonl
```

输出：

```text
outputs/graphrag/nodes.jsonl          GraphRAG 节点索引
outputs/graphrag/edges.jsonl          文档结构边
outputs/graphrag/entities.jsonl       语义实体
outputs/graphrag/relations.jsonl      实体关系
outputs/graphrag/entity_links.jsonl   节点-实体链接
outputs/graphrag/communities.jsonl    局部社区摘要
outputs/graphrag/summary.jsonl        图统计信息
```

运行：

```bash
python scripts/23_build_graphrag.py \
  --nodes outputs/parsed/nodes.jsonl \
  --edges outputs/parsed/edges.jsonl \
  --output-dir outputs/graphrag
```

完整 pipeline 会自动执行该步骤：

```bash
python scripts/06_run_pipeline.py --questions data/questions.csv
```

## Graph Layers

### 1. Document Structure Graph

来源于 PDF 版面和文档结构：

```text
page -> node
section -> text/table/figure/caption
text -> referenced figure/table
figure/table -> caption
chunk -> next chunk
```

这层主要提升跨 chunk、跨页面和图表引用问题。

### 2. Semantic Entity Graph

从 evidence node 中抽取：

- 文档主题。
- 章节概念。
- 关键术语。
- 图表编号。
- 图片实体。
- OCR 和视觉 caption 中的关键对象。

实体和节点之间建立链接，实体之间通过共现、图像描绘、章节包含等关系连接。

### 3. Community Summary Graph

系统按文档、章节或页面聚合 evidence nodes，生成局部社区：

```text
community_id
doc_id
section / page label
node_ids
entity_ids
modalities
top_terms
summary
```

这层适合做演示和诊断：可以说明某个答案来自哪一个局部知识社区。

## How It Enters RAG

GraphRAG 结果兼容原有 KG 接口：

```bash
--kg-dir outputs/graphrag
```

在 MultiRank 中的作用：

- G2：使用 PPR 在文档图上传播 query relevance。
- G3：使用 bridge 和 reference 找到主证据附近的图表/上下文。
- G4：加入 visual grounding、evidence chain 和 GraphRAG/KG 信号，降低无关图片和弱相关文本的排序。

因此当前项目可以对外描述为：

```text
Hybrid Retrieval + GraphRAG-aware MultiRank Reranking + Evidence Chain Generation
```

## Resume-Friendly Description

可以写成：

> 设计并实现面向复杂 PDF 的多模态 GraphRAG 问答系统，将文档解析结果统一为 evidence node，构建文档结构图、语义实体图和社区摘要图，并在重排序阶段融合 BM25、embedding、PPR 图传播、跨模态 bridge、视觉 grounding 与 GraphRAG/KG 信号，最终输出可追溯证据链和证据卡片。

更技术化的版本：

> Built a multimodal GraphRAG pipeline for complex PDF QA, including layout-aware evidence node construction, document graph modeling, entity-relation extraction, community summaries, hybrid retrieval, graph-aware reranking, and grounded evidence-chain generation.

## Current Limitations

- 当前实体抽取是轻量规则方式，适合作为工程原型和可解释结构层。
- 对医学、金融、数学等高专业领域，后续可以接入领域词表或 LLM entity extraction。
- 社区摘要目前是结构化摘要，后续可接入 LLM 生成更自然的 community report。
- GraphRAG 应作为增强信号使用，不应完全替代 embedding 和视觉模型。
