# MultiRank-RAG

面向复杂 PDF 文档的多模态 RAG、GraphRAG 与证据链问答系统。

MultiRank-RAG 的目标不是只返回一个答案，而是从复杂文档中检索、组织并展示一条可追溯的证据链。系统会把 PDF 中的文本、标题、表格、图片、图注、公式、页面布局等信息统一建模为 evidence nodes，再通过混合召回、多路融合、GraphRAG、查询自适应重排序和自我修正机制，生成带证据引用的最终回答与可视化证据卡片。

本项目适用于产品手册、售后知识库、财报/年报、教材、医疗指南、科研论文、图文混排资料等复杂文档场景。仓库中的样例数据和历史实验文件只作为工程参考，项目本身定位为通用的复杂文档多模态知识问答系统。

## 核心特点

- **复杂 PDF 解析**：支持 MinerU 云端/本地解析，也保留轻量级本地解析路径；尽量保留页码、版面、bbox、段落、章节、表格、图片、图注等结构信息。
- **结构化 Chunk 与统一证据节点**：将文本块、标题、表格、图片、公式、图注、页面上下文统一为可检索的 evidence node，避免只做粗粒度文本切块。
- **多模态证据增强**：可接入 QwenVL、Doubao Vision、Xinference 或 OpenAI-compatible 服务，为图片生成 OCR、caption、visual summary、关键对象和面向问答的视觉证据。
- **混合召回与多路融合**：支持 embedding、BM25、lexical、visual、table、GraphRAG/KG、reference、section 等多路召回，并通过 RRF/加权融合形成候选证据集。
- **GraphRAG 结构推理**：构建文档结构边、跨模态邻接边、实体关系、语义关系和 community summary，让系统能够处理跨页、跨图表、跨章节的问题。
- **MultiRank 重排序**：提供 G0-G4 / V0-V4 消融框架，逐步加入语义相似、图传播、bridge evidence、reference matching、视觉 grounding 和证据链约束。
- **证据链自我修正**：在主线检索结果与多路召回结果之间做二次验证，优先保留稳定 Top evidence，只在证据缺模态或置信不足时合并/替换候选。
- **前后端闭环**：FastAPI 后端支持 PDF 上传、解析、检索、证据链生成；React 前端支持文档选择、提问、答案展示、证据卡片与相关性排序。
- **可评测、可消融、可复现**：提供从数据准备、检索评测、证据链评测到实验汇总 Excel 的脚本，方便定位每个模块的真实贡献。

## 总体框架

```text
PDF / Manual / Knowledge Base
        |
        v
1. PDF Parsing
   MinerU cloud/local parser, native fallback
   page / section / bbox / layout / table / figure / caption
        |
        v
2. Evidence Node Construction
   text, title, table, figure, caption, equation, page context
   unified node schema + document edges
        |
        v
3. Multimodal Enhancement
   image crops, OCR, caption, visual summary, QA evidence
   QwenVL / Doubao Vision / Xinference / OpenAI-compatible
        |
        v
4. GraphRAG Index
   structure graph, cross-modal links, entity graph,
   semantic relations, community summaries
        |
        v
5. Hybrid Retrieval
   embedding + BM25 + lexical + visual + table + KG + section routes
        |
        v
6. MultiRank Reranking
   G0 raw retrieval
   G1 semantic rerank
   G2 semantic + graph propagation
   G3 bridge/reference evidence
   G4 adaptive multimodal evidence rerank
        |
        v
7. Evidence Self-Correction
   primary / fallback / merge verifier
   modality guard + chain completeness check
        |
        v
8. Grounded Answer & Evidence Card
   cited answer, <PIC:node_id> visual markers,
   evidence chain, ranked support evidence, frontend visualization
```

## 完整流程

### 1. 文档接入

用户可以通过前端上传 PDF，也可以在离线实验中把 PDF 放入本地数据目录。后端会为每次上传创建独立 job，将原始 PDF、解析结果、候选证据、重排序结果、证据链和可视化文件写入 `outputs/` 下的运行目录。

### 2. PDF 解析与结构保留

解析层优先使用 MinerU。MinerU 可以输出更完整的版面结构、表格、图片和 OCR 信息；当未配置 MinerU API 或本地 MinerU 时，系统仍可走内置轻量解析路径完成基本文本节点构建。

解析后的核心信息包括：

- 文档 ID、页码、章节标题、阅读顺序
- 文本块、标题块、表格块、图片块、图注块、公式块
- bbox 坐标、页面宽高、布局邻接关系
- 表格文本、图片路径、视觉描述、跨模态引用关系

### 3. 结构化 Chunk 与 evidence node

系统不是简单按固定长度切文本，而是根据 PDF 结构生成节点：

- 标题与段落保留章节上下文
- 表格节点保留行列文本、caption、页面位置
- 图片节点保留图注、OCR、visual summary、页面上下文
- 相邻文本、图表、图注之间建立边
- 跨页引用、同节关系、实体关系进入 GraphRAG

这种设计让一个问题可以同时命中文本、表格、图片和它们之间的关系。

### 4. 多模态增强

图片类节点会被进一步增强：

- 裁剪 PDF 页面中的图片区域
- 抽取 OCR 文本
- 生成图片 caption 和视觉摘要
- 识别关键对象、图表趋势、界面标识
- 生成 `qa_evidence`，让图片内容更适合问答检索

可选模型后端包括：

```text
ark                 直接调用火山方舟 / 豆包等服务
xinference          本地或远程 Xinference OpenAI-compatible 网关
openai_compatible   任意 OpenAI-compatible 服务
local               轻量本地 fallback
```

### 5. GraphRAG 构建

GraphRAG 层把文档从一组孤立 chunk 扩展为可推理图：

- `structure edges`：同页、相邻块、标题-正文、图片-图注、表格-说明
- `reference edges`：文本引用图表、图表回链文本
- `entity links`：实体、术语、产品型号、指标名称
- `semantic relations`：相似概念、同主题节点、跨页延续
- `community summaries`：局部主题社区摘要

这些图信号会参与候选扩展、PPR 图传播、bridge evidence 搜索和最终证据链组织。

### 6. 多路召回与融合

系统支持多种召回路线：

- `embedding`：语义向量召回，可接入 Doubao embedding-vision、Xinference 或本地模型
- `bm25`：关键词与稀疏检索
- `lexical`：细粒度词面匹配
- `visual`：图片 caption、OCR、visual summary 召回
- `table`：表格字段、行列结构、数值文本召回
- `kg` / `graph`：GraphRAG 实体与关系召回
- `reference`：图表引用、标题引用、跨页引用召回
- `section`：同节上下文召回

多路召回结果通过融合策略进入候选池，既保证覆盖率，又避免单一 embedding 或单一 BM25 的偏差。

### 7. MultiRank 重排序

MultiRank 是项目的核心排序框架。当前主线使用 G0-G4 分层消融：

| 方法 | 作用 |
|---|---|
| G0 | 原始候选召回结果 |
| G1 | 加入语义相似度与基础相关性 |
| G2 | 加入 GraphRAG / PPR 图传播信号 |
| G3 | 加入 bridge evidence、reference matching 和跨模态关系 |
| G4 | 加入查询自适应权重、视觉 grounding、表格结构信号和证据链约束 |

G4 会根据问题类型自动调整权重。例如：

- 图片定位类问题提高 visual grounding 权重
- 表格/数值类问题提高 table route 权重
- 跨文档/跨页面问题提高 GraphRAG 和 bridge evidence 权重
- 普通文本问答保留语义相似和关键词匹配的稳定性

### 8. 证据链生成与自我修正

系统会从 Top evidence 中组织固定长度证据链，而不是只输出分数最高的一个 chunk。证据链包含：

- 主回答证据
- 同页或同节上下文
- 图表/图注/文本互补证据
- GraphRAG bridge evidence
- 可视化证据引用

当前加入了 `SelfCorrect merge-v2` 自我修正机制：

```text
primary result   = 稳定主线 ABECD + Guard
fallback result  = 多路召回 balanced
verifier         = 检查模态覆盖、visual grounding、chain completeness
action           = primary / fallback / merge
```

默认策略是优先保留主线排序，只在缺少必要模态或证据链不完整时合并 fallback 的图片、表格或 bridge evidence。这比简单替换整套结果更稳定。

### 9. 回答生成与证据卡片

最终答案由证据链驱动生成，支持：

- 文本答案
- `<PIC:node_id>` 图片证据标记
- 证据节点引用
- 多步证据链
- 前端横版证据卡片
- 右侧相关性排序展示

前端页面重点不是聊天框，而是“问题 -> 答案 -> 证据链 -> 相关性排序”的可解释闭环。

## 当前推荐主线

当前推荐使用：

```text
ABECD + Evidence Guard + SelfCorrect merge-v2
```

对应能力：

- A：上下文扩展
- B：查询自适应重排序
- E：GraphRAG 图结构增强
- C：表格结构化增强
- D：视觉 evidence 增强
- Guard：证据链完整性约束
- SelfCorrect：主线结果与多路召回结果的二次验证和合并

在当前 100 题抽样实验中，`SelfCorrect merge-v2` 相比主线提升了 Recall@1、Recall@5、MRR、nDCG@5 和证据链分数，是目前最稳的默认方案。

## 目录结构

```text
backend/      FastAPI 后端，负责上传、解析、检索、证据链和文件服务
web/          React + Vite 前端，负责文档选择、提问、证据卡片和相关性展示
scripts/      离线 pipeline、实验、评测、诊断与数据处理脚本
configs/      环境配置示例、模型配置示例
data/         小规模样例数据与问题文件，不存放私有 PDF
docs/         架构、实验、GraphRAG、模型网关和报告材料
outputs/      运行产物目录，默认不提交到 Git
external/     外部依赖或临时克隆仓库，默认不提交到 Git
```

重要文档：

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：系统架构说明
- [docs/GRAPHRAG.md](docs/GRAPHRAG.md)：GraphRAG 设计
- [docs/MODEL_GATEWAY.md](docs/MODEL_GATEWAY.md)：Ark、Xinference、OpenAI-compatible 配置
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)：实验设计
- [docs/EVALUATION.md](docs/EVALUATION.md)：评测指标
- [scripts/README.md](scripts/README.md)：离线脚本入口
- [backend/README.md](backend/README.md)：后端接口说明
- [web/README.md](web/README.md)：前端运行说明

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果使用前端：

```bash
cd web
npm install
```

### 2. 配置环境变量

复制环境模板：

```bash
cp .env.example .env
```

在 `.env` 中填写本地配置。真实 API key 只放在 `.env`，不要写入代码或 README。

常用配置示例：

```text
RAG_PDF_PARSER=mineru
MINERU_API_MODE=cloud
MINERU_API_URL=https://mineru.net/api/v4
MINERU_API_KEY=<your-mineru-api-key>

RAG_MODEL_PROVIDER=ark
RAG_EMBEDDING_PROVIDER=ark
RAG_EMBEDDING_MODEL=doubao-embedding-vision-250615
RAG_ANSWER_PROVIDER=ark
RAG_ANSWER_MODEL=<your-answer-model-endpoint>

RAG_KG_DIR=outputs/graphrag
RAG_BACKEND_CANDIDATE_RETRIEVER=fusion
RAG_BACKEND_RERANK_RETRIEVER=fusion
```

如需统一走 Xinference：

```text
RAG_MODEL_PROVIDER=xinference
RAG_EMBEDDING_PROVIDER=xinference
RAG_VISUAL_CAPTION_PROVIDER=xinference
RAG_ANSWER_PROVIDER=xinference
XINFERENCE_BASE_URL=http://127.0.0.1:9997/v1
```

### 3. 运行离线 pipeline

轻量样例：

```bash
python scripts/06_run_pipeline.py \
  --sample \
  --candidate-k 10 \
  --rerank-k 3 \
  --rerank-methods G4
```

完整主线实验：

```bash
python scripts/40_run_main_experiment.py \
  --dataset-name sample \
  --variants V4 \
  --build-chains \
  --generate-answers \
  --answer-provider none
```

如果已经配置模型服务，可以把 `--answer-provider none` 改为 `ark`、`xinference` 或 `openai_compatible`。

### 4. 启动后端

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

核心接口：

```text
GET  /api/health
POST /api/analyze
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/files/{path}
```

### 5. 启动前端

```bash
cd web
npm run dev
```

默认访问：

```text
http://127.0.0.1:5173
```

前端支持：

- 上传 PDF
- 选择系统内置文档
- 输入自定义问题
- 选择预设问题
- 查看回答
- 查看横版证据卡片
- 查看证据相关性排序

## 主要脚本

```text
01_parse_pdf.py                 PDF 解析与 evidence node 构建
02_build_graph.py               文档结构图构建
03_retrieve_candidates.py       候选证据召回
04_rerank.py                    G0-G4 MultiRank 重排序
05_evaluate.py                  检索与排序指标评测
09_build_evidence_chains.py     证据链生成
10_build_visual_evidence.py     图片 OCR/caption/visual evidence 增强
11_build_evidence_cards.py      证据卡片生成
12_evaluate_evidence_chains.py  证据链指标评测
23_build_graphrag.py            GraphRAG 实体、关系、社区构建
34_generate_chain_answers.py    基于证据链生成最终答案
40_run_main_experiment.py       主实验入口
42_enhance_multimodal_nodes.py  表格与视觉节点增强
50_export_experiment_summary.py 实验结果汇总为 Excel
52_self_correct_evidence.py     证据链自我修正
```

## 输出产物

运行产物默认写入 `outputs/`，该目录默认被 Git 忽略。

```text
outputs/parsed/nodes.jsonl                    evidence nodes
outputs/parsed/edges.jsonl                    document edges
outputs/graphrag/entities.jsonl               GraphRAG entities
outputs/graphrag/relations.jsonl              GraphRAG relations
outputs/graphrag/communities.jsonl            community summaries
outputs/rankings/candidates.csv               retrieved candidates
outputs/rankings/reranked.csv                 reranked evidence
outputs/evidence_chains/chains.jsonl          evidence chains
outputs/evidence_chains/answers.csv           grounded answers
outputs/evidence_cards/                       rendered evidence cards
outputs/reports/*.xlsx                        experiment summaries
```

## 评测方法

项目把评测拆成两层，避免只看“有没有命中某个 chunk”：

### 检索与排序指标

- Recall@1 / Recall@3 / Recall@5 / Recall@10
- MRR
- nDCG@5
- evidence hit
- modality hit
- citation correctness
- visual grounding hit

### 证据链指标

- chain present
- average step count
- gold node coverage
- gold page hit
- gold modality coverage
- visual grounding hit
- cross-modal hit
- relation support
- evidence chain score

汇总实验：

```bash
python scripts/50_export_experiment_summary.py
```

该脚本会生成：

```text
outputs/reports/openragbench_experiment_summary_*.xlsx
outputs/reports/openragbench_experiment_overview_*.csv
```

## 工程边界与安全

- `.env`、API key、上传 PDF、输出结果、外部仓库和大模型权重不会提交到 Git。
- `outputs/` 只保留 `.gitkeep` 和说明文件，所有实验结果都作为本地运行产物管理。
- `DataFountain/`、`external/`、用户上传文件和私有文档默认被忽略。
- README 中的模型名称和 API key 字段均为占位说明，真实配置请写入本地 `.env`。

## 项目亮点总结

MultiRank-RAG 的核心价值在于把复杂文档问答从“文本 chunk 检索”升级为“多模态证据组织”：

1. 用 MinerU 和结构化 chunk 保留 PDF 的真实版面关系。
2. 用视觉模型把图片、图表和页面区域转为可检索证据。
3. 用 GraphRAG 建模跨页、跨图表、跨章节的关系。
4. 用多路召回提高证据覆盖率。
5. 用 MultiRank 在不同问题类型下自适应选择证据。
6. 用自我修正机制平衡稳定性和覆盖率。
7. 用证据链和证据卡片让最终答案可解释、可检查、可展示。

## 当前状态

项目已经形成完整研究原型：

- 后端 PDF 上传与任务处理可用
- 前端证据展示页面可用
- PDF 解析、视觉增强、GraphRAG、混合召回、MultiRank 重排序、自我修正、证据链生成与评测脚本均已接入
- 当前推荐主线为 `ABECD + Evidence Guard + SelfCorrect merge-v2`

后续优化方向包括更强的视觉 grounding、更精细的表格结构理解、更稳定的本地模型部署、更真实的多轮对话记忆，以及面向业务场景的 answer-level LLM judge 评测。
