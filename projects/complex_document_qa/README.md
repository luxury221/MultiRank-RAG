# Complex Document QA System

这一部分是仓库的主项目：面向复杂 PDF / 图文混排资料的多模态 RAG、GraphRAG 与证据链问答系统。

主项目保留原有代码路径，避免破坏已经可运行的前后端和脚本入口。本目录用于把项目说明、报告和关键入口集中到一起，方便从 GitHub 上理解这是一个完整作品，而不是零散实验文件。

## Scope

主项目关注：

- PDF 上传、解析、结构化 chunk 与 evidence node 构建
- 图片、表格、公式、图注和页面关系的多模态证据建模
- embedding / BM25 / lexical / visual / table / GraphRAG 多路召回
- MultiRank 自适应重排序、证据链生成和答案级自我修正
- FastAPI 后端与 React 前端的完整演示闭环
- 证据卡片、证据相关性排序和可解释问答展示

## End-to-End Chain

复杂文档问答主链路如下：

```text
PDF / benchmark document
  -> parsing and layout extraction
  -> evidence node construction
  -> visual/table/text enrichment
  -> document graph and GraphRAG index
  -> multi-route retrieval
  -> MultiRank G4 reranking
  -> evidence-chain generation
  -> grounded answer and frontend evidence cards
```

### 1. Data Processing

系统优先使用 MinerU 解析复杂 PDF，保留页码、章节、阅读顺序、bbox、表格、图片、图注和 OCR 等结构信息。解析结果不会只切成普通文本 chunk，而是统一转换为 evidence node：

- `text`：正文段落、标题、页面上下文。
- `table`：表格文本、行列结构、caption、数值证据。
- `figure`：图片、页面截图、crop 区域、图注、OCR 和视觉摘要。
- `caption / title / context`：连接正文、图表和页面结构的辅助证据。

公开 benchmark 也会先转换到同一格式：`nodes.jsonl` 存证据节点，`questions.csv/jsonl` 存问题、标准证据、标准模态和页码信息。

### 2. Multimodal Evidence

图片和页面节点会被增强为可检索证据。系统可以接入 QwenVL、Doubao Vision、Xinference 或任意 OpenAI-compatible 视觉模型，生成：

- `crop_image_path` / `page_image_path`
- `visual_caption`
- `ocr_text`
- `visual_summary`
- `key_objects`
- `data_or_trends`
- `qa_evidence`

这一步让图片不再只是文件路径，而是能参与召回、重排序、证据链和前端展示的结构化证据。

### 3. Similarity and Retrieval

召回阶段不是单一路线，而是多路融合：

- `embedding`：文本或视觉 embedding 召回，视觉页面可使用 `doubao-embedding-vision-250615`。
- `bm25`：关键词稀疏检索，适合文本事实、条款、术语问题。
- `lexical`：字符 ngram / TF-IDF 相似度，提升中文短语和术语匹配稳定性。
- `visual`：基于 caption、OCR、visual summary 和图像字段的视觉路线。
- `table`：面向表头、行列、数值和财务表格的结构化路线。
- `kg / graph`：GraphRAG 实体、关系和社区摘要召回。
- `reference / section`：图表引用、同页、同节和上下文扩展路线。

多路结果通过 RRF/加权融合形成候选池。直观上，如果一个证据同时被 embedding、BM25、视觉路线和图结构路线命中，它会比单一路线命中的证据更稳定。

### 4. Graph and Reranking

图构建会生成同页、相邻块、同章节、图表-caption、表格-caption、跨模态邻接、实体关系和语义关系。重排序阶段使用 MultiRank：

| Method | Signal |
|---|---|
| G0 | 原始召回顺序 |
| G1 | 相似度相关性 |
| G2 | 相似度 + GraphRAG/PPR 图传播 |
| G3 | 图传播 + bridge evidence + reference matching |
| G4 | G3 + 视觉 grounding + 表格结构 + 模态匹配 + 证据链适配 |

当前线上推荐主线是 `V5-online-quality`：`multiroute retrieval + G4 rerank + context expansion + adaptive rerank boost + graph context boost + evidence guard`。

### 5. Evidence Chain and Answer

系统不会只把 top-1 chunk 交给大模型，而是组织证据链：

- 主回答证据。
- 同页、同节、相邻上下文。
- 相关表格、图片、图注和页面 crop。
- GraphRAG bridge evidence。
- 经过 Evidence Guard 过滤后的可靠补充证据。

答案生成时使用证据链作为上下文，并在前端展示证据卡片、图片标记、来源页码和相关性排序，形成“问题 -> 答案 -> 证据链 -> 证据卡片”的闭环。

## Evaluation Snapshot

公开评测统一记录在根目录 README 的“公开评测与消融结果”中，覆盖文本手册、表格数值推理、长文档视觉 grounding 和多跳证据链四类任务。当前最适合汇报的结论是：

| Dataset | Best / Key Setting | Main Result |
|---|---|---|
| RAGBench eManual | BM25 V0-V5 | Hit@5/10 已达到 1.000，说明简单文本手册任务接近饱和，可作为负控制。 |
| T2/FinQA | BM25 V5 | nDCG@5 从 0.678 提升到 0.878，体现表格与上下文感知重排收益。 |
| MMLongBench-Doc | Doubao Vision Embedding V2/V4/V5 | Hit@5 从 BM25 的 0.070 提升到 0.910，Strict Visual@5 达到 0.910。 |
| MultiHop-RAG | BM25 V5 | evidence-chain score 达到 0.890，gold node coverage 达到 0.730。 |

这组结果共同说明：系统在简单文本任务上保持稳定，在表格/多跳任务上体现重排和证据链收益，在视觉长文档任务上通过页面级视觉 embedding 打开多模态 grounding 能力。

## Main Paths

```text
backend/                         FastAPI 后端，负责上传、任务、检索与问答接口
web/                             React + Vite 前端，负责复杂文档问答展示
multirank_rag/                   核心 Python package
scripts/01-52_*.py               主项目解析、检索、重排、证据链和评测脚本
configs/                         模型、MinerU、Xinference 与 API 配置模板
docs/                            主项目架构、GraphRAG、模型网关和实验文档
data/pdfs/                       本地复杂文档样例位置，真实 PDF 默认不提交
data/sample/                     可提交的小样例数据和前端预览问题
outputs/                         本地运行产物，默认不提交
projects/complex_document_qa/    主项目索引、报告和作品材料
```

## Report

开题报告已经归档到：

```text
projects/complex_document_qa/reports/开题报告.docx
```

## Run

后端：

```bash
cd backend
python app.py
```

前端：

```bash
cd web
npm install
npm run dev
```

主 pipeline：

```bash
python scripts/06_run_pipeline.py --questions data/questions.csv
```

## Positioning

本目录是复杂文档 QA 项目的展示入口。报告、README、主链路和评测结果都围绕复杂 PDF / 图文混排资料的多模态证据问答展开。
