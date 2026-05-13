# MultiRank-RAG

面向复杂 PDF、产品手册和图文混合知识库的 **多模态 GraphRAG 证据链问答系统**。项目将文档解析、视觉理解、图文向量化、知识图谱、多路召回、分层重排和答案生成整合为一条可落地的工程链路，重点解决“答案是否有依据、图片是否真正支撑回答、证据之间如何关联”的问题。

系统不是简单地把文档切片后交给 LLM，而是将文本、表格、图片、图注、版面位置、章节结构和实体关系统一建模为可追溯证据节点，并通过多模态检索与图结构增强完成高可信问答。

## Highlights

- **多模态文档解析**：基于 MinerU 解析 PDF，保留文本、标题、表格、图片、公式、图注、bbox、页码和章节层级。
- **图文协同理解**：使用 Qwen-VL / 豆包视觉模型生成图片语义、OCR、关键对象和问答证据，再通过 `doubao-embedding-vision` 进行图文联合向量化。
- **GraphRAG 证据建模**：构建文档证据图和轻量知识图谱，覆盖产品、部件、动作、故障、政策、图片等实体关系。
- **多路融合召回**：融合 BM25、lexical、embedding、visual、layout、section、reference、product、KG 等检索信号。
- **MultiRank 分层重排**：在语义相似度基础上叠加 PPR 图传播、跨模态 bridge、视觉 grounding、显式引用和证据链完整度。
- **证据链驱动生成**：先组织可追溯证据链，再进行答案生成、图片证据回传和输出校验。
- **工程化落地**：提供脚本流水线、FastAPI 服务、Web 可视化界面、缓存机制和可复现实验产物。

## Architecture

```text
PDF / 图文资料
   |
   v
MinerU Parser
   |-- text / table / figure / caption / title / equation
   |-- bbox / page / section / source_ref
   v
Evidence Node Builder
   |-- 统一节点 schema
   |-- chunk context
   |-- explicit refs
   v
Visual Grounding
   |-- page rendering
   |-- crop extraction
   |-- Qwen-VL / Doubao vision caption
   |-- OCR / key objects / QA evidence
   v
Index Layer
   |-- text embedding
   |-- multimodal embedding
   |-- BM25 / lexical index
   |-- visual index
   v
GraphRAG Layer
   |-- evidence graph
   |-- entity graph
   |-- product / part / action / fault / policy / image relations
   v
Hybrid Retrieval
   |-- BM25
   |-- embedding
   |-- visual
   |-- layout
   |-- section / reference
   |-- product / KG
   v
MultiRank Reranking
   |-- semantic similarity
   |-- PPR
   |-- bridge score
   |-- visual grounding
   |-- evidence chain
   |-- domain / product / KG signals
   v
Evidence Chain + Answer Generation
   |-- grounded prompt
   |-- self-check
   |-- PIC evidence suffix
   |-- API / UI / CSV export
```

## Core Pipeline

### 1. 文档解析与证据节点构建

入口：`scripts/01_parse_pdf.py`

系统读取 PDF 后调用 MinerU 解析页面结构，将原始文档转换为统一的 evidence node。每个节点保留来源、页码、章节、上下文、bbox、图表引用和视觉字段，后续检索、重排、证据链生成都围绕该节点 schema 展开。

主要节点类型：

```text
page      页面节点
title     标题 / 章节节点
text      正文 chunk
table     表格节点
figure    图片 / 图示节点
caption   图注 / 表注节点
equation  公式节点
```

关键字段：

```text
node_id, doc_id, page, node_type, content, source_ref,
section, parent_chunk_id, previous_node_id, next_node_id,
bbox, crop_image_path, page_image_path,
visual_caption, key_objects, ocr_text, qa_evidence, visual_summary
```

### 2. 视觉证据增强

入口：`scripts/10_build_visual_evidence.py`

视觉链路会先渲染 PDF 页面，再根据 bbox 生成局部裁剪图。对图片、表格和图示节点，系统调用视觉模型生成结构化语义字段，使图片不再只是文件路径，而是可以参与检索、重排和答案生成的证据。

推荐链路：

```text
page image / crop image
   -> Qwen-VL / Doubao vision
   -> visual_caption / OCR / key_objects / qa_evidence
   -> doubao-embedding-vision multimodal embedding
```

视觉模型负责“理解图片内容”，`doubao-embedding-vision` 负责“把图片和文本编码到统一向量空间”。两者分工明确，可以同时启用。

### 3. 图文索引与多路召回

入口：`scripts/03_retrieve_candidates.py`  
核心实现：`scripts/embedding_index.py`、`scripts/rerank_lib.py`

系统默认使用 `fusion` 召回策略，将语义向量、关键词、视觉语义、版面结构、章节引用和图谱关系合并为候选池。

支持的召回路线：

```text
bm25        关键词与短语匹配
lexical     字符级 / 词级相似度
embedding   文本语义向量召回
visual      caption / OCR / crop grounding
layout      bbox / 页面位置 / 图表区域
section     章节结构召回
reference   显式图表引用召回
product     产品名、型号、别名匹配
kg          实体关系召回
fusion      多路融合召回
```

问题会先做意图路由，例如 policy、manual、manual_visual、troubleshooting、visual、text_fact。不同问题类型使用不同权重，避免图片证据干扰政策类问题，也避免纯文本检索漏掉操作图示。

### 4. GraphRAG 知识增强

入口：`scripts/23_build_kg.py`

系统从 evidence node 中抽取实体与关系，构建轻量知识图谱，并将图谱信号注入召回和重排阶段。

实体类型：

```text
product   产品 / 文档对象
part      部件
action    操作动作
fault     故障现象
policy    规则 / 售后 / 服务意图
image     图片证据
```

关系类型：

```text
product_has_part
product_supports_action
product_has_fault
product_has_image
action_targets_part
fault_solved_by_action
image_illustrates_action
image_depicts_part
policy_applies_to_product
```

KG 不替代向量检索，而是作为结构化约束信号，解决“语义相近但产品不对”“答案流畅但证据不支撑”“图片和动作不匹配”等问题。

### 5. MultiRank 分层重排

入口：`scripts/04_rerank.py`  
核心实现：`scripts/rerank_lib.py`

重排阶段采用 G0-G4 分层策略：

```text
G0 = 原始召回顺序
G1 = semantic similarity
G2 = similarity + PPR
G3 = similarity + PPR + bridge + reference
G4 = G3 + visual + evidence chain + domain + product + KG
```

G4 会融合以下信号：

- `sim_score`：问题与节点的语义相似度
- `ppr_score`：证据图上的 PageRank 传播分数
- `bridge_score`：跨模态或跨节点桥接能力
- `ref_score`：显式图表引用匹配
- `visual_score`：图片、OCR、caption、bbox 与问题的匹配程度
- `chain_score`：候选节点与上下文证据的连贯性
- `domain_score`：问题意图与节点领域的匹配
- `product_score`：产品、型号、对象一致性
- `kg_score`：实体与关系图谱信号

### 6. 证据链与答案生成

入口：`scripts/09_build_evidence_chains.py`、`scripts/21_generate_answer.py`

答案生成前会先组织 evidence chain，包括主证据、同页/同章节上下文、图注、表格、图片和 KG 补充证据。LLM 只基于证据回答，图片类问题会保留 `<PIC> ;["image_id"]` 格式，便于外部系统定位图片来源。

后处理模块包含：

- 图片后缀规范化
- 低置信答案识别
- 截断检测
- 候选答案裁决
- 保守回退
- 高风险答案精修

## Repository Layout

```text
backend/      FastAPI 服务与异步任务编排
web/          React + Vite 前端可视化
scripts/      解析、索引、召回、重排、证据链、答案生成脚本
configs/      环境变量模板与 chunk 模板
data/         输入 PDF、问题集和样例数据
outputs/      解析产物、索引、图谱、证据链和导出结果
docs/         项目文档与辅助说明
demo/         演示资源
```

## Quick Start

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如需启用前端：

```bash
cd web
npm install
```

### 2. 配置环境变量

可以参考：

```text
configs/doubao_optimized.env.example
```

核心配置项：

```bash
RAG_PDF_PARSER=mineru

RAG_VISUAL_CAPTION_PROVIDER=qwen
RAG_QWEN_VL_MODEL=qwen-vl-plus
RAG_QWEN_API_KEY_ENV=DASHSCOPE_API_KEY

RAG_EMBEDDING_PROVIDER=ark
RAG_EMBEDDING_MODEL=doubao-embedding-vision-250615
ARK_EMBEDDING_DIMENSIONS=1024

RAG_ANSWER_PROVIDER=ark
RAG_ANSWER_MODEL=<your-ark-endpoint-id>

RAG_BACKEND_CANDIDATE_RETRIEVER=fusion
RAG_BACKEND_RERANK_RETRIEVER=fusion
RAG_BACKEND_ENABLE_KG=1
```

### 3. 运行完整离线流水线

```bash
python scripts/06_run_pipeline.py \
  --questions data/questions.csv \
  --candidate-k 50 \
  --rerank-k 10
```

常用跳过参数：

```bash
--skip-parse     # 跳过 PDF 解析
--skip-visual    # 跳过视觉增强
--skip-kg        # 跳过知识图谱构建
```

### 4. 启动后端服务

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

API：

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

前端默认连接：

```text
http://127.0.0.1:8765
```

## Outputs

典型产物：

```text
outputs/parsed/nodes.jsonl          统一证据节点
outputs/parsed/edges.jsonl          文档证据图边
outputs/visual/pages/               页面渲染图
outputs/visual/crops/               局部裁剪图
outputs/embeddings/                 embedding 缓存
outputs/kg/entities.jsonl           知识图谱实体
outputs/kg/relations.jsonl          知识图谱关系
outputs/text_index/nodes.jsonl      文本索引
outputs/visual_index/images.jsonl   视觉索引
outputs/evidence_chains.jsonl       证据链
outputs/evidence_cards.jsonl        前端证据卡片
```

## Engineering Design

项目按“可替换、可观测、可回退”的方式组织：

- **可替换解析器**：MinerU 为默认解析器，保留 native parser 回退能力。
- **可替换模型层**：视觉 caption、embedding、答案生成均通过配置切换供应商。
- **可复用节点 schema**：解析、视觉、图谱、检索和生成共享同一 evidence node。
- **可解释重排**：每条候选结果保留 sim、PPR、bridge、visual、KG 等分项分数。
- **可追溯输出**：答案可回溯到节点、页码、图片、caption 和关系图谱。
- **可控生成**：通过 self-check、候选裁决、截断检测和保守回退降低生成风险。

## Main Scripts

```text
01_parse_pdf.py                         PDF 解析与节点生成
02_build_graph.py                       文档证据图构建
03_retrieve_candidates.py               多路候选召回
04_rerank.py                            G0-G4 分层重排
06_run_pipeline.py                      全链路流水线
09_build_evidence_chains.py             证据链生成
10_build_visual_evidence.py             视觉裁剪与图片理解
11_build_evidence_cards.py              前端证据卡片导出
21_generate_answer.py                    答案生成与后处理
23_build_kg.py                          知识图谱构建
24_ablate_retrieval.py                  检索消融分析
26_judge_submissions.py                 候选答案裁决
29_refine_submission_answers.py         高风险答案精修
embedding_index.py                      向量索引与缓存
rerank_lib.py                           召回、路由、重排核心逻辑
```

## Tech Stack

- Python, FastAPI, NetworkX, scikit-learn
- PyMuPDF, pdfplumber, pypdf, MinerU
- OpenAI-compatible API clients
- Qwen-VL, Doubao Vision, Doubao Embedding Vision
- React, Vite, Tailwind CSS, lucide-react

## Roadmap

- 接入独立向量数据库，如 Milvus、Qdrant、FAISS 或 Chroma
- 将轻量 KG 存储迁移到图数据库，如 Neo4j 或 NebulaGraph
- 增加跨文档多跳推理与答案级 citation
- 增强表格结构理解与数值型问答能力
- 引入在线评测面板，支持召回、重排、答案质量的持续监控
