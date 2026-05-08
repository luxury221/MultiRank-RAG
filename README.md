# 多模态 RAG 证据链检测系统

一个面向复杂 PDF 文档问答的多模态 RAG 原型系统。项目将 PDF 解析为文本、表格、图片、图注和页面节点，结合 embedding 检索、图结构重排序、视觉 grounding 和证据链生成，最终在 Web 页面中展示横版证据卡片和按相关性排序的证据节点。

项目既支持系统预置 PDF 的离线评测，也支持用户在页面中上传 PDF 并输入自定义问题，由后端动态生成证据链和证据卡片。

## 核心功能

- PDF 多模态节点构建：文本块、表格、图片、图注、页面节点。
- 视觉证据定位：为节点生成页面截图和局部裁剪图。
- 混合检索：支持 lexical、embedding、hybrid 三种召回和重排序方式。
- 图结构增强：基于同页关系、页面归属、正文引用表格/图片、图注关系构建 evidence graph。
- G0-G4 对比：从原始召回、语义重排、PPR、Bridge 到视觉增强 G4 排序。
- 证据链生成：自动组织主证据、图邻居、显式引用、图表证据、视觉补充等角色。
- 横版证据卡片：生成 1920 x 1080 的证据链卡片，适合展示和汇报。
- Web UI：上传 PDF 或选择系统 PDF，选择问题后查看证据卡片与证据相关性排序。
- FastAPI 后端：支持上传 PDF 后在线解析、检索、重排、生成证据链和证据卡片。

## 系统架构

```text
PDF
  |
  v
PDF parsing
  -> text/table/figure/caption/page nodes
  -> page images and evidence crops
  |
  v
Evidence graph
  -> same_page
  -> belongs_to_page
  -> text_ref_table / text_ref_figure
  -> table_caption / figure_caption
  |
  v
Retrieval and reranking
  -> G0 candidate retrieval
  -> G1 semantic ranking
  -> G2 semantic + PPR
  -> G3 semantic + PPR + bridge/reference
  -> G4 visual grounded reranking
  |
  v
Evidence chain and evidence card
  |
  v
React frontend + FastAPI backend
```

## 目录结构

```text
backend/
  app.py                         # FastAPI 后端，负责上传 PDF 动态分析

web/
  src/                           # React 前端源码
  public/                        # 前端静态数据和图片资源，构建前由脚本生成
  package.json
  vite.config.ts

scripts/
  01_parse_pdf.py                # PDF/content_list 解析为节点
  02_build_graph.py              # 构建证据图
  03_retrieve_candidates.py      # 候选证据召回
  04_rerank.py                   # G0-G4 重排序
  05_evaluate.py                 # 指标评测
  06_run_pipeline.py             # 一键运行离线流水线
  09_build_evidence_chains.py    # 构建证据链
  10_build_visual_evidence.py    # 页面截图、裁剪图、可选 VLM caption
  11_build_evidence_cards.py     # 生成横版证据卡片
  12_check_evidence_cards.py     # 检查证据卡片质量
  13_export_frontend_data.py     # 导出前端 app-data.json 和图片资源
  embedding_index.py             # embedding 索引与缓存
  rerank_lib.py                  # 检索、重排序、图信号核心逻辑

data/
  questions.csv                  # 系统预置问题
  manual_nodes.csv               # 可选人工补充节点
  pdfs/                          # PDF 原文，默认不提交到 GitHub

outputs/                         # 运行结果，默认不提交到 GitHub
```

## 环境准备

建议使用 Python 3.10 或以上版本。GPU 环境可加速 embedding，但不是强制要求。

```bash
pip install -r requirements.txt
```

前端依赖：

```bash
cd web
npm install
```

本项目默认使用 `BAAI/bge-m3` 作为 embedding 模型。可通过环境变量修改：

```bash
export RAG_EMBEDDING_MODEL=BAAI/bge-m3
export RAG_EMBEDDING_DEVICE=auto
```

Windows PowerShell 示例：

```powershell
$env:RAG_EMBEDDING_MODEL="BAAI/bge-m3"
$env:RAG_EMBEDDING_DEVICE="auto"
```

## 快速运行 Web 系统

### 1. 启动后端

通用命令：

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

如果使用本项目调试时的 GPU conda 环境：

```powershell
D:\conda_envs\rag-gpu\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

后端健康检查：

```text
http://127.0.0.1:8765/api/health
```

### 2. 生成前端数据

如果要展示系统预置 PDF 和问题的已有结果，需要先导出前端数据：

```bash
python scripts/13_export_frontend_data.py
```

### 3. 构建并启动前端

```bash
cd web
npm run build
cd dist
python -m http.server 5174 --bind 127.0.0.1
```

访问页面：

```text
http://127.0.0.1:5174/
```

## 页面使用方式

首页分为两种入口：

1. 上传本地 PDF  
   用户选择 PDF 后输入自定义问题，前端调用后端 API。后端会动态完成 PDF 解析、证据检索、G4 重排序、证据链构建和横版证据卡片生成。

2. 选择系统 PDF  
   用户选择项目预置文档后，页面显示该文档对应的预设问题。选择问题后展示已经离线生成的证据卡片和证据排序结果。

证据展示页为上下结构：

- 上方：横版证据卡片，展示问题、答案、证据链和视觉裁剪。
- 下方：证据相关性排序，从高到低展示证据节点、页码、节点类型、裁剪图和相关性分数。

## 上传 PDF API

后端主要接口：

```text
GET  /api/health
POST /api/analyze
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/files/{path}
```

上传 PDF 示例：

```bash
curl -X POST \
  -F "pdf=@example.pdf;type=application/pdf" \
  -F "question=这篇文档的主要结论是什么？" \
  http://127.0.0.1:8765/api/analyze
```

返回 `job_id` 后，轮询：

```bash
curl http://127.0.0.1:8765/api/jobs/<job_id>
```

任务完成后，返回结果中包含：

- `question`：问题、答案、证据卡片 URL、质量状态。
- `steps`：证据链步骤。
- `rankings`：G0-G4 排序结果。

## 离线实验流水线

将 PDF 放入 `data/pdfs/`，并准备 `data/questions.csv` 后，可以运行完整离线流程：

```bash
python scripts/06_run_pipeline.py \
  --questions data/questions.csv \
  --candidate-retriever lexical \
  --rerank-retriever hybrid \
  --embedding-device auto
```

常见输出：

```text
outputs/parsed/nodes.jsonl
outputs/parsed/edges.jsonl
outputs/rankings/candidates.csv
outputs/rankings/reranked.csv
outputs/metrics/summary_metrics.csv
outputs/evidence_chains/chain_steps.csv
outputs/evidence_cards/*_evidence_card.png
```

生成或更新离线结果后，重新导出前端数据：

```bash
python scripts/13_export_frontend_data.py
cd web
npm run build
```

## 数据格式

`data/questions.csv` 字段：

```text
question_id,doc_id,question,answer,question_type,gold_node_ids,gold_pages,gold_modalities,evidence_note
```

`outputs/parsed/nodes.jsonl` 主要字段：

```text
node_id, doc_id, page, node_type, content, source_ref,
page_image_path, crop_image_path, bbox, visual_summary, visual_caption
```

`outputs/evidence_chains/chain_steps.csv` 主要字段：

```text
question_id, chain_step, role, node_id, node_type, page,
relation, score, visual_score, crop_image_path, content_preview, reason
```

## 可选视觉模型接入

`scripts/10_build_visual_evidence.py` 支持本地 caption、Qwen-VL 和豆包 Ark 兼容接口。使用云端视觉模型前需要配置 API key。

Qwen 示例：

```bash
export DASHSCOPE_API_KEY=your_key
python scripts/10_build_visual_evidence.py \
  --caption-provider qwen \
  --qwen-model qwen-vl-plus \
  --max-captions 50
```

豆包 Ark 示例：

```bash
export ARK_API_KEY=your_key
export ARK_MODEL=your_model_endpoint
python scripts/10_build_visual_evidence.py \
  --caption-provider doubao \
  --max-captions 50
```

## GitHub 提交说明

仓库已经包含 `.gitignore`，默认不会提交以下内容：

- PDF 原文：`data/pdfs/*.pdf`
- 运行结果：`outputs/`
- 前端构建产物：`web/dist/`
- 前端静态导出资源：`web/public/outputs/`
- 前端依赖：`web/node_modules/`
- API key、`.env`、缓存和日志

如果希望公开演示数据，可以额外准备脱敏的小型 PDF 或示例 JSON，并单独调整 `.gitignore`。

## 当前限制

- 上传 PDF 的在线分析会随 PDF 页数、embedding 模型和设备性能变慢。
- 默认 PDF 解析主要依赖文本抽取和 PyMuPDF 页面裁剪，复杂表格结构仍可能需要人工补充或更强的文档解析器。
- 云端视觉 caption 需要自行配置 API key，默认上传流程不会调用外部视觉 API。
- `outputs/` 和 `data/pdfs/` 默认不进入 GitHub，因此克隆仓库后需要重新准备 PDF 并运行流水线。

## 项目定位

本项目是一个多模态 RAG 证据链展示与评测原型，重点不只是回答问题，而是展示答案背后的证据来源、页面定位、图文关系和排序依据。它适合用于课程设计、开题报告原型、论文实验演示，以及复杂文档问答系统的前期验证。
