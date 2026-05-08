# 多模态 RAG 证据链检测系统

本项目实现了一个面向复杂 PDF 文档问答的多模态 RAG 证据链检测系统。系统不仅返回答案相关证据，还会展示证据来自哪一页、哪一个文本块/表格/图像节点，以及这些证据之间如何通过图结构组成一条可解释的证据链。

核心目标是验证一种“查询感知的多模态证据重排序与证据链展示”流程：将 PDF 拆解为多模态节点，构建文档证据图，结合 embedding 检索、图结构信号、引用关系和视觉 grounding，生成最终的 G4 排序结果与横版证据卡片。

## 技术流程

```text
PDF 文档
  |
  v
解析与切分
  -> page 节点
  -> text 节点
  -> table 节点
  -> figure 节点
  -> caption 节点
  |
  v
视觉 grounding
  -> 页面截图
  -> 节点 bbox
  -> 局部证据裁剪图
  -> 可选 VLM caption / qa_evidence
  |
  v
证据图构建
  -> same_page
  -> belongs_to_page
  -> table_caption / figure_caption
  -> text_ref_table / text_ref_figure
  |
  v
候选召回与重排序
  -> G0 原始候选
  -> G1 语义相似度
  -> G2 语义 + PPR
  -> G3 语义 + PPR + Bridge + 引用
  -> G4 G3 + 视觉 grounding
  |
  v
证据链生成
  -> 主证据
  -> 显式引用证据
  -> 图表证据
  -> 图邻居证据
  -> 视觉补充证据
  |
  v
横版证据卡片 + Web 展示
```

## PDF 解析与 Chunk 实现

解析入口位于 `scripts/01_parse_pdf.py`。

系统支持两种来源：

- 直接解析 PDF。
- 接入 RAG-Anything/MinerU 导出的 `content_list` JSON。

直接解析 PDF 时，系统会依次尝试：

1. `pypdf`
2. `PyPDF2`
3. `PyMuPDF`
4. `pdfplumber`
5. `pdftotext`

每一页都会先生成一个 `page` 节点，然后对页面文本进行 chunk。

Chunk 逻辑：

- 优先按空行切分段落。
- 如果段落结构不足，则按行累积。
- 默认 `chunk_size=900`。
- 当累积文本长度达到阈值，或检测到图注/表注模式时，形成一个 chunk。
- 超长 chunk 会继续按句子边界拆分。

节点类型判断：

- 匹配 `Figure/Fig./图` 等图注模式时，生成 `caption` 节点，并额外推断 `figure` 节点。
- 匹配 `Table/表` 等表注模式时，生成 `caption` 节点，并额外推断 `table` 节点。
- 包含多行数字、分隔符、表格提示词时，推断为 `table`。
- 其他内容默认为 `text`。

节点基础字段：

```text
node_id, doc_id, page, node_type, content, source_ref
```

视觉增强后会补充：

```text
page_image_path, crop_image_path, bbox, bbox_source,
visual_summary, visual_caption, visual_title, qa_evidence
```

## 视觉 Grounding 实现

视觉证据构建位于 `scripts/10_build_visual_evidence.py`。

系统使用 `PyMuPDF` 对 PDF 页面进行渲染，生成：

- 整页截图：`outputs/visual/pages/`
- 节点裁剪图：`outputs/visual/crops/`

裁剪定位逻辑：

- 对文本节点，使用页面 text block 与节点文本做相似匹配，找到最佳文本块 bbox。
- 对图注节点，寻找附近 image block，合并图注和图片区域。
- 对表格节点，寻找图注附近的表格样式文本区域。
- 如果是视觉节点但无法精确匹配，则选择页面中较大的 image block 或投影区域。

裁剪图并不是装饰图片，而是证据链里的 grounding 依据。后续 G4 排序和证据卡片都会使用这些裁剪图。

视觉 caption 可选：

- 默认本地流程只生成裁剪图和 `visual_summary`。
- 可接入 Qwen-VL 或豆包 Ark，对裁剪图生成结构化视觉描述。
- 结构化字段包括 `visual_title`、`visual_type`、`key_objects`、`data_or_trends`、`qa_evidence`、`limitations`。

## Embedding 与向量索引

Embedding 实现在 `scripts/embedding_index.py`。

默认模型：

```text
BAAI/bge-m3
```

可通过环境变量修改：

```text
RAG_EMBEDDING_MODEL
RAG_EMBEDDING_DEVICE
RAG_EMBEDDING_BATCH_SIZE
```

节点 embedding 文本格式：

```text
node_type
content
```

也就是说，模型不仅看到 chunk 内容，还会看到该节点是文本、表格、图像还是图注。

当前项目没有接入 Milvus、FAISS、Chroma 这类独立向量数据库，而是实现了一个轻量级本地向量索引：

- 使用 `sentence-transformers` 编码节点。
- embedding 使用 L2-normalized 向量。
- 查询时编码问题，与所有节点 embedding 做矩阵乘法得到相似度。
- embedding 缓存为 `.npz` 文件，存储在 `outputs/embeddings/`。
- 缓存 key 由模型名、节点内容 hash、node_id 序列共同决定。

这种实现更适合课程项目和小规模实验：可复现、依赖少、容易解释。后续如果数据规模变大，可以将 `EmbeddingIndex` 替换为 FAISS、Milvus 或 Qdrant。

## 候选召回

候选召回位于 `scripts/03_retrieve_candidates.py` 和 `rerank_lib.retrieve_candidates`。

支持三种召回方式：

```text
lexical
embedding
hybrid
```

Lexical 召回：

- 使用 `TfidfVectorizer`
- 字符级 n-gram
- 默认 `ngram_range=(2, 4)`
- 适合中英文混合、术语匹配和表格编号匹配。

Embedding 召回：

- 使用 `BAAI/bge-m3` 等语义模型。
- 适合语义相近但字面不完全一致的问题。

Hybrid 召回：

```text
hybrid_score = alpha * embedding_score + (1 - alpha) * lexical_score
```

默认 `hybrid_alpha=0.7`。

如果问题指定了 `doc_id`，召回池会优先限制在对应文档内，避免跨文档误召回。

## 证据图设计

证据图构建位于 `scripts/02_build_graph.py`。

节点类型包括：

```text
page, text, table, figure, caption, equation
```

边类型包括：

```text
belongs_to_page
same_page
table_caption
figure_caption
text_ref_table
text_ref_figure
related
```

边的含义：

- `belongs_to_page`：节点属于某一页。
- `same_page`：同页内不同证据节点之间的弱连接。
- `table_caption`：表格与表注之间的强连接。
- `figure_caption`：图片与图注之间的强连接。
- `text_ref_table`：正文显式引用某个表格。
- `text_ref_figure`：正文显式引用某个图片。

证据图使用 `networkx` 存储和计算。它在后续重排序中提供 PPR、Bridge 和邻居补证逻辑。

## 重排序方法 G0-G4

核心逻辑位于 `scripts/rerank_lib.py`。

### G0：原始候选顺序

G0 保留初始召回结果，用作 baseline。

```text
G0 = candidate retrieval order
```

### G1：语义相似度重排

G1 只使用查询和节点之间的语义相似度。

```text
G1 = Sim(q, node)
```

Sim 可以来自 lexical、embedding 或 hybrid。

### G2：语义相似度 + 查询感知 PPR

G2 引入证据图上的 Personalized PageRank。

```text
G2 = alpha * Sim + beta * PPR
```

PPR 的 personalization 由候选节点的相似度分布初始化，因此它是查询感知的图传播，而不是静态 PageRank。

系统会根据问题类型动态调整 PPR 信号权重。例如纯文本事实题会弱化图传播，表格/图像问题会增强结构传播。

### G3：语义 + PPR + Bridge + 引用信号

G3 进一步加入跨模态桥接分数和显式引用分数。

```text
G3 = λs * Sim + λp * PPR + λb * Bridge + λr * Reference
```

Bridge 分数衡量一个候选节点是否能连接到问题需要的其他模态。例如：

- 文本节点旁边有相关表格。
- 图注连接到图片节点。
- 正文引用了某个 figure/table。
- 同页存在可补充的视觉证据。

Reference 分数用于处理问题里显式出现的 `Table 1`、`Figure 2`、`图 3` 等引用。

### G4：视觉 Grounding 增强排序

G4 在 G3 基础上加入视觉信号。

```text
G4 = G3 + visual_weight * visual_score * max(0.05, 1 - G3)
```

视觉信号来自：

- 节点是否有裁剪图。
- 节点是否有视觉 caption 或 `qa_evidence`。
- 节点类型是否匹配问题意图。
- 视觉描述与问题的 lexical 相似度。
- 视觉节点是否与主证据存在图关系。

`visual_weight` 会根据问题类型动态变化：

- 文本事实题：较小。
- 表格/图像题：较大。
- 跨模态/定位题：中等偏高。

这样可以避免视觉信号过度干扰纯文本问题，同时在多模态问题中突出视觉证据。

## 证据链生成

证据链构建位于 `scripts/09_build_evidence_chains.py`。

系统不是简单展示 Top-K，而是把证据组织成角色链路。

证据角色包括：

```text
main_evidence
explicit_reference
table_or_figure
caption
context_text
graph_neighbor
visual_companion
```

构建逻辑：

1. 选取 G4 Top-1 作为主证据。
2. 如果问题包含图表编号，优先补充显式引用证据。
3. 如果问题需要视觉信息，优先补充 table/figure/caption 节点。
4. 沿证据图查找邻居节点，补充上下文或跨模态节点。
5. 如果视觉问题仍缺少视觉节点，强制寻找同页或图邻居视觉补充节点。
6. 最多保留若干步，默认 `max_steps=5`。

输出：

```text
outputs/evidence_chains/chains.jsonl
outputs/evidence_chains/chain_steps.csv
outputs/evidence_chains/evidence_chains.md
```

## 证据卡片生成

证据卡片生成位于 `scripts/11_build_evidence_cards.py`。

当前卡片为横版：

```text
1920 x 1080
```

卡片内容包括：

- 问题
- 答案
- 来源页
- 证据链步骤
- 节点角色
- 节点页码和类型
- 局部视觉裁剪图
- 简短证据摘要

卡片使用 `Pillow` 生成，不依赖外部生图服务。因此它可复现、速度快，也适合本地演示。

质量检查脚本：

```text
scripts/12_check_evidence_cards.py
```

检查内容包括：

- 卡片是否存在。
- 尺寸是否为 1920 x 1080。
- 是否非空白。
- 问题和答案是否存在。
- 证据链步骤是否足够。
- 是否有可访问的裁剪图。
- 多模态问题是否包含视觉节点。

## 后端实现简述

后端位于 `backend/app.py`，使用 FastAPI。

主要接口：

```text
GET  /api/health
POST /api/analyze
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/files/{path}
```

上传 PDF 后，后端会创建一个异步 job，结果写入：

```text
outputs/upload_jobs/{job_id}/
```

每个 job 包含：

```text
pdfs/
nodes.jsonl
edges.jsonl
candidates.csv
reranked.csv
chain_steps.csv
evidence_card.png
result.json
status.json
```

前端通过轮询 `GET /api/jobs/{job_id}` 获取进度和最终结果。

## 前端实现简述

前端位于 `web/`，使用 React + Vite。

页面流程：

1. 选择 PDF：上传本地 PDF 或选择系统 PDF。
2. 选择问题：上传 PDF 显示自定义问题；系统 PDF 显示预设问题。
3. 展示证据：上方显示横版证据卡片，下方按相关性展示证据节点。

系统 PDF 的前端数据由脚本导出：

```text
scripts/13_export_frontend_data.py
```

导出结果：

```text
web/public/app-data.json
web/public/outputs/
```

上传 PDF 时，前端直接调用 FastAPI 后端。

## 运行方式

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

安装前端依赖：

```bash
cd web
npm install
```

启动后端：

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

导出系统 PDF 的前端数据：

```bash
python scripts/13_export_frontend_data.py
```

构建并启动前端：

```bash
cd web
npm run build
cd dist
python -m http.server 5174 --bind 127.0.0.1
```

访问：

```text
http://127.0.0.1:5174/
```

## 离线实验命令

完整流水线：

```bash
python scripts/06_run_pipeline.py \
  --questions data/questions.csv \
  --candidate-retriever lexical \
  --rerank-retriever hybrid \
  --embedding-device auto
```

单步运行：

```bash
python scripts/01_parse_pdf.py
python scripts/10_build_visual_evidence.py
python scripts/02_build_graph.py
python scripts/03_retrieve_candidates.py
python scripts/04_rerank.py
python scripts/09_build_evidence_chains.py
python scripts/11_build_evidence_cards.py
python scripts/12_check_evidence_cards.py
python scripts/13_export_frontend_data.py
```

## 当前技术边界

- 当前向量索引是本地 `.npz` 缓存和内存矩阵检索，不是独立向量数据库。
- PDF 表格结构主要依赖文本抽取和规则推断，复杂表格仍可能需要 RAG-Anything/MinerU 或人工补充节点。
- 默认上传流程不会调用外部 VLM API，因此视觉理解主要体现为裁剪图 grounding 和可选 caption 字段。
- 大 PDF 首次运行会较慢，主要耗时在页面渲染、embedding 编码和证据卡片生成。
