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
  -> title / equation 节点
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
  -> section_title / same_section
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

直接解析 PDF 时，系统采用 layout-aware 优先策略：

1. `PyMuPDF`：读取页面 text block、image block、字体大小、block bbox 和页面尺寸。
2. `pdfplumber`：可选抽取结构化表格，保留行列文本和表格 bbox。
3. `pypdf` / `PyPDF2` / `pdfplumber.extract_text` / `pdftotext`：当 layout 解析不可用时作为文本回退链路。

每一页都会先生成一个 `page` 节点，然后对页面文本进行 chunk。

Layout-aware 解析会额外产生：

- 文本块坐标：`bbox`、`bbox_source=pymupdf_text_block`。
- 图片区域节点：由 PDF image block 生成 `figure` 节点。
- 结构化表格节点：由 `pdfplumber` 生成 `table` 节点，并记录 `table_rows`、`table_cols`。
- 版面属性：`layout_parser`、`layout_block_id`、`layout_role`、`font_size`、`line_count`、`page_width`、`page_height`。
- 多栏阅读顺序：自动判断单栏/双栏页面，记录 `layout_column` 和 `reading_order`，减少双栏论文串读。
- 噪声过滤：过滤重复页眉页脚、页码以及窄而高的侧边栏噪声，例如 arXiv 侧栏。
- 坐标优先视觉裁剪：后续 visual grounding 会优先使用解析阶段得到的 bbox，匹配失败时再回退到文本相似匹配。

Chunk 逻辑已经加入论文领域模板，借鉴 RAGFlow 的“按文档类型选择切片策略”思想，但实现保持轻量、可解释。解析器支持：

```text
auto, general, ai, math, finance, medical
```

前端上传 PDF 时可以由用户先选择论文领域；后端仍会运行 `auto` 识别作为保险，并在节点中记录自动识别结果、置信度和候选领域分布。如果用户选择的模板与自动识别不一致，系统保留用户选择，同时把自动判断写入元数据，便于后续检查。

`auto` 会根据文件名和前几页内容自动判断论文领域；如果没有足够强的领域信号，会回落到 `general`，避免把普通工程、化学、材料类论文误切成 AI 或数学模板。

模板配置位于：

```text
configs/chunk_templates/paper_templates.json
```

代码中保留默认模板作为 fallback；配置文件可以继续扩展关键词、章节别名、保护结构块和推荐 chunk 长度。

模板化 chunk 的主要差异：

- `general`：按通用论文结构切分，适合工程、环境、化学、材料等未被强匹配的领域。
- `ai`：强化 `model`、`architecture`、`algorithm`、`dataset`、`benchmark`、`ablation`、`evaluation` 等章节和算法块。
- `math`：强化 `theorem`、`lemma`、`definition`、`proof`、`equation` 等理论结构，公式密集块会标记为 `equation`。
- `finance`：强化 `data`、`variables`、`empirical strategy`、`regression results`、`risk analysis` 等实证金融结构。
- `medical`：强化 `study design`、`participants`、`intervention`、`outcomes`、`statistical analysis`、`adverse events` 等医学论文结构。

基础 chunk 逻辑：

- 优先按空行切分段落。
- 如果段落结构不足，则按行累积。
- 默认 `chunk_size=900`；领域模板会在未显式指定 chunk size 时使用自己的推荐长度。
- 当累积文本长度达到阈值，或检测到图注/表注模式时，形成一个 chunk。
- 超长 chunk 会继续按句子边界拆分。
- 识别章节标题时生成 `title` 节点，并将后续 chunk 标记到同一 `section`。
- 算法、定理、证明、定义、回归、公式等结构会作为 protected block 保留，避免被普通段落切碎。
- 每个章节标题会作为父级 chunk，普通段落、图表、公式等作为子级 chunk，通过 `parent_chunk_id` 和 `section_id` 形成层级结构。
- 每个 chunk 会记录前后文锚点、显式图表引用和同页图注锚点，检索时可以利用上下文，展示时仍保留原始证据块。

节点类型判断：

- 匹配 `Figure/Fig./图` 等图注模式时，生成 `caption` 节点，并额外推断 `figure` 节点。
- 匹配 `Table/表` 等表注模式时，生成 `caption` 节点，并额外推断 `table` 节点。
- 包含多行数字、分隔符、表格提示词时，推断为 `table`。
- 数学模板下匹配公式、定理证明上下文时，推断为 `equation` 或理论文本节点。
- 其他内容默认为 `text`。

节点基础字段：

```text
node_id, doc_id, page, node_type, content, source_ref,
paper_domain, chunk_template, requested_chunk_template,
auto_chunk_template, auto_domain_confidence, domain_candidates,
section, section_id, parent_chunk_id, chunk_level,
chunk_strategy, structure_type,
previous_node_id, next_node_id,
previous_chunk_preview, next_chunk_preview,
nearby_caption_refs, explicit_refs,
bbox, bbox_source, layout_parser, layout_block_id,
layout_role, layout_column, reading_order,
page_width, page_height, layout_column_count,
filtered_header_footer_blocks
```

Chunk 质量报告位于 `scripts/14_chunk_quality_report.py`。它会统计每篇论文的领域模板、章节数、节点类型分布、平均 chunk 长度、父子 chunk 覆盖率、前后文覆盖率、bbox 覆盖率、layout 节点率、结构化表格数量、图表配对率和孤立 chunk 比例。输出默认写入：

```text
outputs/metrics/chunk_quality.csv
outputs/metrics/chunk_quality.json
```

Layout bbox 可视化检查位于 `scripts/15_visualize_layout_bboxes.py`。它会把节点 bbox 画到 PDF 页面截图上，便于检查标题、正文、表格、图像和图注是否被正确定位。默认输出到：

```text
outputs/layout_debug/
outputs/metrics/layout_bbox_debug.csv
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
type: node_type
paper_domain: ai/math/finance/medical/general
section: abstract/method/results/...
structure_type: algorithm/theorem/regression/outcome/...
chunk_strategy: section_aware_paragraph/protected_paper_block/...
explicit_refs: figure:1;table:2
previous_context: ...
next_context: ...
content
```

也就是说，模型不仅看到 chunk 内容，还会看到该节点是文本、表格、图像还是图注，以及它所属的论文领域、章节、结构块类型、显式引用和轻量前后文。这样可以增强“问方法找方法章节”“问实验结果找 results/evaluation”“问医学结局找 outcomes”等论文问答场景下的语义召回。

当前项目没有接入 Milvus、FAISS、Chroma 这类独立向量数据库，而是实现了一个轻量级本地向量索引：

- 使用 `sentence-transformers` 编码节点。
- embedding 使用 L2-normalized 向量。
- 查询时编码问题，与所有节点 embedding 做矩阵乘法得到相似度。
- embedding 缓存为 `.npz` 文件，存储在 `outputs/embeddings/`。
- 缓存 key 由模型名、节点内容 hash、node_id 序列共同决定。

这种实现更适合课程项目和小规模实验：可复现、依赖少、容易解释。后续如果数据规模变大，可以将 `EmbeddingIndex` 替换为 FAISS、Milvus 或 Qdrant。

## 候选召回

候选召回位于 `scripts/03_retrieve_candidates.py` 和 `rerank_lib.retrieve_candidates`。

支持四种召回方式：

```text
fusion
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

Fusion 召回：

Fusion 是当前默认候选召回方式，借鉴 RAGFlow 的多路召回与融合思想，但实现保持在本项目可解释范围内。它会把多条证据来源用 Reciprocal Rank Fusion 融合：

- `lexical`：字符级 TF-IDF，保留术语、表格编号、中文短语匹配能力。
- `embedding`：语义向量召回，适合字面不一致但语义接近的问题；仅在已构建 embedding index 时启用。
- `section`：基于 `section`、`structure_type`、`chunk_strategy` 的论文章节/结构块召回。
- `reference`：基于 `Figure 1`、`Table 2`、`图 3` 等显式引用召回。
- `visual`：基于 `table/figure/caption`、视觉 caption、bbox 和 layout role 的多模态证据召回。
- `layout`：面向定位类问题，提升带 bbox、裁剪图、页面图的节点。

融合时会根据问题意图动态调权。例如表格/图像问题会提高 `visual` 和 `reference` 路权重，定位问题会提高 `layout` 路权重，纯文本事实题会降低视觉干扰。

如果问题指定了 `doc_id`，召回池会优先限制在对应文档内，避免跨文档误召回。

## 证据图设计

证据图构建位于 `scripts/02_build_graph.py`。

节点类型包括：

```text
page, title, text, table, figure, caption, equation
```

边类型包括：

```text
belongs_to_page
same_page
section_title
same_section
parent_section
chunk_sequence
table_caption
figure_caption
text_ref_table
text_ref_figure
related
```

边的含义：

- `belongs_to_page`：节点属于某一页。
- `same_page`：同页内不同证据节点之间的弱连接。
- `section_title`：章节标题与该章节内容之间的结构连接。
- `same_section`：同一论文章节内相邻证据节点之间的连接。
- `parent_section`：父级章节 chunk 与子级证据 chunk 之间的层级连接。
- `chunk_sequence`：同一文档中相邻 chunk 之间的前后文连接。
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
- 候选证据与同一章节标题、方法段、结果段存在结构连接。
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
  --candidate-retriever fusion \
  --rerank-retriever fusion \
  --embedding-device auto
```

单步运行：

```bash
python scripts/01_parse_pdf.py
python scripts/14_chunk_quality_report.py --nodes outputs/parsed/nodes.jsonl
python scripts/15_visualize_layout_bboxes.py --nodes outputs/parsed/nodes.jsonl --max-pages-per-doc 4
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
