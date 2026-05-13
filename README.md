# 多模态 RAG 证据链系统

这是一个面向复杂 PDF 和图文混合资料的多模态 RAG 项目。系统会把 PDF 解析成文本、表格、图片、图注、标题等证据节点，结合视觉理解、图文 embedding、BM25、知识图谱和图结构重排序，返回可追溯的证据链和最终回答。

项目重点不是只生成一段答案，而是完整保留“答案从哪里来、证据之间如何关联、图片是否真的支持回答”的链路。

## 核心能力

- **PDF 解析**：默认使用 MinerU 解析 PDF，保留文本、表格、图片、公式、bbox、图注和页面结构。
- **视觉理解**：对图片和表格裁剪图调用 Qwen-VL 或豆包视觉模型，生成 `visual_caption`、`key_objects`、`ocr_text`、`qa_evidence` 等结构化字段。
- **图文 embedding**：使用 `doubao-embedding-vision-250615` 时，视觉节点会把“裁剪图 + 文本上下文 + 视觉识别结果”一起编码。
- **多路召回**：默认 `fusion`，融合 lexical、BM25、embedding、商品/实体匹配、KG、section、reference、visual、layout 等路线。
- **GraphRAG-lite**：从节点中抽取产品、部件、动作、故障、政策、图片实体和关系，构建轻量知识图谱。
- **G4 重排序**：融合语义相似度、PPR、跨模态 bridge、引用、视觉 grounding、证据链连贯性、领域意图、商品匹配和 KG 信号。
- **证据链与答案生成**：先组织证据链，再由 LLM 基于证据生成答案；图片类问题会做二阶段视觉重排，答案生成后可进行 self-check。
- **前端展示**：FastAPI 后端和 Web 前端支持上传 PDF、查看检索结果、证据链、裁剪图和证据卡片。

## 项目结构

```text
backend/   API 服务与异步任务编排
web/       前端展示与可视化
scripts/   解析、视觉增强、召回、重排、证据链、导出
configs/   环境变量模板
data/      输入样例与问题集
outputs/   中间产物与结果文件
```

## 系统流程

```text
PDF / 图文资料
  |
  v
MinerU 解析
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
  -> bbox 定位
  -> 局部裁剪图
  -> Qwen-VL / 豆包视觉 caption
  |
  v
节点增强
  -> visual_title
  -> key_objects
  -> ocr_text
  -> data_or_trends
  -> qa_evidence
  -> visual_summary
  |
  v
索引构建
  -> embedding index
  -> text index
  -> visual index
  -> KG entities / relations
  |
  v
问题路由
  -> policy / manual / troubleshooting / visual / text_fact
  |
  v
多路召回 fusion
  -> lexical
  -> BM25
  -> embedding-vision
  -> product/entity
  -> KG
  -> section
  -> reference
  -> visual
  -> layout
  |
  v
G0-G4 重排序
  -> G0 原始召回
  -> G1 相似度
  -> G2 相似度 + PPR
  -> G3 相似度 + PPR + Bridge + 引用
  -> G4 G3 + 视觉 + 证据链 + 领域 + 商品 + KG
  |
  v
证据链生成
  -> 主证据
  -> 上下文证据
  -> 图表证据
  -> 图注证据
  -> KG/邻居补充证据
  |
  v
答案生成
  -> 证据格式化
  -> 图片二阶段重排
  -> LLM 生成
  -> self-check
  |
  v
证据卡片 / Web 展示 / CSV 导出
```

## 数据解析流程

解析入口是 `scripts/01_parse_pdf.py`。

默认流程：

1. 读取 `data/pdfs/` 中的 PDF。
2. 调用 MinerU 解析页面结构、文本、图片、表格、公式和 bbox。
3. 将 MinerU 输出转换为统一 evidence node。
4. 按 chunk 模板切分长文本，并保留章节、前后文和显式图表引用。
5. 写入 `outputs/parsed/nodes.jsonl`。

统一节点类型：

```text
page      页面节点
title     标题/章节节点
text      正文 chunk
table     表格节点
figure    图片/图示节点
caption   图注/表注节点
equation  公式节点
```

常见字段：

```text
node_id, doc_id, page, node_type, content, source_ref,
section, section_id, parent_chunk_id,
previous_node_id, next_node_id,
previous_chunk_preview, next_chunk_preview,
explicit_refs, nearby_caption_refs,
bbox, bbox_source, page_width, page_height,
crop_image_path, page_image_path,
visual_caption, key_objects, ocr_text, qa_evidence, visual_summary
```

## 图片解析与视觉理解

视觉证据构建入口是 `scripts/10_build_visual_evidence.py`。

系统会先渲染 PDF 页面，再为 text/table/figure/caption 节点生成裁剪图：

```text
outputs/visual/pages/
outputs/visual/crops/
```

裁剪定位优先使用 MinerU 或 layout parser 给出的 bbox。如果没有可靠 bbox，则回退到文本块匹配、图注邻近图片、表格邻近文本区域或页面投影区域。

图片识别分成两层：

- **Qwen-VL / 豆包视觉模型**：负责理解图片内容，输出结构化文本证据。
- **doubao-embedding-vision**：负责把图片和文本一起编码成向量，用于召回和重排序。

推荐顺序是：

```text
裁剪图 -> Qwen-VL 生成视觉证据 -> doubao-embedding-vision 编码图文联合表示
```

这样图片节点既有可读的结构化语义，又保留真实图像信息参与向量检索。

## Embedding 与索引

Embedding 实现在 `scripts/embedding_index.py`。

模型读取顺序：

```text
RAG_EMBEDDING_MODEL
ARK_EMBEDDING_MODEL
BAAI/bge-m3
```

推荐配置：

```text
RAG_EMBEDDING_PROVIDER=ark
RAG_EMBEDDING_MODEL=doubao-embedding-vision-250615
ARK_EMBEDDING_DIMENSIONS=1024
```

普通文本节点使用文本 embedding。视觉节点如果存在 `crop_image_path`，Ark 模式下会调用多模态 embedding 接口，将图片和节点文本一起编码。

缓存输出：

```text
outputs/embeddings/*.npz
outputs/embeddings/*_items.jsonl
```

当前项目使用本地 `.npz` 向量矩阵和内存检索，没有强依赖 Milvus、Qdrant、Chroma 或 FAISS。后续数据规模变大时，可以替换 `EmbeddingIndex` 为独立向量数据库。

## GraphRAG-lite 知识图谱

知识图谱构建入口是 `scripts/23_build_kg.py`。

输出目录：

```text
outputs/kg/entities.jsonl
outputs/kg/relations.jsonl
outputs/kg/product_profiles.jsonl
outputs/text_index/nodes.jsonl
outputs/visual_index/images.jsonl
```

实体类型：

```text
product  产品/文档对象
part     部件
action   操作动作
fault    故障现象
policy   规则/政策/服务意图
image    图片
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

KG 的作用不是替代向量检索，而是作为额外结构信号：

- 政策类问题优先定位对应规则节点。
- 部件/操作类问题利用“产品-部件-动作”关系增强召回。
- 故障类问题利用“故障-处理动作-相关部件”关系增强排序。
- 图片类问题利用“图片-部件/动作”关系辅助选择视觉证据。

为了避免噪声，泛化动作词如“使用/use”不会单独驱动 KG；必须和具体产品、部件、故障或政策共同出现。

## 候选召回

候选召回入口是 `scripts/03_retrieve_candidates.py`，核心逻辑在 `rerank_lib.retrieve_candidates`。

支持的 retriever：

```text
fusion
lexical
bm25
embedding
hybrid
kg
```

默认使用 `fusion`。Fusion 会先做问题路由，再动态融合多条召回路线。

问题路由类型：

```text
policy
manual
manual_visual
troubleshooting
visual
text_fact
general
```

召回路线：

- `lexical`：字符级 TF-IDF，适合中文短语、术语和编号匹配。
- `bm25`：适合短关键词、型号、部件、政策词和动作词。
- `embedding`：语义向量召回，适合字面不同但语义接近的问题。
- `hybrid`：embedding 与 lexical 加权。
- `product`：商品名、文档对象和别名匹配。
- `kg`：知识图谱实体与关系召回。
- `section`：章节、结构块、chunk 策略召回。
- `reference`：显式图表编号召回。
- `visual`：视觉 caption、OCR、bbox、layout role 召回。
- `layout`：位置类问题和有 bbox/crop 的节点召回。

不同问题会使用不同权重。例如：

- 规则/政策问题提高 BM25、section、KG-policy，降低视觉干扰。
- 手册操作问题提高 product、section、visual、embedding。
- 故障排查问题提高 product、KG-fault/action 和上下文。
- 图片/表格/位置问题提高 visual、layout、reference。
- 纯文本事实题降低 visual 和 layout。

## 证据图

证据图构建入口是 `scripts/02_build_graph.py`。

节点来自解析阶段的 evidence nodes。边类型包括：

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

证据图用于：

- PPR 图传播。
- 跨模态 bridge。
- 图表和图注补证。
- 同页/同章节上下文补证。
- 证据链组织。

## G0-G4 重排序

重排序核心位于 `scripts/rerank_lib.py`。

```text
G0 = 原始召回顺序
G1 = Sim(q, node)
G2 = alpha * Sim + beta * PPR
G3 = λs * Sim + λp * PPR + λb * Bridge + λr * Reference
G4 = G3 + Visual + Chain + Domain + Product + KG
```

G4 信号说明：

- `visual_score`：图片、表格、图注、caption、OCR、裁剪图与问题的匹配程度。
- `chain_score`：候选证据是否能通过图关系连接到补充证据。
- `domain_score`：面向业务场景的问题意图和节点结构匹配。
- `product_score`：产品名、别名、文档对象匹配。
- `kg_score`：实体和关系图谱匹配。

G4 的目标是让最终证据既语义相关，又结构合理，还能解释为什么选中这段文本或这张图片。

## 证据链生成

证据链构建入口是 `scripts/09_build_evidence_chains.py`。

系统不是简单展示 Top-K，而是把证据组织成角色链：

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

1. 选择 G4 Top-1 作为主证据。
2. 如果问题包含图表编号，补充显式引用证据。
3. 如果问题需要视觉信息，补充 table/figure/caption 节点。
4. 沿证据图查找相邻上下文和跨模态节点。
5. 如果视觉问题缺少视觉节点，尝试查找同页或图邻居视觉证据。
6. 默认最多保留 `max_steps=5` 步。

输出：

```text
outputs/evidence_chains/chains.jsonl
outputs/evidence_chains/chain_steps.csv
outputs/evidence_chains/evidence_chains.md
```

## 答案生成与 self-check

答案生成使用证据链而不是只看 Top-1。文本模型读取顺序：

```text
RAG_ANSWER_MODEL
ARK_TEXT_MODEL_PRO
ARK_TEXT_MODEL
ARK_MODEL
```

答案生成流程：

```text
G4 rows
  -> 证据筛选
  -> 证据链格式化
  -> 图片二阶段重排
  -> LLM 生成答案
  -> self-check
  -> 清理非法图片后缀和过长文本
```

图片二阶段重排会重新检查候选图片是否真的回答问题，只保留最有用的 1-3 张图。规则类、政策类、纯文本类问题不会强行追加图片。

self-check 会检查：

- 是否回答了所有子问题。
- 是否编造了证据里没有的价格、时限或承诺。
- 是否误加了 `<PIC>`。
- 是否缺少必要的处理步骤、凭证要求或条件说明。
- 答案是否过长、过泛或不够直接。

## 后端与前端

后端位于 `backend/app.py`，使用 FastAPI。

接口：

```text
GET  /api/health
POST /api/analyze
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/files/{path}
```

上传 PDF 后，后端会创建异步 job，输出到：

```text
outputs/upload_jobs/{job_id}/
```

每个 job 包含：

```text
pdfs/
nodes.raw.jsonl
nodes.jsonl
edges.jsonl
kg/
text_index/
visual_index/
candidates.csv
reranked.csv
chain_steps.csv
evidence_card.png
result.json
status.json
```

后端任务顺序：

```text
parse
  -> visual crops + caption
  -> graph
  -> KG/text/visual index
  -> retrieve
  -> rerank
  -> evidence chain
  -> answer
  -> evidence card
```

前端位于 `web/`，用于上传 PDF、查看处理进度、证据卡片、证据链和排序结果。

## 内置样例知识库

仓库内置了一套售后样例知识库，默认落在 `outputs/after_sales_kb/`。它用于验证从 PDF 解析、视觉增强、知识图谱、召回、重排到答案导出的整条链路。

常用命令：

```bash
python scripts/22_enrich_images.py --nodes outputs/after_sales_kb/nodes.jsonl
python scripts/23_build_kg.py \
  --nodes outputs/after_sales_kb/nodes.jsonl \
  --kg-dir outputs/kg \
  --text-dir outputs/text_index \
  --visual-dir outputs/visual_index
python scripts/21_generate_answer.py \
  --questions outputs/after_sales_kb/questions.csv \
  --nodes outputs/after_sales_kb/nodes.jsonl \
  --rankings outputs/after_sales_kb/reranked.csv \
  --visual-index outputs/visual_index/images.jsonl \
  --output outputs/after_sales_kb/full_test_no_llm.csv \
  --cache outputs/after_sales_kb/full_test_no_llm_cache.jsonl \
  --no-llm
```

`22_enrich_images.py` 会调用 Qwen-VL 刷新图片说明；如果只验证已有产物或没有视觉模型密钥，可以跳过。

`--no-llm` 用于不依赖外部文本模型的链路验证；正式生成答案时去掉该参数，并配置 `RAG_ANSWER_MODEL`。

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

推荐环境变量：

```powershell
[Environment]::SetEnvironmentVariable('ARK_API_KEY','你的 Ark API Key','User')
[Environment]::SetEnvironmentVariable('ARK_BASE_URL','https://ark.cn-beijing.volces.com/api/v3','User')
[Environment]::SetEnvironmentVariable('RAG_EMBEDDING_PROVIDER','ark','User')
[Environment]::SetEnvironmentVariable('RAG_EMBEDDING_MODEL','doubao-embedding-vision-250615','User')
[Environment]::SetEnvironmentVariable('ARK_EMBEDDING_DIMENSIONS','1024','User')
[Environment]::SetEnvironmentVariable('RAG_VISUAL_CAPTION_PROVIDER','qwen','User')
[Environment]::SetEnvironmentVariable('RAG_QWEN_VL_MODEL','qwen-vl-plus','User')
[Environment]::SetEnvironmentVariable('RAG_QWEN_API_KEY_ENV','DASHSCOPE_API_KEY','User')
[Environment]::SetEnvironmentVariable('RAG_ANSWER_PROVIDER','ark','User')
[Environment]::SetEnvironmentVariable('RAG_ANSWER_MODEL','你的文本模型 endpoint id','User')
```

仓库提供配置模板：

```text
configs/doubao_optimized.env.example
```

启动后端：

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

启动前端：

```bash
cd web
npm run dev
```

访问：

```text
http://127.0.0.1:5173/
```

## 离线流水线

核心流水线（到证据卡片）：

```bash
python scripts/06_run_pipeline.py \
  --questions data/questions.csv \
  --parser mineru \
  --mineru-backend pipeline \
  --mineru-method auto \
  --visual-caption-provider qwen \
  --candidate-retriever fusion \
  --rerank-retriever fusion \
  --embedding-model doubao-embedding-vision-250615 \
  --embedding-device auto
```

单步运行：

```bash
python scripts/01_parse_pdf.py
python scripts/14_chunk_quality_report.py --nodes outputs/parsed/nodes.jsonl
python scripts/15_visualize_layout_bboxes.py --nodes outputs/parsed/nodes.jsonl --max-pages-per-doc 4
python scripts/10_build_visual_evidence.py
python scripts/22_enrich_images.py --nodes outputs/parsed/nodes.jsonl
python scripts/02_build_graph.py
python scripts/23_build_kg.py --nodes outputs/parsed/nodes.jsonl
python scripts/03_retrieve_candidates.py --retriever fusion
python scripts/04_rerank.py --retriever fusion
python scripts/09_build_evidence_chains.py
python scripts/11_build_evidence_cards.py
python scripts/12_check_evidence_cards.py
python scripts/21_generate_answer.py \
  --questions data/questions.csv \
  --nodes outputs/parsed/nodes.jsonl \
  --rankings outputs/rankings/reranked.csv \
  --visual-index outputs/visual_index/images.jsonl \
  --output outputs/answers.csv \
  --cache outputs/answers_cache.jsonl
python scripts/13_export_frontend_data.py
```

快速消融诊断：

```bash
python scripts/24_ablate_retrieval.py \
  --questions data/questions.csv \
  --nodes outputs/parsed/nodes.jsonl \
  --edges outputs/parsed/edges.jsonl \
  --kg-dir outputs/kg \
  --output-dir outputs/ablation \
  --top-k 10 \
  --candidate-k 50
```

如果要把 embedding 路线也纳入消融：

```bash
python scripts/24_ablate_retrieval.py --include-embedding
```

## 验证记录

仓库当前的内置样例目录已完成一次全量跑通：

- `outputs/after_sales_kb/questions.csv`：400 条问题
- `outputs/after_sales_kb/nodes.jsonl`：6707 条节点
- `outputs/after_sales_kb/edges.jsonl`：34900 条边
- `outputs/after_sales_kb/reranked.csv`：20000 条排序记录
- `outputs/after_sales_kb/evidence_chains/chains.jsonl`：400 条证据链
- `outputs/after_sales_kb/full_test_no_llm.csv`：400 条链路验证答案

## 主要输出

```text
outputs/parsed/nodes.jsonl
outputs/parsed/edges.jsonl
outputs/visual/pages/
outputs/visual/crops/
outputs/kg/entities.jsonl
outputs/kg/relations.jsonl
outputs/text_index/nodes.jsonl
outputs/visual_index/images.jsonl
outputs/rankings/candidates.csv
outputs/rankings/reranked.csv
outputs/evidence_chains/
outputs/evidence_cards/
outputs/metrics/
```

## 技术边界

- 当前向量索引是本地 `.npz` 缓存和内存矩阵检索，不是独立向量数据库。
- KG 是轻量规则和结构字段抽取，不是完整数据库型知识图谱；它用于增强召回和重排，不替代原始证据。
- PDF 表格结构主要依赖 MinerU 输出；复杂跨页表格和图片化表格仍可能需要额外校验。
- 视觉 caption 质量依赖图片裁剪质量和 VLM 能力；裁剪区域错误会影响后续 embedding 和答案。
- 如果配置 Ark embedding-vision，会真实调用图文 embedding API；未配置时会回退到本地 embedding 或非 API 召回路线。
- 答案生成依赖证据链质量；如果解析、召回或图片识别错误，LLM 只能基于错误证据生成相对合理的回答。
