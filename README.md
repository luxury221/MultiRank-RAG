# MultiRank-RAG

MultiRank-RAG 是一个面向复杂 PDF、产品手册和图文混合知识库的多模态 RAG 证据问答系统。项目将 PDF 解析、视觉证据理解、图文检索、轻量知识图谱、分层重排序和证据链生成整合为一条可复现的工程链路，目标是让回答不仅能生成结论，还能说明结论由哪些文本、图片、表格或页面结构支撑。

当前仓库聚焦于一个完整的多模态 RAG 应用系统：用户可以上传或选择 PDF 文档，系统解析文档中的文本、图片、表格和版面关系，围绕用户问题检索证据、生成答案，并在前端展示可追溯的证据链。仓库中的公开数据集仅作为 benchmark 与工程调试参考，不是项目的核心叙事。

## Highlights

- 多模态证据节点：统一建模 text、title、table、figure、caption、equation 等节点。
- MinerU PDF 解析：保留章节、页码、bbox、图表和版面结构信息。
- 视觉证据增强：对图片和图表生成 caption、OCR、key objects、QA evidence 等字段。
- 多路召回：支持 BM25、lexical、embedding、KG、visual、fusion 等召回方式。
- MultiRank G0-G4 重排序：融合语义相似度、PPR 图传播、bridge、reference、visual grounding、domain、product 和 KG 信号。
- 证据链生成：回答前组织主证据、上下文证据、图片证据和结构化关系。
- 工程化演示：提供离线脚本、FastAPI 服务、React 可视化页面和输出质量诊断工具。

## Architecture

```text
PDF / Manual / Reference Knowledge Base
        |
        v
PDF Parser
  - MinerU / native parser
  - page, section, bbox, table, figure
        |
        v
Evidence Node Builder
  - nodes.jsonl
  - edges.jsonl
  - unified evidence schema
        |
        v
Visual Evidence Enrichment
  - image crop
  - visual_caption
  - OCR / key_objects / qa_evidence
        |
        v
Index and KG Layer
  - BM25 / lexical index
  - embedding index
  - visual index
  - product, part, action, fault, policy graph
        |
        v
Hybrid Retrieval
  - candidate pool
  - question routing
  - optional query expansion
        |
        v
MultiRank Reranking
  - G0 raw retrieval
  - G1 semantic
  - G2 semantic + PPR
  - G3 bridge + reference
  - G4 visual + chain + KG + product/domain
        |
        v
Evidence Chain and Answer
  - grounded prompt
  - self-check
  - PIC suffix validation
  - API / UI / CSV output
```

更多细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## Repository Layout

```text
backend/      FastAPI 后端，支持上传 PDF、运行分析任务、返回证据链
web/          React + Vite 前端，用于证据检测和可视化展示
scripts/      离线流水线脚本，覆盖解析、召回、重排、生成、诊断
configs/      环境变量示例和 chunk 模板配置
data/         本地输入问题、样例数据和待解析 PDF 目录
docs/         架构说明、实验记录、展示材料和仓库维护说明
demo/         早期 Streamlit 演示入口
outputs/      运行产物目录，默认不提交到 Git
external/     外部依赖仓库，默认不提交到 Git
```

## Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

参考配置文件：

```text
configs/doubao_optimized.env.example
```

关键配置项包括：

```bash
RAG_PDF_PARSER=mineru
RAG_EMBEDDING_MODEL=doubao-embedding-vision-250615
RAG_ANSWER_MODEL=<your-doubao-endpoint-id>
RAG_BACKEND_CANDIDATE_RETRIEVER=fusion
RAG_BACKEND_RERANK_RETRIEVER=fusion
RAG_BACKEND_ENABLE_KG=1
```

实际 API key 请写入本机环境变量或私有 `.env` 文件，不要提交到 Git。

### 3. Run Offline Pipeline

```bash
python scripts/06_run_pipeline.py \
  --questions data/questions.csv \
  --candidate-k 50 \
  --rerank-k 10
```

常用参数：

```bash
--skip-parse
--skip-visual
--skip-kg
--candidate-retriever fusion
--rerank-retriever fusion
```

### 4. Run Backend

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

主要接口：

```text
GET  /api/health
POST /api/analyze
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/files/{path}
```

### 5. Run Frontend

```bash
cd web
npm install
npm run dev
```

前端默认连接：

```text
http://127.0.0.1:8765
```

## Data and Benchmarks

仓库支持两类输入：

- 自定义 PDF：放入 `data/pdfs/` 或通过前端上传。
- 参考 benchmark：例如本地公开知识库目录，用于验证检索、重排、图文证据选择和答案生成效果。

这些数据用于验证系统能力，而不是定义项目本身。正式项目说明、答辩和 README 应重点描述系统架构、技术方法、证据链能力和前后端体验。

## Important Notes

- `outputs/`、外部数据目录、`external/` 和 `web/node_modules/` 默认不进入版本库。
- 不要把 API key、私有文档、外部数据集或生成的大量中间文件发布到 GitHub。
- benchmark 结果只能作为参考，不能替代真实业务场景下的多模态证据质量评估。
- 本地格式校验只能保证产物结构正确，不能等价于答案质量评分。

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): 系统架构和核心链路。
- [docs/EVALUATION.md](docs/EVALUATION.md): 评估方法、benchmark 用法和已知局限。
- [docs/REPOSITORY_GUIDE.md](docs/REPOSITORY_GUIDE.md): 仓库整理和协作规范。
- [scripts/README.md](scripts/README.md): 脚本入口说明。
- [backend/README.md](backend/README.md): 后端接口说明。
- [web/README.md](web/README.md): 前端启动说明。

## Current Status

项目已经具备完整原型能力，包括离线文档问答、上传 PDF 分析、证据链展示和证据卡片生成。当前主要改进方向是提升检索稳定性、降低图片误匹配、建立更可靠的本地评估流程，并将实验版本管理得更清晰。
