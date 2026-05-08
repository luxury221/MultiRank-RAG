# 面向复杂文档问答的查询感知跨模态节点重要性重排序

本项目围绕开题报告中的目标实现一个最小可行闭环：复杂 PDF 被拆成文本、表格、图表、页面、图注等证据节点，系统先召回 Top-K 候选证据，再对候选池执行 G0-G3 四组排序，对比原始排序、语义排序、查询感知 PPR 和 Bridge 跨模态桥接分数的效果。

当前仓库已经包含一套不依赖外部 API 的简化解析与重排序路线，`external/RAG-Anything` 作为可选上游保留。如果 RAG-Anything/MinerU 跑通，可以把它导出的 `content_list` JSON 交给本项目脚本；如果短期跑不通，也可以直接用本项目的 PDF 提取和人工补节点方式完成实验。

## 目录结构

```text
data/
  pdfs/                 # 放 2-3 份复杂 PDF
  questions.csv          # 问题、答案和证据标注
  manual_nodes.csv       # 自动解析缺失时的人工补充节点
  sample/                # 可自检的示例节点和问题
outputs/
  parsed/                # nodes.jsonl、edges.jsonl
  rankings/              # candidates.csv、reranked.csv
  metrics/               # per_question_metrics.csv、summary_metrics.csv
scripts/
  01_parse_pdf.py
  02_build_graph.py
  03_retrieve_candidates.py
  04_rerank.py
  05_evaluate.py
  06_run_pipeline.py
demo/
  app.py
docs/
  项目文档、讲稿、模板
```

## 环境准备

```bash
pip install -r requirements.txt
```

如果暂时不安装 `pypdf/pdfplumber`，脚本仍能运行，但真实 PDF 只能生成占位节点。建议安装后再解析 PDF。

## 快速自检

仓库内置了 `data/sample` 示例，可直接验证重排序和指标链路：

```bash
python scripts/06_run_pipeline.py --sample
```

成功后会生成：

- `outputs/rankings/candidates.csv`
- `outputs/rankings/reranked.csv`
- `outputs/metrics/per_question_metrics.csv`
- `outputs/metrics/summary_metrics.csv`

## 使用真实 PDF

1. 将 PDF 放入 `data/pdfs/`。
2. 按 `data/questions.csv` 的表头填写 20-30 个问题。
3. 如果自动解析漏掉表格、图表或图注，把关键证据补入 `data/manual_nodes.csv`。
4. 运行完整流水线：

```bash
python scripts/06_run_pipeline.py
```

单步运行也可以：

```bash
python scripts/01_parse_pdf.py
python scripts/02_build_graph.py
python scripts/03_retrieve_candidates.py
python scripts/04_rerank.py
python scripts/05_evaluate.py
```

## 数据格式

`data/questions.csv` 字段：

```text
question_id,doc_id,question,answer,question_type,gold_node_ids,gold_pages,gold_modalities,evidence_note
```

`outputs/parsed/nodes.jsonl` 字段：

```text
node_id, doc_id, page, node_type, content, source_ref
```

`outputs/parsed/edges.jsonl` 字段：

```text
source_id, target_id, edge_type, weight
```

## 四组排序

- G0：保留初步召回的 Top-K 原始顺序。
- G1：仅按 `Sim(q, v)` 语义相似度重排。
- G2：融合语义相似度和查询感知 PPR。
- G3：融合语义相似度、查询感知 PPR 和 Bridge 分数。

默认融合权重：

```text
G2: alpha = 0.7, beta = 0.3
G3: lambda_s = 0.6, lambda_p = 0.25, lambda_b = 0.15
```

## 启动 Demo

```bash
streamlit run demo/app.py
```

Demo 会优先读取真实输出；如果还没有真实数据，会回退到 `data/sample` 示例。页面中可以选择问题、查看 G0/G3 Top-K 对比、答案页码和节点关系，也支持上传 PDF 后重新解析建图。

## RAG-Anything 接入方式

如果 RAG-Anything 已经成功解析文档并导出了 `content_list` JSON，可以运行：

```bash
python scripts/01_parse_pdf.py --content-list path/to/content_list.json
python scripts/02_build_graph.py
python scripts/03_retrieve_candidates.py
python scripts/04_rerank.py
python scripts/05_evaluate.py
```

这条路线会把 RAG-Anything 的文本、表格、图像、公式等内容转成本项目统一节点格式，再进入同一套 G0-G3 重排序和评测流程。

## 最低交付检查

- `data/pdfs/` 至少 2 份 PDF。
- `data/questions.csv` 至少 20 个问题，并补全标准答案和证据标注。
- `outputs/rankings/reranked.csv` 包含 G0-G3 四组结果。
- `outputs/metrics/summary_metrics.csv` 可用于汇报指标对比。
- Demo 能展示至少 3 个成功案例和 2 个失败案例。
