# Repository Guide

这份文档说明仓库中哪些内容属于正式项目源码，哪些属于本地数据、运行产物或比赛归档。

## Tracked Source

建议提交到 Git 的内容：

- `README.md`
- `requirements.txt`
- `pyproject.toml`
- `backend/`
- `web/src/`
- `web/package.json`
- `web/package-lock.json`
- `configs/`
- `multirank_rag/`
- `scripts/`
- `docs/`
- `projects/complex_document_qa/`
- `competitions/tianchi_legal/`
- `data/sample/`
- `data/pdfs/README.md`
- `data/tianchi_legal/README.md`

## Project Tracks

```text
projects/complex_document_qa/      复杂文档问答系统主项目
competitions/tianchi_legal/        阿里天池法律问答比赛归档
```

主项目和比赛分区都通过清单文件引用根目录中的稳定代码路径。这样可以让仓库更清楚，同时不破坏已有启动命令。

## Local-Only Content

默认不提交：

- `outputs/`: 运行产物、日志、缓存和大部分中间索引。
- `external/`: 克隆的第三方仓库。
- `web/node_modules/`: 前端依赖。
- `data/pdfs/*.pdf`: 用户或实验 PDF。
- `data/uploads/`: 前端上传文件。
- `data/tianchi_legal/raw/`: 天池官方原始数据。
- `data/tianchi_legal/extracted/`: 天池解压和解析产物。
- `data/tianchi_legal/processed/`: 天池规范化中间产物。
- `.env` / `.env.*`: 私有配置和 API key。
- 模型权重、向量缓存和大体积二进制产物。

## Preserved Artifacts

少量关键材料会被明确保留：

- `projects/complex_document_qa/reports/开题报告.docx`
- `competitions/tianchi_legal/submissions/*.jsonl`

这些文件用于保证项目完整度和实验可追溯性，不代表所有输出文件都需要进入 Git。

## Before Publishing

发布仓库前建议检查：

- `git status --short`
- 是否误提交了 API key。
- 是否误提交了大体积模型权重、缓存或完整官方数据集。
- README 是否聚焦项目本身，而不是某一次临时实验。
- 前后端启动命令是否仍然可用。
