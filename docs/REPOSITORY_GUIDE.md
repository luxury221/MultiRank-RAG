# Repository Guide

这份文档说明仓库中哪些内容属于项目源码，哪些属于本地数据或运行产物。

## Tracked Source

建议提交到 Git 的内容：

- `README.md`
- `requirements.txt`
- `backend/`
- `web/src/`
- `web/package.json`
- `web/package-lock.json`
- `configs/`
- `scripts/`
- `docs/`
- `data/sample/`
- `data/pdfs/README.md`

## Local-Only Content

默认不提交：

- `outputs/`: 运行产物、日志、提交文件、缓存和中间索引。
- `DataFountain/`: 外部 benchmark 数据。
- `external/`: 克隆的第三方仓库。
- `web/node_modules/`: 前端依赖。
- `data/pdfs/*.pdf`: 用户或实验 PDF。
- `.env` / `.env.*`: 私有配置和 API key。

## Output Management

`outputs/` 可以保留在本地作为实验工作区，但不要让它成为仓库主体。正式复现应通过脚本重新生成关键产物。

如需保留重要实验结果，建议写入 `docs/EVALUATION.md` 或单独的实验记录文档，而不是提交大量 CSV、JSONL、日志和图片缓存。

## Naming Suggestions

实验文件建议使用清晰命名：

```text
baseline_YYYYMMDD
ablation_<module>_<setting>
candidate_<short_description>
best_known_<metric_or_date>
```

避免使用难以判断含义的文件名，例如 `final2.csv`、`new_new.csv`、`fixed_last.csv`。

## Before Publishing

发布仓库前建议检查：

- `git status --short`
- 是否误提交了 API key。
- 是否误提交了大体积输出目录。
- README 是否聚焦项目本身，而不是某一次实验。
- 前后端启动命令是否仍然可用。

