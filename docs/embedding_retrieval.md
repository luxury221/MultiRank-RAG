# Embedding 检索说明

当前默认流程拆成两段：

```text
候选召回 G0: lexical Top-50
重排 G1-G3: hybrid Top-10
```

G0 作为关键词基线保留下来，G1/G2/G3 使用 embedding + lexical 的 hybrid 相似度，这样对比时能清楚看到“语义检索”和“图结构重排”的影响。

默认 hybrid 分数为：

```text
hybrid_score = 0.7 * normalized_embedding_score + 0.3 * normalized_lexical_score
```

embedding 模型默认使用 `BAAI/bge-m3`，节点向量缓存到：

```text
outputs/embeddings/
```

模型文件缓存到 Conda 环境变量配置的：

```text
D:\hf_cache
```

## 常用命令

默认运行：

```powershell
$env:HF_HOME='D:\hf_cache'; $env:TRANSFORMERS_CACHE='D:\hf_cache'; $env:TEMP='D:\pip_tmp'; $env:TMP='D:\pip_tmp'; Push-Location "\\wsl.localhost\Ubuntu-22.04\home\blacklions\workspace\Linux\多模态RAG"; D:\conda_envs\rag-gpu\python.exe scripts\06_run_pipeline.py --skip-parse; Pop-Location
```

显式指定候选池和最终重排数量：

```powershell
$env:HF_HOME='D:\hf_cache'; $env:TRANSFORMERS_CACHE='D:\hf_cache'; $env:TEMP='D:\pip_tmp'; $env:TMP='D:\pip_tmp'; Push-Location "\\wsl.localhost\Ubuntu-22.04\home\blacklions\workspace\Linux\多模态RAG"; D:\conda_envs\rag-gpu\python.exe scripts\06_run_pipeline.py --skip-parse --candidate-k 50 --rerank-k 10; Pop-Location
```

只改变重排检索器：

```powershell
$env:HF_HOME='D:\hf_cache'; $env:TRANSFORMERS_CACHE='D:\hf_cache'; $env:TEMP='D:\pip_tmp'; $env:TMP='D:\pip_tmp'; Push-Location "\\wsl.localhost\Ubuntu-22.04\home\blacklions\workspace\Linux\多模态RAG"; D:\conda_envs\rag-gpu\python.exe scripts\06_run_pipeline.py --skip-parse --candidate-retriever lexical --rerank-retriever embedding; Pop-Location
```

兼容旧参数，把候选召回和重排都改成同一种检索器：

```powershell
$env:HF_HOME='D:\hf_cache'; $env:TRANSFORMERS_CACHE='D:\hf_cache'; $env:TEMP='D:\pip_tmp'; $env:TMP='D:\pip_tmp'; Push-Location "\\wsl.localhost\Ubuntu-22.04\home\blacklions\workspace\Linux\多模态RAG"; D:\conda_envs\rag-gpu\python.exe scripts\06_run_pipeline.py --skip-parse --retriever hybrid --hybrid-alpha 0.7; Pop-Location
```

`--hybrid-alpha` 越大越偏语义 embedding，越小越偏关键词精确匹配。

## 当前重排权重

当前数据集上，图结构信号改成按问题类型动态启用：

- 文本事实、证据定位：基本回退到 G1 相似度排序，避免同页噪声误伤。
- 表格问答、图表理解、跨模态综合：启用更强的 Bridge，突出文本与表格/图/图注之间的关系。
- 如果问题显式出现 `Figure 2`、`Table 3`、`图 4`、`表 7`，G3 会额外计算 `ref_score`，把对应编号的图、表、图注、表题和相邻正文拉近。
- `ref_score` 也按题型动态启用：表格问答和图文一致性较强，图表理解中等，跨模态很轻，文本事实和证据定位关闭。
- 图文一致性：保守启用，优先避免把相邻但不相关的图片推上来。

默认基础权重为：

```text
G2 = dynamic(Sim + PPR)
G3 = dynamic(Sim + Bridge + Ref)

G2 base: 0.93 * Sim + 0.07 * PPR
G3 base: 0.85 * Sim + 0.15 * Bridge
G3 with explicit Figure/Table reference: Sim + Bridge + dynamic(Ref)
```

动态权重会把未启用的图结构权重还给 Sim，所以不适合图结构的问题不会被强行扰动。

每次运行完整流水线后，还会生成：

```text
outputs/metrics/g1_g3_comparison.csv
outputs/metrics/type_comparison.csv
outputs/visual/pages/<doc_id>/page_*.png
outputs/visual/crops/<doc_id>/<node_id>.png
outputs/evidence_chains/chains.jsonl
outputs/evidence_chains/chain_steps.csv
outputs/evidence_chains/evidence_chains.md
```

对比表用于直接查看 G3 相比 G1 哪些问题提升、哪些问题退步，以及不同问题类型上的平均变化。

其中 `outputs/evidence_chains/` 是 G4 证据链输出：

- `chains.jsonl`：每个问题一条完整证据链，适合程序读取。
- `chain_steps.csv`：每个证据节点一行，适合页面展示和表格分析。
- `evidence_chains.md`：可直接阅读的证据链展示稿。

证据链默认按“主证据 -> 显式编号证据 -> 图表/图注 -> 图关系补充 -> 上下文文本”的顺序组织。

`outputs/visual/` 是视觉证据层输出：

- `pages/`：PDF 页面渲染图，用于页面级定位。
- `crops/`：证据节点对应的真实裁剪图，包括文本块、表格、图注和图像区域。
- 节点文件 `outputs/parsed/nodes.jsonl` 会新增 `page_image_path`、`crop_image_path`、`bbox`、`bbox_source`、`visual_summary` 字段。

默认流水线会生成真实裁剪与 bbox。若要实验性启用 VLM caption，可额外传入：

```powershell
D:\conda_envs\rag-gpu\python.exe scripts\06_run_pipeline.py --skip-parse --visual-caption-model Salesforce/blip-image-captioning-base --visual-max-captions 20
```

当前 BLIP-base 对学术图表的描述较弱，更适合作为链路验证；最终展示建议优先使用真实裁剪图、bbox 和图表上下文摘要。

若使用阿里云百炼/Qwen-VL，可先设置环境变量：

```powershell
$env:DASHSCOPE_API_KEY="你的key"
```

然后只处理 1 张裁剪图做连通测试：

```powershell
D:\conda_envs\rag-gpu\python.exe scripts\10_build_visual_evidence.py --caption-provider qwen --qwen-model qwen-vl-plus --max-captions 1
```

确认效果后，可把 Qwen-VL 接入完整流水线：

```powershell
D:\conda_envs\rag-gpu\python.exe scripts\06_run_pipeline.py --skip-parse --visual-caption-provider qwen --qwen-model qwen-vl-plus --visual-max-captions 20
```

若使用火山方舟/豆包视觉理解，建议把结果写入独立字段，避免覆盖 Qwen 结果：

```powershell
$env:ARK_API_KEY="你的火山方舟key"
$env:ARK_MODEL="你的豆包视觉理解Endpoint或模型名"
D:\conda_envs\rag-gpu\python.exe scripts\10_build_visual_evidence.py --caption-provider doubao --caption-field-prefix doubao --max-captions 1 --caption-node-ids TB_AI综述类_4_1
```

这会生成 `doubao_visual_caption`、`doubao_visual_title`、`doubao_qa_evidence` 等并行字段，便于和 Qwen 结果做同图对比。
