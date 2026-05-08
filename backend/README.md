# 多模态 RAG 后端 API

后端负责上传 PDF 的在线分析流程：

1. 保存用户上传的 PDF。
2. 解析 PDF 为文本、表格、图注、页面节点。
3. 生成页面图和证据裁剪图。
4. 构建图关系。
5. 召回候选证据并进行 G4 重排序。
6. 生成证据链和证据卡片。
7. 返回给前端展示。

启动命令：

```powershell
D:\conda_envs\rag-gpu\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

接口：

```text
GET  /api/health
POST /api/analyze
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/files/{path}
```

前端默认连接 `http://127.0.0.1:8765`。
