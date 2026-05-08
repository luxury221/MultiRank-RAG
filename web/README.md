# 多模态 RAG 证据检测页面

这个目录是从 Figma 页面改造来的 React/Vite 前端。页面读取 `public/app-data.json`，展示当前项目里的 PDF、问题、G4 证据链、检索结果和证据卡片。

常用流程：

```bash
python scripts/13_export_frontend_data.py
cd web
npm install
npm run build
cd dist
python -m http.server 5174 --bind 127.0.0.1
```

在当前 WSL 项目路径下，推荐用 Python 静态服务器打开构建产物：

```bash
cd /home/blacklions/workspace/Linux/多模态RAG/web/dist
python3 -m http.server 5174 --bind 127.0.0.1
```

然后访问 `http://127.0.0.1:5174/`。

上传 PDF 功能需要同时启动后端：

```powershell
D:\conda_envs\rag-gpu\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```
