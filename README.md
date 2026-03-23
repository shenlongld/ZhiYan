# RAG智能问答保研信息交互网站

基于Vue3 + FastAPI + RAGFlow的保研信息知识库问答系统。

## 功能特性

- 上传PDF、Markdown等文档
- 创建和管理RAGFlow知识库
- 上传文件到指定知识库
- 智能问答对话

## 项目结构

```
rag_baoyan/
├── frontend/          # Vue3前端项目
├── backend/          # FastAPI后端项目
└── docs/             # 文档
```

## 快速开始

### 1. 配置后端

```bash
cd backend
pip install -r requirements.txt
```

创建 `.env` 文件：

```env
RAGFLOW_HOST=http://localhost:9380
RAGFLOW_API_KEY=your_ragflow_api_key
```

启动后端：

```bash
uvicorn app.main:app --reload --port 8000
```

### 2. 配置前端

```bash
cd frontend
npm install
npm run dev
```

### 3. 配置RAGFlow

确保RAGFlow服务运行中，并获取API密钥。

## API接口

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/v1/knowledge-bases` | GET | 获取知识库列表 |
| `/api/v1/knowledge-bases` | POST | 创建知识库 |
| `/api/v1/documents/upload` | POST | 上传文档 |
| `/api/v1/documents/{id}/parse` | POST | 触发解析 |
| `/api/v1/documents/{id}/status` | GET | 获取解析状态 |
| `/api/v1/chat` | POST | 智能问答 |

## 技术栈

- **前端**: Vue 3 + Vite + Element Plus + Pinia
- **后端**: Python FastAPI + SQLAlchemy
- **RAG引擎**: RAGFlow API
