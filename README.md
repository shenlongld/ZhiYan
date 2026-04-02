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
uvicorn app.main:app --reload --port 8900
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

## 保研入营Bar建模（简化深度学习）

已在 `model/train_value.py` 和 `model/train_school.py` 提供可直接运行的建模脚本，基于
`data/baoyan_experience_profiles_*.jsonl` 自动完成：

- 高校入营 bar 建模（特征：院校层级、成绩、竞赛、论文）
- 竞赛/论文含金量反推（使用“无明确成绩但已入营”样本）
- 训练集/验证集划分

运行方式：

```bash
mkdir -p model
python3 model/vectorize.py --input data/baoyan_experience_profiles_20260329_134614.jsonl
python3 model/train_value.py --input data/baoyan_experience_profiles_20260329_134614.jsonl
python3 model/train_school.py --input data/baoyan_experience_profiles_20260329_134614.jsonl
```

默认输出文件：

- `model/result/compact.jsonl`：训练压缩样本
- `model/result/vocab.json`：词表
- `model/result/vectors.jsonl`：向量化样本
- `model/result/comp_value.json`：竞赛含金量
- `model/result/paper_value.json`：论文级别含金量
- `model/result/split.json`：训练/验证划分
- `model/result/school_bar.json`：各高校建议 bar 与权重
