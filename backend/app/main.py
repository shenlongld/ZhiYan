"""
FastAPI应用主入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .api import knowledge_bases_router, documents_router, chat_router
from .core.config import settings

# 创建FastAPI应用
app = FastAPI(
    title="RAG智能问答保研信息交互平台",
    description="基于RAGFlow的保研信息知识库问答系统API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(knowledge_bases_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG智能问答保研信息交互平台 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "ragflow_configured": bool(settings.ragflow_api_key)
    }


# 确保上传目录存在
os.makedirs(settings.upload_dir, exist_ok=True)
