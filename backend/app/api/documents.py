"""
文档管理API路由
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import os
import shutil
from datetime import datetime

from ..models.schemas import (
    DocumentUploadResponse,
    ParseStatusResponse
)
from ..services.ragflow_service import get_ragflow_service

router = APIRouter(prefix="/documents", tags=["文档管理"])

# 上传文件存储目录
UPLOAD_DIR = "uploads"


@router.post("/upload")
async def upload_document(
    knowledge_base_id: str = Form(...),
    file: UploadFile = File(...)
):
    """上传文档到知识库"""
    try:
        # 确保上传目录存在
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # 保存上传的文件
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 调用RAGFlow上传
        service = get_ragflow_service()
        result = await service.upload_document(
            knowledge_base_id=knowledge_base_id,
            file_path=file_path,
            filename=file.filename
        )
        
        return DocumentUploadResponse(
            code=0,
            message="文件上传成功",
            data={
                "id": result.get("id", ""),
                "name": file.filename,
                "knowledge_base_id": knowledge_base_id,
                "status": result.get("status", "uploaded"),
                "size": len(content),
                "created_at": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/{document_id}/parse")
async def parse_document(
    knowledge_base_id: str,
    document_id: str
):
    """触发文档解析"""
    try:
        service = get_ragflow_service()
        result = await service.parse_document(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id
        )
        
        return ParseStatusResponse(
            code=0,
            message="解析任务已提交",
            data={
                "document_id": document_id,
                "status": result.get("status", "parsing"),
                "progress": result.get("progress", 0)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")


@router.get("/{document_id}/status")
async def get_document_status(
    knowledge_base_id: str,
    document_id: str
):
    """获取文档解析状态"""
    try:
        service = get_ragflow_service()
        result = await service.get_document_status(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id
        )
        
        return ParseStatusResponse(
            code=0,
            message="success",
            data={
                "document_id": document_id,
                "status": result.get("status", "unknown"),
                "progress": result.get("progress", 0)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")
