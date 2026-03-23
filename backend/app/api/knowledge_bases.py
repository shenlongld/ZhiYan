"""
知识库管理API路由
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import os
import shutil
from datetime import datetime
import json
import time
import traceback

from ..models.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseListResponse,
    DocumentUploadResponse,
    ParseStatusResponse,
    ErrorResponse
)
from ..services.ragflow_service import get_ragflow_service

router = APIRouter(prefix="/knowledge-bases", tags=["知识库管理"])

LOG_PATH = "/Users/magicmuffin/Documents/大三下/NLP/zhiyan/.cursor/debug-084109.log"
SESSION_ID = "084109"
RUN_ID = "debug_pre"


def _ndjson_log(*, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # 仅用于调试：写入 NDJSON 便于后续逐行分析
    payload = {
        "sessionId": SESSION_ID,
        "runId": RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


@router.get("", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases():
    """获取知识库列表"""
    try:
        service = get_ragflow_service()
        kbs = await service.list_knowledge_bases()
        
        # 转换为响应格式
        data = []
        for kb in kbs:
            data.append(KnowledgeBaseResponse(
                id=kb.get("id", ""),
                name=kb.get("name", ""),
                description=kb.get("description", ""),
                document_count=kb.get("document_count", 0),
                chunk_count=kb.get("chunk_count", 0)
            ))
        
        return KnowledgeBaseListResponse(code=0, message="success", data=data)
    except Exception as e:
        return KnowledgeBaseListResponse(
            code=1, 
            message=f"获取知识库列表失败: {str(e)}", 
            data=[]
        )


@router.post("", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(kb: KnowledgeBaseCreate):
    """创建知识库"""
    try:
        service = get_ragflow_service()

        # #region agent log
        _ndjson_log(
            hypothesis_id="H1",
            location="knowledge_bases.py:create_knowledge_base:entry",
            message="request received",
            data={
                "name": kb.name,
                "description_is_none": kb.description is None,
                "description_len": len(kb.description or ""),
            },
        )
        # #endregion

        result = await service.create_knowledge_base(
            name=kb.name,
            description=kb.description or ""
        )

        # #region agent log
        _ndjson_log(
            hypothesis_id="H2",
            location="knowledge_bases.py:create_knowledge_base:after_service",
            message="service returned",
            data={
                "result_type": type(result).__name__,
                "result_keys": list(result.keys()) if isinstance(result, dict) else None,
            },
        )
        # #endregion
        
        return KnowledgeBaseResponse(
            id=result.get("id", ""),
            name=result.get("name", ""),
            description=result.get("description", ""),
            document_count=0,
            chunk_count=0
        )
    except Exception as e:
        # #region agent log
        _ndjson_log(
            hypothesis_id="H1",
            location="knowledge_bases.py:create_knowledge_base:except",
            message="exception raised in route handler",
            data={
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            },
        )
        # #endregion
        raise HTTPException(status_code=500, detail=f"创建知识库失败: {str(e)}")


@router.get("/{kb_id}/documents")
async def list_documents(kb_id: str):
    """获取知识库中的文档列表"""
    try:
        service = get_ragflow_service()
        docs = await service.list_documents(kb_id)
        return {"code": 0, "message": "success", "data": docs}
    except Exception as e:
        return {"code": 1, "message": str(e), "data": []}
