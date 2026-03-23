"""
聊天API路由
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from ..models.schemas import ChatRequest, ChatResponse
from ..services.ragflow_service import get_ragflow_service

router = APIRouter(prefix="/chat", tags=["智能问答"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """发送聊天消息"""
    try:
        service = get_ragflow_service()
        result = await service.chat(
            knowledge_base_id=request.knowledge_base_id,
            question=request.question,
            session_id=None
        )
        
        return ChatResponse(
            code=0,
            message="success",
            data={
                "answer": result.get("answer", ""),
                "references": result.get("references", []),
                "question": request.question
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天请求失败: {str(e)}")


@router.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """获取聊天历史"""
    # 实际实现需要存储和返回聊天历史
    return {
        "code": 0,
        "message": "success",
        "data": {
            "session_id": session_id,
            "messages": []
        }
    }
