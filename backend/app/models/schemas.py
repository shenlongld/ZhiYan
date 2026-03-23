from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


class KnowledgeBaseBase(BaseModel):
    """知识库基础模型"""
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")


class KnowledgeBaseCreate(KnowledgeBaseBase):
    """创建知识库请求"""
    pass


class KnowledgeBaseResponse(KnowledgeBaseBase):
    """知识库响应"""
    id: str
    created_at: Optional[datetime] = None
    document_count: int = 0
    chunk_count: int = 0


class KnowledgeBaseListResponse(BaseModel):
    """知识库列表响应"""
    code: int = 0
    message: str = "success"
    data: List[KnowledgeBaseResponse] = []


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    code: int = 0
    message: str = "success"
    data: dict = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = "user"  # user or assistant
    content: str
    created_at: Optional[str] = None


class ChatRequest(BaseModel):
    """聊天请求"""
    knowledge_base_id: str = Field(..., description="知识库ID")
    question: str = Field(..., description="问题内容")
    stream: bool = Field(False, description="是否流式响应")


class ChatResponse(BaseModel):
    """聊天响应"""
    code: int = 0
    message: str = "success"
    data: dict = Field(default_factory=dict)


class ParseStatusResponse(BaseModel):
    """解析状态响应"""
    code: int = 0
    message: str = "success"
    data: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """错误响应"""
    code: int = 1
    message: str
    data: Any = None
