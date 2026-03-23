from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # RAGFlow配置
    ragflow_host: str = "http://localhost:9380"
    ragflow_api_key: Optional[str] = None

    # RAGFlow 数据集嵌入模型（embedding model）
    ragflow_embedding_model: str = "BAAI/bge-large-zh-v1.5@BAAI"

    # RAGFlow 问答模型（LLM：支持 OpenAI 兼容 API 调用）
    ragflow_llm_model: str = "gpt-4o-mini"
    ragflow_llm_api_key: Optional[str] = None
    ragflow_llm_base_url: Optional[str] = "https://api.openai.com/v1"
    ragflow_llm_temperature: float = 0.1
    ragflow_llm_max_tokens: int = 2048

    # 外接大模型问答：提示词模板文件（单文件配置）
    # 该路径相对于 backend 根目录，例如: app/prompts/external_qa_prompt.txt
    ragflow_qa_prompt_path: str = "app/prompts/external_qa_prompt.txt"

    # 外接大模型问答：检索结果取多少条作为上下文
    ragflow_retrieve_top_k: int = 5
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8900
    debug: bool = True
    
    # 上传配置
    upload_dir: str = "uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
