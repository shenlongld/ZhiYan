"""
RAGFlow API 服务封装
"""

import json
import os
import sys
import time
import traceback
from typing import List, Optional, Dict, Any

import httpx

from ..core.config import settings

# 将项目根目录（包含 ragflow.py）加入 Python 搜索路径，便于复用已封装好的调用逻辑
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from ragflow import RAGFlow as RAGFlowWrapper
except Exception:  # pragma: no cover
    RAGFlowWrapper = None

try:
    # ragflow_sdk 的类型校验期望 llm 是 ragflow_sdk.modules.chat.LLM
    from ragflow_sdk.modules.chat import LLM as SDKChatLLM  # type: ignore
except Exception:  # pragma: no cover
    SDKChatLLM = None


class RAGFlowService:
    """RAGFlow API服务类"""
    
    def __init__(self, api_key: Optional[str] = None, host: Optional[str] = None):
        self.api_key = api_key or settings.ragflow_api_key
        self.host = (host or settings.ragflow_host or "").rstrip("/")
        # 这里不再拼 /v1：让调用逻辑和项目根目录 ragflow.py 保持一致
        self.base_url = self.host
        self.client = httpx.AsyncClient(timeout=120.0)
        self._rag: Optional[Any] = None

    def _get_rag(self):
        if not self.api_key:
            return None
        if self._rag is not None:
            return self._rag
        if RAGFlowWrapper is None:
            raise RuntimeError("无法导入项目根目录 ragflow.py（RAGFlowWrapper 为空）")
        # ragflow.py 内部已做了 base_url 尾随斜杠归一等处理
        self._rag = RAGFlowWrapper(api_key=self.api_key, base_url=self.base_url)
        return self._rag
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        """获取知识库列表"""
        if not self.api_key:
            return self._get_mock_knowledge_bases()

        try:
            rag = self._get_rag()
            datasets = rag.list_datasets()
            result: List[Dict[str, Any]] = []
            for ds in datasets:
                raw_dataset = getattr(ds, "_dataset", None)
                # DataSet wrapper 只暴露 id/name，这里尽量从底层对象取 description
                desc = getattr(raw_dataset, "description", None) or ""

                # document_count / chunk_count：先尝试从底层对象拿；拿不到则回退用 list_documents 统计
                document_count = getattr(raw_dataset, "document_count", None)
                chunk_count = getattr(raw_dataset, "chunk_count", None)

                if document_count is None:
                    try:
                        docs = ds.list_documents(page_size=1000)
                        document_count = len(docs)
                    except Exception:
                        document_count = 0

                if chunk_count is None:
                    # chunk_count 统计需要遍历 chunks，代价较高；如果底层对象没有字段则先置 0
                    chunk_count = 0

                result.append(
                    {
                        "id": getattr(ds, "id", "") or "",
                        "name": getattr(ds, "name", "") or "",
                        "description": desc,
                        "document_count": int(document_count or 0),
                        "chunk_count": int(chunk_count or 0),
                    }
                )
            return result
        except Exception as e:
            print(f"获取知识库列表失败: {e}")
            return self._get_mock_knowledge_bases()

    async def list_documents(self, knowledge_base_id: str) -> List[Dict[str, Any]]:
        """获取知识库内文档列表（用于前端展示/文档计数校验）"""
        if not self.api_key:
            return []
        try:
            rag = self._get_rag()
            dataset = rag.get_dataset(knowledge_base_id)
            docs = dataset.list_documents(page_size=1000)
            result: List[Dict[str, Any]] = []
            for doc in docs:
                result.append(
                    {
                        "id": getattr(doc, "id", "") or "",
                        "name": getattr(doc, "name", "") or "",
                    }
                )
            return result
        except Exception as e:
            print(f"获取文档列表失败: {e}")
            return []
    
    async def create_knowledge_base(self, name: str, description: str = "") -> Dict[str, Any]:
        """创建知识库"""
        LOG_PATH = "/Users/magicmuffin/Documents/大三下/NLP/zhiyan/.cursor/debug-084109.log"
        SESSION_ID = "084109"
        RUN_ID = "debug_pre"

        def _ndjson_log(*, hypothesis_id: str, location: str, message: str, data: dict) -> None:
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

        # #region agent log
        _ndjson_log(
            hypothesis_id="H3",
            location="ragflow_service.py:create_knowledge_base:entry",
            message="service call start",
            data={
                "name": name,
                "description_len": len(description or ""),
                "api_key_is_set": bool(self.api_key),
                "base_url": self.base_url,
            },
        )
        # #endregion

        if not self.api_key:
            return {
                "id": f"kb_{name}_{hash(name) % 10000}",
                "name": name,
                "description": description
            }
        
        try:
            rag = self._get_rag()
            embedding_model = getattr(settings, "ragflow_embedding_model", None)
            if embedding_model:
                embedding_model = str(embedding_model).strip()
                # RAGFlow 要求: <model_name>@<provider>
                if ("@" not in embedding_model) or embedding_model.startswith("@") or embedding_model.endswith("@"):
                    _ndjson_log(
                        hypothesis_id="H8",
                        location="ragflow_service.py:create_knowledge_base:invalid_embedding_model",
                        message="embedding_model format invalid",
                        data={"embedding_model": embedding_model},
                    )
                    raise ValueError(
                        "RAGFlow embedding_model 配置格式错误，必须为 <model_name>@<provider>，例如: BAAI/bge-m3@BAAI"
                    )
                dataset = rag.create_dataset(
                    name=name,
                    description=description,
                    embedding_model=embedding_model,
                )
            else:
                # 若 env 未配置或为空，则让 ragflow.py 使用其自身默认 embedding_model
                dataset = rag.create_dataset(
                    name=name,
                    description=description,
                )
            # DataSet wrapper 暴露 id/name，这里尽量取底层 description 字段
            desc = getattr(getattr(dataset, "_dataset", None), "description", None)
            if desc is None:
                desc = description
            data = {
                "id": getattr(dataset, "id", "") or "",
                "name": getattr(dataset, "name", "") or name,
                "description": desc or "",
            }

            # #region agent log
            _ndjson_log(
                hypothesis_id="H5",
                location="ragflow_service.py:create_knowledge_base:after_create",
                message="dataset created",
                data={
                    "returned_type": type(data).__name__,
                    "returned_keys": list(data.keys()) if isinstance(data, dict) else None,
                    "id_is_empty": (not data.get("id")),
                    "embedding_model_setting": str(embedding_model)[:80] if embedding_model is not None else None,
                    "embedding_model_setting_is_empty": (embedding_model == ""),
                },
            )
            # #endregion

            return data
        except Exception as e:
            # #region agent log
            _ndjson_log(
                hypothesis_id="H4",
                location="ragflow_service.py:create_knowledge_base:except",
                message="httpx call failed",
                data={
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback_tail": traceback.format_exc().splitlines()[-8:],
                },
            )
            # #endregion

            print(f"创建知识库失败: {e}")
            # 对于已配置 api_key 的情况：创建失败就直接抛出，让上层明确知道是配置/调用问题
            raise
    
    async def upload_document(self, knowledge_base_id: str, file_path: str, filename: str) -> Dict[str, Any]:
        """上传文档到知识库"""
        if not self.api_key:
            return {
                "id": f"doc_{hash(filename) % 100000}",
                "name": filename,
                "knowledge_base_id": knowledge_base_id,
                "status": "completed"
            }
        
        try:
            rag = self._get_rag()
            dataset = rag.get_dataset(knowledge_base_id)
            with open(file_path, "rb") as f:
                blob = f.read()
            docs = dataset.upload_documents([{"display_name": filename, "blob": blob}])
            doc0 = docs[0] if docs else None
            return {
                "id": getattr(doc0, "id", "") if doc0 else "",
                "name": getattr(doc0, "name", "") if doc0 else filename,
                "knowledge_base_id": knowledge_base_id,
                "status": "completed",
            }
        except Exception as e:
            print(f"上传文档失败: {e}")
            return {
                "id": f"doc_{hash(filename) % 100000}",
                "name": filename,
                "knowledge_base_id": knowledge_base_id,
                "status": "completed"
            }
    
    async def parse_document(self, knowledge_base_id: str, document_id: str) -> Dict[str, Any]:
        """触发文档解析"""
        if not self.api_key:
            return {"status": "completed", "progress": 100}
        
        try:
            rag = self._get_rag()
            dataset = rag.get_dataset(knowledge_base_id)
            # #region agent log
            LOG_PATH = "/Users/magicmuffin/Documents/大三下/NLP/zhiyan/.cursor/debug-084109.log"
            SESSION_ID = "084109"
            RUN_ID = "debug_parse_pre"

            def _ndjson_log(*, hypothesis_id: str, location: str, message: str, data: dict) -> None:
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

            ds_obj = getattr(dataset, "_dataset", None)
            dataset_embedding_model = getattr(ds_obj, "embedding_model", None) or getattr(
                dataset, "embedding_model", None
            )
            _ndjson_log(
                hypothesis_id="H6",
                location="ragflow_service.py:parse_document:pre",
                message="dataset embedding model before parse",
                data={
                    "knowledge_base_id": knowledge_base_id,
                    "document_id": document_id,
                    "dataset_embedding_model_is_none": dataset_embedding_model is None,
                    "dataset_embedding_model": str(dataset_embedding_model)[:120]
                    if dataset_embedding_model is not None
                    else None,
                },
            )
            # #endregion
            # 同步触发；解析进度由 get_document_status 再查询
            dataset.parse_documents([document_id])
            return {"status": "parsing", "progress": 0, "document_id": document_id}
        except Exception as e:
            # #region agent log
            LOG_PATH = "/Users/magicmuffin/Documents/大三下/NLP/zhiyan/.cursor/debug-084109.log"
            SESSION_ID = "084109"
            RUN_ID = "debug_parse_pre"

            def _ndjson_log(*, hypothesis_id: str, location: str, message: str, data: dict) -> None:
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

            _ndjson_log(
                hypothesis_id="H7",
                location="ragflow_service.py:parse_document:except",
                message="parse_documents raised exception",
                data={
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback_tail": traceback.format_exc().splitlines()[-8:],
                },
            )
            # #endregion
            print(f"触发解析失败: {e}")
            return {"status": "completed", "progress": 100}
    
    async def get_document_status(self, knowledge_base_id: str, document_id: str) -> Dict[str, Any]:
        """获取文档解析状态"""
        if not self.api_key:
            return {"status": "completed", "progress": 100}
        
        try:
            rag = self._get_rag()
            dataset = rag.get_dataset(knowledge_base_id)
            docs = dataset.list_documents(id=document_id, page_size=1)
            if not docs:
                return {"status": "unknown", "progress": 0, "document_id": document_id}

            doc = docs[0]
            raw = getattr(doc, "_document", None)
            # 尝试从底层对象取解析状态/进度（字段名可能随 SDK 版本变化）
            status = (
                getattr(raw, "status", None)
                or getattr(raw, "parse_status", None)
                or getattr(raw, "processing_status", None)
            )
            progress = (
                getattr(raw, "progress", None)
                or getattr(raw, "parse_progress", None)
            )

            return {
                "status": status or "unknown",
                "progress": progress if isinstance(progress, int) else 0,
                "document_id": document_id,
            }
        except Exception as e:
            print(f"获取状态失败: {e}")
            return {"status": "completed", "progress": 100}
    
    async def chat(self, knowledge_base_id: str, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """发送对话请求"""
        if not self.api_key:
            return self._get_mock_chat_response(question)
        
        try:
            # 1) 读取 prompts/1.txt,2.txt,3.txt 作为固定上下文（暂不依赖知识库切片）
            backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            prompts_dir = os.path.join(backend_root, "app", "prompts")
            ctx_files = ["1.txt", "2.txt", "3.txt"]

            context_parts: List[str] = []
            reference_sources: List[str] = []
            for fn in ctx_files:
                p = os.path.join(prompts_dir, fn)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        txt = f.read().strip()
                    if txt:
                        context_parts.append(f"[{fn}]\n{txt}")
                        reference_sources.append(fn)
                except FileNotFoundError:
                    # 缺文件则跳过，避免直接 500
                    continue

            context = "\n\n".join(context_parts).strip()

            # 2) 读取提示词文件并填充变量
            prompt_path = getattr(settings, "ragflow_qa_prompt_path", "app/prompts/external_qa_prompt.txt")
            abs_prompt_path = os.path.join(backend_root, prompt_path)
            with open(abs_prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            prompt = prompt_template.format(question=question, context=context)

            # 3) 调用外接大模型 API（OpenAI 兼容 /chat/completions）
            llm_model = getattr(settings, "ragflow_llm_model", None)
            llm_api_key = getattr(settings, "ragflow_llm_api_key", None)
            llm_base_url = getattr(settings, "ragflow_llm_base_url", None)
            if not llm_model or not llm_base_url:
                raise RuntimeError("外接大模型参数未配置：需要 ragflow_llm_model / ragflow_llm_base_url")

            url = llm_base_url.rstrip("/") + "/chat/completions"
            headers: Dict[str, str] = {"Content-Type": "application/json"}
            if llm_api_key:
                headers["Authorization"] = f"Bearer {llm_api_key}"

            payload: Dict[str, Any] = {
                "model": llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": getattr(settings, "ragflow_llm_temperature", 0.1),
                "max_tokens": getattr(settings, "ragflow_llm_max_tokens", 2048),
            }

            resp = await self.client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            answer = ""
            try:
                answer = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or ""
                )
            except Exception:
                answer = ""

            # references 前端要求每项至少有 source 字段
            references = [{"source": s} for s in reference_sources[:5]]

            return {"answer": answer or "", "references": references}
        except Exception as e:
            print(f"聊天请求失败: {e}")
            return self._get_mock_chat_response(question)
    
    def _get_mock_knowledge_bases(self) -> List[Dict[str, Any]]:
        """返回模拟知识库数据（未配置API时使用）"""
        return [
            {
                "id": "kb_demo_1",
                "name": "保研经验分享",
                "description": "学长学姐的保研经验汇总",
                "document_count": 5,
                "chunk_count": 128
            },
            {
                "id": "kb_demo_2",
                "name": "夏令营信息",
                "description": "各大高校夏令营招生信息",
                "document_count": 12,
                "chunk_count": 256
            }
        ]
    
    def _get_mock_chat_response(self, question: str) -> Dict[str, Any]:
        """返回模拟聊天响应"""
        responses = [
            f"根据您的问题「{question}」，以下是相关建议：\n\n1. 建议提前准备好个人简历和成绩单\n2. 关注各高校的夏令营报名时间\n3. 准备好专业课和英语面试\n\n如有更多问题，欢迎继续提问！",
            f"关于「{question}」这个问题，我建议：\n\n1. 首先了解目标院校的招生要求\n2. 提前联系导师，了解课题组情况\n3. 准备好研究计划书\n\n祝您保研顺利！"
        ]
        import random
        return {
            "answer": random.choice(responses),
            "references": [
                {"content": "参考内容片段1...", "source": "保研指南.pdf"},
                {"content": "参考内容片段2...", "source": "面试技巧.md"}
            ]
        }
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()


# 全局服务实例
ragflow_service: Optional[RAGFlowService] = None


def get_ragflow_service() -> RAGFlowService:
    """获取RAGFlow服务实例"""
    global ragflow_service
    if ragflow_service is None:
        ragflow_service = RAGFlowService()
    return ragflow_service
