"""
LLM 结构化提取器：内置三套 Prompt + JSON 容错解析 + 日期转换辅助。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


PROMPT_POLICY = """Role: 专业的保研数据专家。
Task: 从网页文本提取：学校学院、标题、起止日期(YYYY-MM-DD)、硬门槛(排名/英语)、链接。
Constraint: 无日期填 null，JSON 格式输出。

重要：用户消息开头会给出【权威页面URL】，字段 official_url 必须与该 URL 完全一致（逐字符相同），禁止填写登录页、SSO、门户首页或猜测链接。

输出字段:
- school_college
- announcement_title
- start_date
- end_date
- admission_threshold (对象，至少包含 ranking_requirement, english_requirement, other_constraints)
- official_url（必须等于【权威页面URL】）
仅输出一个 JSON 对象，不要输出解释。
"""

PROMPT_EXPERIENCE = """Role: 保研面经分析师。
Task: 从知乎/小红书内容提取：专业、流程(用+连接)、真题记录(按类别)、导师标签(如["学术牛"])、年份、核心摘要。
输出字段:
- school_major
- assessment_flow
- question_bank (对象，建议机考/笔试/面试分类)
- mentor_tags (字符串数组)
- experience_year (整数，可为 null)
- summary_digest
仅输出 JSON。
"""

PROMPT_QUOTA = """Role: 数据分析官。
Task: 提取专业代码及名称、今年拟招数、去年拟招数、是否全日制。
输出字段:
- school_name
- dept_name
- major_full_name
- current_year_quota
- prev_year_quota
- is_full_time
仅输出 JSON。
"""


class LLMExtractor:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY 不能为空")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _chat(
        self, prompt: str, content: str, *, max_user_chars: int = 16000
    ) -> str:
        body = content[:max_user_chars] if len(content) > max_user_chars else content
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": body},
            ],
        )
        return resp.choices[0].message.content or ""

    async def scan_and_download_attachments(
        self, html_or_text: str, base_url: str, *, max_files: int = 5
    ) -> List[Path]:
        """
        从网页 HTML/正文中识别 .pdf/.docx/.xlsx 链接，异步下载到项目根目录 temp_files/。
        """
        try:
            from . import attachments as att
        except ImportError:
            import attachments as att  # type: ignore

        urls = att.scan_attachment_urls(html_or_text, base_url)
        att.ensure_temp_dir()
        paths: List[Path] = []
        async with httpx.AsyncClient() as client:
            for url in urls[:max_files]:
                p = await att.download_attachment(client, url, att.TEMP_DIR)
                if p:
                    paths.append(p)
        return paths

    async def extract_policy_with_attachments(
        self, tavily_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        合并网页 Markdown + 附件解析结果，再调用政策提取；失败时仅记录日志，由调用方决定是否回退。
        """
        try:
            from . import attachments as att
        except ImportError:
            import attachments as att  # type: ignore

        base_url = tavily_item.get("url") or ""
        raw = tavily_item.get("raw_content") or ""
        snippet = tavily_item.get("content") or ""
        scan_blob = f"{raw}\n{snippet}"

        if raw.strip() and ("<" in raw[:500] or "<html" in raw.lower()[:500]):
            web_md = att.html_to_markdown(raw)
        else:
            web_md = raw or snippet

        paths: List[Path] = []
        try:
            paths = await self.scan_and_download_attachments(scan_blob, base_url)
            sections: List[tuple[str, str]] = []
            for p in paths:
                md = att.parse_attachment_file(p)
                if md.strip():
                    sections.append((p.name, md))
            merged = att.merge_web_and_attachments(web_md, sections)
            merged = att.truncate_for_llm(merged, max_tokens=10000)
            canon = f"【权威页面URL】\n{base_url}\n\n"
            raw_llm = await self._chat(
                PROMPT_POLICY, canon + merged, max_user_chars=32000
            )
            return await self.parse_json_with_retry(raw_llm)
        except Exception as e:
            logger.warning("附件增强提取失败，将交由调用方回退: %s", e)
            raise
        finally:
            att.safe_unlink_paths(paths)

    def _strip_code_fence(self, text: str) -> str:
        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r"\s*```\s*$", "", raw).strip()
        return raw

    def _loads_first_json_object(self, text: str) -> Dict[str, Any]:
        """
        只解析第一个顶层 JSON 对象，避免：
        - 模型输出两个 {...}{...}
        - 第一个 } 之后还有解释文字 导致 json.loads 报 Extra data
        """
        raw = self._strip_code_fence(text)
        start = raw.find("{")
        if start == -1:
            raise ValueError("未找到 JSON 对象")
        decoder = json.JSONDecoder()
        try:
            obj, _end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError("根节点必须是 JSON 对象")
        return obj

    async def parse_json_with_retry(self, raw_response: str) -> Dict[str, Any]:
        try:
            return self._loads_first_json_object(raw_response)
        except Exception:
            fixed = await self._chat(
                "你是 JSON 修复器。请把输入修复为严格 JSON 对象，仅输出 JSON。",
                raw_response,
            )
            return self._loads_first_json_object(fixed)

    async def extract_policy(
        self, content: str, *, max_user_chars: int = 16000
    ) -> Dict[str, Any]:
        raw = await self._chat(
            PROMPT_POLICY, content, max_user_chars=max_user_chars
        )
        return await self.parse_json_with_retry(raw)

    async def extract_experience(self, content: str) -> Dict[str, Any]:
        raw = await self._chat(PROMPT_EXPERIENCE, content)
        return await self.parse_json_with_retry(raw)

    async def extract_quota(self, content: str) -> Dict[str, Any]:
        raw = await self._chat(PROMPT_QUOTA, content)
        return await self.parse_json_with_retry(raw)


def to_date(value: Any) -> Optional[date]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, date):
        return value
    s = str(value).strip()[:10]
    try:
        year, month, day = s.split("-")
        return date(int(year), int(month), int(day))
    except Exception:
        return None
