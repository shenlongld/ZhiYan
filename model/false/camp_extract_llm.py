"""
从经验贴文本中抽取「目标院校·学院」及对该营是否入营（任务 1 夏令营维度）。
使用 OpenAI 兼容接口（含 DeepSeek：OPENAI_BASE_URL=https://api.deepseek.com/v1）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[1]
_POSTGRAD_ENV = _REPO_ROOT / "postgrad_agent" / ".env"


def load_postgrad_dotenv() -> None:
    """读取仓库内 postgrad_agent/.env（与采集端共用 OPENAI_* / 网关）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if _POSTGRAD_ENV.is_file():
        load_dotenv(_POSTGRAD_ENV)
    else:
        load_dotenv()


SYSTEM_PROMPT = """你是保研数据抽取助手。根据发帖者的经验描述，列出其**尝试或提及的每一个目标院校夏令营/预推免**（粒度：学校全称·学院全称；学院实在无法判断时用「学校全称·学院未标注」）。

对每个目标单独判断：发帖者对该营是否**获得入营资格/参加夏令营**。
- admitted=true：明确入营、参营、拿到营资格等。
- admitted=false：明确未入营、被拒、未过初审、简历筛未过等。
- admitted=null：未提及该营结果或无法判断。

只输出一个 JSON 数组，不要 markdown、不要解释。数组元素形如：
{"school_college":"浙江大学·计算机科学与技术学院","admitted":true}
若无任何目标院校，输出 []。"""


def _strip_code_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    return raw


def parse_llm_camp_json_array(raw_response: str) -> List[Dict[str, Any]]:
    s = _strip_code_fence(raw_response)
    start = s.find("[")
    if start == -1:
        return []
    decoder = json.JSONDecoder()
    try:
        arr, _ = decoder.raw_decode(s, start)
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    return [x for x in arr if isinstance(x, dict)]


def camp_llm_user_blob(row: Dict[str, Any]) -> str:
    """与建模一致的可见文本，供 LLM 推断多校入营情况。"""
    parts: List[str] = []
    camp = row.get("camp_admission")
    if isinstance(camp, dict):
        if camp.get("detail"):
            parts.append(f"【夏令营/申请详情】\n{camp.get('detail')}")
        if camp.get("admitted") is not None:
            parts.append(f"【结构化入营标记】admitted={camp.get('admitted')}")
    if row.get("admission_detail"):
        parts.append(f"【申请说明】\n{row.get('admission_detail')}")
    if row.get("source_title"):
        parts.append(f"【标题】\n{row.get('source_title')}")
    if row.get("notes"):
        parts.append(f"【备注】\n{row.get('notes')}")
    if row.get("blogger_school"):
        parts.append(f"【本科学校】{row.get('blogger_school')}")
    if row.get("target_major"):
        parts.append(f"【专业/方向】{row.get('target_major')}")
    return "\n\n".join(parts).strip() or "(无正文)"


def _llm_env() -> Tuple[str, str, str]:
    api_key = (
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("RAGFLOW_LLM_API_KEY") or "").strip()
        or (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    )
    base_url = (
        (os.getenv("OPENAI_BASE_URL") or "").strip()
        or (os.getenv("DEEPSEEK_BASE_URL") or "").strip()
    )
    model = (
        (os.getenv("OPENAI_MODEL") or "").strip()
        or (os.getenv("DEEPSEEK_MODEL") or "").strip()
        or "deepseek-chat"
    )
    return api_key, base_url, model


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_postgrad_dotenv()
        key, base, model = _llm_env()
        if not key:
            raise RuntimeError(
                "夏令营 LLM 需要 API Key：请设置 OPENAI_API_KEY、RAGFLOW_LLM_API_KEY 或 DEEPSEEK_API_KEY"
            )
        if not base:
            raise RuntimeError(
                "请设置 OPENAI_BASE_URL（DeepSeek 示例：https://api.deepseek.com/v1）"
            )
        return cls(api_key=key, base_url=base.rstrip("/"), model=model)


def call_camp_extract_sync(blob: str, cfg: LLMConfig) -> str:
    url = f"{cfg.base_url}/chat/completions"
    with httpx.Client(timeout=cfg.timeout) as client:
        r = client.post(
            url,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg.model,
                "temperature": 0.15,
                "max_tokens": 2048,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"下面是一条保研经验贴相关文本，请抽取目标院校夏令营及入营情况：\n\n{blob[:24000]}",
                    },
                ],
            },
        )
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()


def blob_digest(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CampLLMCache:
    """按正文摘要缓存 LLM 结果，避免重复计费。"""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._mem: Dict[str, List[Dict[str, Any]]] = {}
        if path and path.is_file():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    d = obj.get("digest")
                    camps = obj.get("camps")
                    if isinstance(d, str) and isinstance(camps, list):
                        self._mem[d] = [x for x in camps if isinstance(x, dict)]

    def get(self, digest: str) -> Optional[List[Dict[str, Any]]]:
        return self._mem.get(digest)

    def set(self, digest: str, camps: List[Dict[str, Any]]) -> None:
        self._mem[digest] = camps

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for digest, camps in sorted(self._mem.items()):
                f.write(
                    json.dumps(
                        {"digest": digest, "camps": camps},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        tmp.replace(self.path)
