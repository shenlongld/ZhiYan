"""
postgrad_agent 入口与异步集成测试。
覆盖：
- 渠道 A: 官网政策
- 渠道 B: 研招网名额
- 渠道 C: 知乎面经
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from dotenv import load_dotenv
from sqlmodel import Session

try:
    from .extractor import LLMExtractor, to_date
    from .models import (
        ExperienceArchive,
        OfficialPolicy,
        QuotaData,
        SourcePlatform,
        create_db_and_tables,
    )
    from .searcher import TavilySearcher
except ImportError:
    from extractor import LLMExtractor, to_date
    from models import (
        ExperienceArchive,
        OfficialPolicy,
        QuotaData,
        SourcePlatform,
        create_db_and_tables,
    )
    from searcher import TavilySearcher


def _pick_best_text(item: Dict[str, Any]) -> str:
    title = item.get("title", "")
    content = item.get("content", "")
    raw = item.get("raw_content", "")
    url = item.get("url", "")
    return f"标题: {title}\n链接: {url}\n摘要: {content}\n正文: {raw}"


def _with_canonical_page_url(item: Dict[str, Any], body: str) -> str:
    """强制模型把 official_url 对齐 Tavily 本条结果的链接。"""
    u = (item.get("url") or "").strip()
    return f"【权威页面URL】\n{u}\n\n{body}"


def _is_bad_portal_url(url: str) -> bool:
    if not url or not url.startswith("http"):
        return True
    u = url.lower()
    bad = (
        "/user/login",
        "/login",
        "sso.",
        "passport.",
        "/signin",
        "oauth",
    )
    return any(b in u for b in bad)


def _score_official_search_item(item: Dict[str, Any]) -> int:
    """越高越像「浙大计算机学院保研/夏令营」正式通知页。"""
    title = item.get("title") or ""
    content = item.get("content") or ""
    raw = (item.get("raw_content") or "")[:4000]
    url = (item.get("url") or "").lower()
    blob = f"{title}\n{content}\n{raw}"
    b_lower = blob.lower()

    score = 0
    if "cs.zju.edu.cn" in url or "www.cs.zju.edu.cn" in url:
        score += 25
    if "计算机学院" in blob or "计算机科学与技术学院" in blob:
        score += 12
    if "计算机科学与技术" in blob and "学院" in blob:
        score += 6
    if "夏令营" in blob or "推免" in blob or "推荐免试" in blob:
        score += 8
    if "2026" in blob:
        score += 5
    elif "2025" in blob:
        score += 2

    if "zibs" in b_lower or "国际联合商学院" in blob:
        score -= 25
    if ("商学院" in blob or "open day" in b_lower) and "计算机学院" not in blob:
        score -= 12
    if "软件学院" in blob and "计算机学院" not in blob and "计算机科学与技术学院" not in blob:
        score -= 4

    if _is_bad_portal_url(item.get("url") or ""):
        score -= 30

    return score


def _pick_best_official_item(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise ValueError("empty results")
    ranked = sorted(results, key=_score_official_search_item, reverse=True)
    best = ranked[0]
    logging.info(
        "渠道 A: 选中 Tavily 条目标题=%r score=%s url=%s",
        (best.get("title") or "")[:60],
        _score_official_search_item(best),
        best.get("url"),
    )
    if _score_official_search_item(best) < 0:
        logging.warning("渠道 A: 相关度分数仍偏低，结果可能不是计院夏令营，请检查检索词或 Tavily 索引")
    return best


def _safe_int(v: Any) -> int | None:
    if v in (None, "", "null"):
        return None
    try:
        return int(v)
    except Exception:
        return None


async def _handle_official_channel(
    searcher: TavilySearcher, extractor: LLMExtractor, session: Session
) -> OfficialPolicy | None:
    query = "浙江大学 计算机科学与技术学院 2026 夏令营 推免 通知"
    cs_domains = ["cs.zju.edu.cn", "www.cs.zju.edu.cn"]
    res = await searcher.search(
        query,
        max_results=10,
        search_depth="advanced",
        include_domains=cs_domains,
    )
    results = res.get("results") or []
    if not results:
        logging.info("渠道 A: 限定计院域名无结果，改为全网检索")
        res = await searcher.search(query, max_results=10, search_depth="advanced")
        results = res.get("results") or []

    if not results:
        return None

    top = _pick_best_official_item(results)
    tavily_url = (top.get("url") or "").strip()

    try:
        parsed = await extractor.extract_policy_with_attachments(top)
        logging.info("渠道 A: 已使用网页+附件合并上下文提取政策")
    except Exception as e:
        logging.warning("渠道 A: 附件流程不可用，回退纯网页摘要: %s", e)
        body = _pick_best_text(top)
        parsed = await extractor.extract_policy(
            _with_canonical_page_url(top, body)
        )

    # 溯源链接必须以 Tavily 本条为准，避免模型填成登录页
    payload = {
        "school_college": parsed.get("school_college") or "浙江大学-计算机科学与技术学院",
        "announcement_title": parsed.get("announcement_title")
        or top.get("title", "未命名公告"),
        "start_date": to_date(parsed.get("start_date")),
        "end_date": to_date(parsed.get("end_date")),
        "admission_threshold": parsed.get("admission_threshold") or {},
        "official_url": tavily_url,
        "data_source": "学校官网",
    }
    sc = payload["school_college"] or ""
    if "cs.zju" in tavily_url.lower() and (
        "国际联合商学院" in sc or "ZIBS" in sc or "zibs" in sc.lower()
    ):
        payload["school_college"] = "浙江大学-计算机科学与技术学院"
        logging.info("渠道 A: 链接为计院域名，已修正误识别的学院名称")

    if not payload["official_url"]:
        logging.warning("渠道 A: Tavily 未返回 url，跳过入库")
        return None
    return OfficialPolicy.upsert(session, payload)


async def _handle_quota_channel(
    searcher: TavilySearcher, extractor: LLMExtractor, session: Session
) -> QuotaData | None:
    query = "浙江大学 081200 计算机科学与技术 推免 拟招生 人数 研招网"
    res = await searcher.search(query, max_results=8, search_depth="advanced")
    results = res.get("results") or []
    if not results:
        return None

    parsed: Dict[str, Any] | None = None
    for item in results[:5]:
        try:
            cand = await extractor.extract_quota(_pick_best_text(item))
        except Exception as e:
            logging.warning("渠道 B: 单条解析失败，试下一条: %s", e)
            continue
        parsed = cand
        cur = _safe_int(cand.get("current_year_quota"))
        prev = _safe_int(cand.get("prev_year_quota"))
        if cur is not None or prev is not None:
            logging.info("渠道 B: 在条目中抽到名额数字，url=%s", item.get("url"))
            break

    if parsed is None:
        return None

    payload = {
        "school_name": parsed.get("school_name") or "浙江大学",
        "dept_name": parsed.get("dept_name") or "计算机科学与技术学院",
        "major_full_name": parsed.get("major_full_name") or "(081200)计算机科学与技术",
        "current_year_quota": _safe_int(parsed.get("current_year_quota")),
        "prev_year_quota": _safe_int(parsed.get("prev_year_quota")),
        "is_full_time": bool(parsed.get("is_full_time", True)),
        "last_sync_date": date.today(),
    }
    return QuotaData.upsert(session, payload)


async def _handle_experience_channel(
    searcher: TavilySearcher, extractor: LLMExtractor, session: Session
) -> List[ExperienceArchive]:
    query = "site:zhihu.com 浙江大学 计算机 保研 机考 面经"
    res = await searcher.search(query, max_results=8, search_depth="advanced")
    results = res.get("results") or []
    archives: List[ExperienceArchive] = []

    # 要求“3 条以上深度面经”，从前 5 条里尽量提 3 条
    for item in results[:5]:
        parsed = await extractor.extract_experience(_pick_best_text(item))
        payload = {
            "school_major": parsed.get("school_major") or "浙江大学-计算机相关",
            "source_platform": SourcePlatform.ZHIHU,
            "assessment_flow": parsed.get("assessment_flow") or "机考+面试",
            "question_bank": parsed.get("question_bank") or {},
            "mentor_tags": parsed.get("mentor_tags") or [],
            "experience_year": _safe_int(parsed.get("experience_year")),
            "original_post_url": item.get("url"),
            "summary_digest": parsed.get("summary_digest") or "",
        }
        if not payload["original_post_url"]:
            continue
        archives.append(ExperienceArchive.upsert(session, payload))
        if len(archives) >= 3:
            break

    return archives


async def run_test():
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    # 兼容：与 backend 相同的 RAGFLOW_LLM_* 命名
    openai_api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("RAGFLOW_LLM_API_KEY", "").strip()
    )
    openai_base_url = (
        os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("RAGFLOW_LLM_BASE_URL", "").strip()
        or None
    )
    openai_model = (
        os.getenv("OPENAI_MODEL", "").strip()
        or os.getenv("RAGFLOW_LLM_MODEL", "").strip()
        or "gpt-4o-mini"
    )
    temperature = float(
        os.getenv("OPENAI_TEMPERATURE", "").strip()
        or os.getenv("RAGFLOW_LLM_TEMPERATURE", "").strip()
        or "0.1"
    )
    max_tokens = int(
        os.getenv("OPENAI_MAX_TOKENS", "").strip()
        or os.getenv("RAGFLOW_LLM_MAX_TOKENS", "").strip()
        or "4096"
    )

    if not tavily_api_key:
        raise RuntimeError("缺少 TAVILY_API_KEY，请在 .env 中配置")
    if not openai_api_key:
        raise RuntimeError(
            "缺少 LLM 鉴权密钥：请配置 OPENAI_API_KEY 或 RAGFLOW_LLM_API_KEY（火山控制台 API Key，不是 endpoint id）"
        )

    engine = create_db_and_tables(os.getenv("DATABASE_URL", "sqlite:///./postgrad_agent.db"))
    searcher = TavilySearcher(api_key=tavily_api_key)
    extractor = LLMExtractor(
        api_key=openai_api_key,
        model=openai_model,
        base_url=openai_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # expire_on_commit=False：commit/refresh 后仍可在 Session 外安全读取属性（避免 DetachedInstanceError）
    with Session(engine, expire_on_commit=False) as session:
        official_obj = await _handle_official_channel(searcher, extractor, session)
        quota_obj = await _handle_quota_channel(searcher, extractor, session)
        exp_objs = await _handle_experience_channel(searcher, extractor, session)
        # 会话内快照，打印不触发懒加载
        official_dump = official_obj.model_dump() if official_obj else None
        quota_dump = quota_obj.model_dump() if quota_obj else None
        exp_rows = [o.model_dump() for o in exp_objs]

    print("=== run_test 完成 ===")
    if official_dump is not None:
        print(
            "A 官网政策(JSON，已尽量合并附件解析):\n"
            + json.dumps(official_dump, ensure_ascii=False, indent=2, default=str)
        )
    else:
        print("A 官网政策: 无结果")
    print("B 研招名额:", quota_dump if quota_dump is not None else "无结果")
    print("C 知乎面经数量:", len(exp_rows))
    for i, row in enumerate(exp_rows, 1):
        print(f"  [{i}] {row.get('school_major')} | {row.get('original_post_url')}")


if __name__ == "__main__":
    asyncio.run(run_test())
