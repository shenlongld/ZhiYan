"""
批量抓取并结构化“保研经验贴”，落地到本地 JSONL/CSV（不依赖 SQL）。

用法示例：
  cd /home/slld/Desktop/ZhiYan/postgrad_agent
  python collect_experience_local.py --target-count 150
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

try:
    from dotenv import load_dotenv
except ImportError:
    # 允许在未安装 python-dotenv 的环境下运行；
    # 此时依赖外部已注入的环境变量。
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

try:
    from .extractor import LLMExtractor
    from .searcher import TavilySearcher
except ImportError:
    from extractor import LLMExtractor
    from searcher import TavilySearcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

try:
    from .sparse_experience import (
        build_school_college_key,
        extract_major_hits,
        is_target_major_item,
        merge_competitions,
        merge_research,
        normalize_competition_achievements,
        normalize_text,
        sparse_extract_competitions,
        sparse_extract_rank_fields,
        sparse_extract_research,
    )
except ImportError:
    from sparse_experience import (
        build_school_college_key,
        extract_major_hits,
        is_target_major_item,
        merge_competitions,
        merge_research,
        normalize_competition_achievements,
        normalize_text,
        sparse_extract_competitions,
        sparse_extract_rank_fields,
        sparse_extract_research,
    )


ALLOWED_TIERS = ["清北", "华五", "c9", "顶9", "中九", "次九", "末九", "其他", "未知"]
ALLOWED_PROGRAM = ["专硕", "学硕", "直博", "未说明"]

PROMPT_EXPERIENCE_PROFILE = f"""Role: 你是保研经验贴结构化抽取专家。
Task: 从输入的帖子内容中，抽取“博主画像 + 入营结果”。
注意：请优先关注计算机相关方向（计算机/人工智能/电子信息/软件工程等）。
注意：请优先、尽可能完整提取“成绩/排名”线索（如前x%、rank x、x/y、GPA）。

请严格输出一个 JSON 对象，字段如下：
- blogger_school: 字符串，可为 null
- target_major: 字符串（如“计算机科学与技术/人工智能/电子信息/软件工程”等），可为 null
- school_tier: 枚举之一 {ALLOWED_TIERS}
- rank_text: 字符串（例如“专业前5%/1/200”），可为 null
- rank_percent: 数字，百分比值（如 5.0），未知则 null
- rank_num: 整数，未知则 null
- rank_total: 整数，未知则 null
- research_achievements: 数组，元素对象包含：
  - type: 字符串（论文/专利/项目/科研经历等）
  - title: 字符串，可为 null
  - author_order: 字符串（如“一作/二作/共同一作/通讯/未知”）
  - venue_or_level: 字符串（期刊/会议/分区/级别），可为 null
- competition_achievements: 数组，元素对象包含：
  - name: 字符串
  - level: 字符串（国赛/省赛/校赛/国际等），可为 null
  - award: 字符串（金奖/一等奖等），可为 null
  - normalized_name: 字符串（标准赛项名），可为 null
- camp_admission:
  - admitted: 布尔或 null（是否入营）
  - program_type: 枚举之一 {ALLOWED_PROGRAM}
  - detail: 字符串（如“拿到夏令营 offer / 预推免入营 / 未通过”），可为 null
- source_year: 整数，可为 null
- confidence: 0-1 的数字
- notes: 字符串，可为 null

规则：
1) 只能依据输入内容，禁止臆测。
2) 没有出现就填 null 或空数组。
3) 必须是有效 JSON，不要输出解释文字，不要 markdown 代码块。
"""


def detect_platform(url: str) -> str:
    u = (url or "").lower()
    if "zhihu.com" in u:
        return "知乎"
    if "xiaohongshu.com" in u:
        return "小红书"
    if "csdn.net" in u:
        return "CSDN"
    if "cnblogs.com" in u:
        return "博客园"
    return "其他"


def safe_int(v: Any) -> Optional[int]:
    if v in (None, "", "null"):
        return None
    try:
        return int(v)
    except Exception:
        return None


def safe_float(v: Any) -> Optional[float]:
    if v in (None, "", "null"):
        return None
    try:
        return float(v)
    except Exception:
        return None


def norm_tier(v: Any) -> str:
    s = (str(v).strip() if v is not None else "")
    return s if s in ALLOWED_TIERS else "未知"


def norm_program(v: Any) -> str:
    s = (str(v).strip() if v is not None else "")
    return s if s in ALLOWED_PROGRAM else "未说明"


def as_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def as_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def has_core_info(d: Dict[str, Any]) -> bool:
    research = d.get("research_achievements") or []
    comp = d.get("competition_achievements") or []
    rank_text = d.get("rank_text")
    admission = as_dict(d.get("camp_admission"))
    admitted = admission.get("admitted")
    program = admission.get("program_type")
    return bool(
        rank_text
        or research
        or comp
        or admitted is not None
        or (program and program != "未说明")
    )


def flatten_csv_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    admission = rec.get("camp_admission") or {}
    return {
        "source_url": rec.get("source_url"),
        "source_title": rec.get("source_title"),
        "platform": rec.get("platform"),
        "source_year": rec.get("source_year"),
        "blogger_school": rec.get("blogger_school"),
        "target_major": rec.get("target_major"),
        "school_college": rec.get("school_college"),
        "target_major_hits": ",".join(rec.get("target_major_hits") or []),
        "school_tier": rec.get("school_tier"),
        "rank_text": rec.get("rank_text"),
        "rank_percent": rec.get("rank_percent"),
        "rank_num": rec.get("rank_num"),
        "rank_total": rec.get("rank_total"),
        "camp_admitted": admission.get("admitted"),
        "program_type": admission.get("program_type"),
        "admission_detail": admission.get("detail"),
        "research_achievements_json": json.dumps(
            rec.get("research_achievements") or [], ensure_ascii=False
        ),
        "competition_achievements_json": json.dumps(
            rec.get("competition_achievements") or [], ensure_ascii=False
        ),
        "competition_tags": ",".join(rec.get("competition_tags") or []),
        "confidence": rec.get("confidence"),
        "notes": rec.get("notes"),
    }


def build_content(item: Dict[str, Any], *, max_raw_chars: int = 18000) -> str:
    title = item.get("title", "")
    url = item.get("url", "")
    snippet = item.get("content", "")
    raw = item.get("raw_content", "") or ""
    raw = raw[:max_raw_chars]
    return f"标题: {title}\n链接: {url}\n摘要: {snippet}\n正文:\n{raw}"


def _is_likely_image_url(url: str) -> bool:
    u = (url or "").lower()
    if not u.startswith("http"):
        return False
    return any(ext in u for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"])


def extract_image_urls(item: Dict[str, Any], *, max_images: int = 4) -> List[str]:
    page_url = item.get("url", "") or ""
    raw = item.get("raw_content", "") or ""
    content = item.get("content", "") or ""
    blob = f"{raw}\n{content}"
    urls: List[str] = []

    # HTML img src
    for m in re.findall(r"""<img[^>]+src=["']([^"']+)["']""", blob, flags=re.IGNORECASE):
        u = urljoin(page_url, m.strip())
        if _is_likely_image_url(u):
            urls.append(u)

    # Markdown image
    for m in re.findall(r"""!\[[^\]]*\]\(([^)]+)\)""", blob):
        u = urljoin(page_url, m.strip())
        if _is_likely_image_url(u):
            urls.append(u)

    # Bare image links
    for m in re.findall(r"""https?://[^\s)"']+\.(?:jpg|jpeg|png|webp|gif|bmp)""", blob, flags=re.IGNORECASE):
        u = m.strip()
        if _is_likely_image_url(u):
            urls.append(u)

    # 去重保序
    seen: set[str] = set()
    out: List[str] = []
    for u in urls:
        if u in seen:
            continue
        # 简单过滤 data URI 与明显无效地址
        pu = urlparse(u)
        if pu.scheme not in ("http", "https"):
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= max_images:
            break
    return out


def is_info_weak(rec: Dict[str, Any]) -> bool:
    has_rank = bool(rec.get("rank_text") or rec.get("rank_percent") or rec.get("rank_num"))
    has_research = bool(rec.get("research_achievements"))
    has_comp = bool(rec.get("competition_achievements"))
    admission = as_dict(rec.get("camp_admission"))
    has_admission = admission.get("admitted") is not None or (
        admission.get("program_type") and admission.get("program_type") != "未说明"
    )
    return not (has_rank and (has_research or has_comp) and has_admission)


async def extract_one_with_vision(
    item: Dict[str, Any],
    extractor: LLMExtractor,
    *,
    vision_model: str,
    max_vision_images: int,
) -> Optional[Dict[str, Any]]:
    image_urls = extract_image_urls(item, max_images=max_vision_images)
    if not image_urls:
        return None

    prompt = (
        "你是保研经验贴结构化提取器。请结合图片内容抽取 JSON，字段与文本抽取一致："
        "blogger_school,target_major,school_tier,rank_text,rank_percent,rank_num,rank_total,"
        "research_achievements,competition_achievements,camp_admission,source_year,confidence,notes。"
        "重点：优先提取成绩/排名（前x%、rank、x/y、GPA）以及竞赛级别。"
        "仅输出一个合法 JSON 对象。"
    )
    text_hint = (
        f"标题: {item.get('title','')}\n链接: {item.get('url','')}\n"
        f"摘要: {item.get('content','')}"
    )
    user_parts: List[Dict[str, Any]] = [{"type": "text", "text": text_hint}]
    for u in image_urls:
        user_parts.append({"type": "image_url", "image_url": {"url": u}})

    resp = await extractor.client.chat.completions.create(
        model=vision_model,
        temperature=0.1,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_parts},
        ],
    )
    raw = resp.choices[0].message.content or ""
    return await extractor.parse_json_with_retry(raw)


@dataclass
class CollectorConfig:
    target_count: int = 200
    per_query_results: int = 50
    max_extract_items: int = 300
    max_queries: int = 50
    min_confidence: float = 0.20
    search_timeout_sec: float = 30.0
    enable_vision_fallback: bool = False
    enable_sparse_match: bool = True
    include_school_queries: bool = False
    max_vision_images: int = 4
    vision_model: str = ""
    out_dir: str = str(DEFAULT_DATA_DIR)


def build_queries(include_school_queries: bool = False) -> List[str]:
    schools = [
        "清华",
        "北大",
        "浙大",
        "复旦",
        "上交",
        "南大",
        "中科大",
        "哈工大",
        "西安交大",
        "北航",
        "同济",
        "东南",
        "武大",
        "华科",
        "中大",
        "厦大",
        "天大",
        "南开",
    ]
    major_expr = "计算机 OR 人工智能 OR 电子信息 OR 软件工程 OR 网安 OR 数据科学"
    platforms = ["site:zhihu.com"]
    intents = [
        "保研 经验贴 入营 直博 学硕 专硕 排名 科研 竞赛",
        "预推免 夏令营 保研 面经 排名 一作 国奖",
        "保研 经验贴 论文 一作 CCF SCI EI",
        "保研 经验贴 美赛 数学建模 ICPC 蓝桥杯 挑战杯",
        "保研 经验贴 offer 优营 入营资格",
        "保研 经验贴 无科研 无竞赛 rank",
        "保研 经验贴 机试 面试 笔试 真题",
        "保研 经验贴 电子信息 人工智能 计算机",
    ]
    years = ["2026", "2025", "2024", "2023"]
    base_queries: List[str] = []

    for p in platforms:
        for it in intents:
            base_queries.append(f"{p} ({major_expr}) {it}")

    for y in years:
        base_queries.append(f"site:zhihu.com ({major_expr}) {y}届 保研 经验贴")
        base_queries.append(f"site:zhihu.com ({major_expr}) {y}年 保研 夏令营 预推免 面经")

    base_queries.extend(
        [
            f"site:zhihu.com ({major_expr}) 保研 CCF-A CCF-B CCF-C 一作",
            f"site:zhihu.com ({major_expr}) 保研 SCI EI 论文 作者位次",
            f"site:zhihu.com ({major_expr}) 保研 美赛 MCM ICM 数模 国赛",
            f"site:zhihu.com ({major_expr}) 保研 ICPC CCPC 蓝桥杯 挑战杯 互联网+",
        ]
    )

    dedup_base: List[str] = []
    seen = set()
    for q in base_queries:
        if q in seen:
            continue
        seen.add(q)
        dedup_base.append(q)
    base_queries = dedup_base
    if not include_school_queries:
        return base_queries
    school_queries = [
        f"site:zhihu.com {s} ({major_expr}) 保研 经验贴 入营 排名 科研 竞赛" for s in schools
    ]
    return base_queries + school_queries


async def collect_candidates(searcher: TavilySearcher, cfg: CollectorConfig) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    queries = build_queries(include_school_queries=cfg.include_school_queries)
    if cfg.max_queries > 0:
        queries = queries[: cfg.max_queries]
    total_q = len(queries)
    for i, q in enumerate(queries, 1):
        print(f"   检索 query {i}/{total_q}: {q[:80]}...")
        try:
            res = await searcher.search(
                q,
                max_results=cfg.per_query_results,
                search_depth="advanced",
                timeout_sec=cfg.search_timeout_sec,
            )
        except Exception as e:
            print(f"   - query {i} 失败，已跳过: {e}")
            continue
        for item in res.get("results") or []:
            url = (item.get("url") or "").strip()
            if not url or not url.startswith("http"):
                continue
            if not is_target_major_item(item):
                continue
            if url in unique:
                continue
            unique[url] = item
    return list(unique.values())


async def extract_one(
    item: Dict[str, Any],
    extractor: LLMExtractor,
    *,
    enable_vision_fallback: bool = False,
    enable_sparse_match: bool = True,
    vision_model: str = "",
    max_vision_images: int = 4,
) -> Optional[Dict[str, Any]]:
    content = build_content(item)
    raw = await extractor._chat(PROMPT_EXPERIENCE_PROFILE, content, max_user_chars=24000)
    parsed = as_dict(await extractor.parse_json_with_retry(raw))
    parsed_adm = as_dict(parsed.get("camp_admission"))

    rec = {
        "source_url": item.get("url"),
        "source_title": item.get("title"),
        "platform": detect_platform(item.get("url") or ""),
        "blogger_school": parsed.get("blogger_school"),
        "target_major": parsed.get("target_major"),
        "target_major_hits": extract_major_hits(content),
        "school_tier": norm_tier(parsed.get("school_tier")),
        "rank_text": parsed.get("rank_text"),
        "rank_percent": safe_float(parsed.get("rank_percent")),
        "rank_num": safe_int(parsed.get("rank_num")),
        "rank_total": safe_int(parsed.get("rank_total")),
        "research_achievements": as_list(parsed.get("research_achievements")),
        "competition_achievements": normalize_competition_achievements(
            parsed.get("competition_achievements")
        ),
        "camp_admission": {
            "admitted": parsed_adm.get("admitted"),
            "program_type": norm_program(parsed_adm.get("program_type")),
            "detail": parsed_adm.get("detail"),
        },
        "source_year": safe_int(parsed.get("source_year")),
        "confidence": safe_float(parsed.get("confidence")),
        "notes": parsed.get("notes"),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    rec["competition_tags"] = sorted(
        {
            x.get("normalized_name")
            for x in rec["competition_achievements"]
            if isinstance(x, dict) and x.get("normalized_name")
        }
    )

    if enable_sparse_match:
        adm = as_dict(rec.get("camp_admission"))
        sparse_blob = "\n".join(
            [
                normalize_text(item.get("title")),
                normalize_text(item.get("content")),
                normalize_text(item.get("raw_content"))[:12000],
                normalize_text(rec.get("notes")),
                normalize_text(adm.get("detail")),
                normalize_text(rec.get("source_title")),
            ]
        )
        sparse_comp = sparse_extract_competitions(sparse_blob)
        sparse_research = sparse_extract_research(sparse_blob)
        sparse_rank = sparse_extract_rank_fields(sparse_blob)
        rec["competition_achievements"] = merge_competitions(
            rec.get("competition_achievements") or [], sparse_comp
        )
        rec["research_achievements"] = merge_research(
            rec.get("research_achievements") or [], sparse_research
        )
        if not rec.get("rank_text") and sparse_rank.get("rank_text"):
            rec["rank_text"] = sparse_rank.get("rank_text")
        if rec.get("rank_percent") is None and sparse_rank.get("rank_percent") is not None:
            rec["rank_percent"] = sparse_rank.get("rank_percent")
        if rec.get("rank_num") is None and sparse_rank.get("rank_num") is not None:
            rec["rank_num"] = sparse_rank.get("rank_num")
        if rec.get("rank_total") is None and sparse_rank.get("rank_total") is not None:
            rec["rank_total"] = sparse_rank.get("rank_total")

        rec["competition_tags"] = sorted(
            {
                x.get("normalized_name")
                for x in rec["competition_achievements"]
                if isinstance(x, dict) and x.get("normalized_name")
            }
        )

    need_rank_rescue = (
        rec.get("rank_text") in (None, "")
        and rec.get("rank_percent") is None
        and rec.get("rank_num") is None
    )
    if enable_vision_fallback and (is_info_weak(rec) or need_rank_rescue):
        try:
            v_model = vision_model or extractor.model
            parsed_v = await extract_one_with_vision(
                item,
                extractor,
                vision_model=v_model,
                max_vision_images=max_vision_images,
            )
            parsed_v = as_dict(parsed_v)
            if parsed_v:
                rec["blogger_school"] = parsed_v.get("blogger_school") or rec["blogger_school"]
                rec["target_major"] = parsed_v.get("target_major") or rec["target_major"]
                rec["school_tier"] = norm_tier(parsed_v.get("school_tier") or rec["school_tier"])
                rec["rank_text"] = parsed_v.get("rank_text") or rec["rank_text"]
                rec["rank_percent"] = safe_float(parsed_v.get("rank_percent")) or rec["rank_percent"]
                rec["rank_num"] = safe_int(parsed_v.get("rank_num")) or rec["rank_num"]
                rec["rank_total"] = safe_int(parsed_v.get("rank_total")) or rec["rank_total"]
                v_research = as_list(parsed_v.get("research_achievements"))
                if v_research:
                    rec["research_achievements"] = v_research
                v_comp = normalize_competition_achievements(
                    parsed_v.get("competition_achievements")
                )
                if v_comp:
                    rec["competition_achievements"] = v_comp
                v_adm = as_dict(parsed_v.get("camp_admission"))
                rec["camp_admission"] = {
                    "admitted": v_adm.get("admitted"),
                    "program_type": norm_program(v_adm.get("program_type")),
                    "detail": v_adm.get("detail"),
                }
                rec["source_year"] = safe_int(parsed_v.get("source_year")) or rec["source_year"]
                rec["confidence"] = safe_float(parsed_v.get("confidence")) or rec["confidence"]
                rec["notes"] = parsed_v.get("notes") or rec["notes"]
                rec["competition_tags"] = sorted(
                    {
                        x.get("normalized_name")
                        for x in rec["competition_achievements"]
                        if isinstance(x, dict) and x.get("normalized_name")
                    }
                )
        except Exception:
            pass

    rec["school_college"] = build_school_college_key(
        rec.get("blogger_school"), rec.get("target_major")
    )

    if not has_core_info(rec):
        return None
    return rec


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [flatten_csv_row(r) for r in rows]
    if not flat:
        headers = [
            "source_url",
            "source_title",
            "platform",
            "source_year",
            "blogger_school",
            "target_major",
            "school_college",
            "school_tier",
            "rank_text",
            "rank_percent",
            "rank_num",
            "rank_total",
            "camp_admitted",
            "program_type",
            "admission_detail",
            "research_achievements_json",
            "competition_achievements_json",
            "competition_tags",
            "confidence",
            "notes",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


async def run(
    cfg: CollectorConfig,
    *,
    tavily_api_key_arg: Optional[str] = None,
    llm_api_key_arg: Optional[str] = None,
    llm_base_url_arg: Optional[str] = None,
    llm_model_arg: Optional[str] = None,
) -> None:
    load_dotenv()

    tavily_api_key = (
        (tavily_api_key_arg or "").strip() or (os.getenv("TAVILY_API_KEY") or "").strip()
    )
    llm_api_key = (
        (llm_api_key_arg or "").strip()
        or (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("RAGFLOW_LLM_API_KEY") or "").strip()
    )
    llm_base_url = (
        (llm_base_url_arg or "").strip()
        or (os.getenv("OPENAI_BASE_URL") or "").strip()
        or (os.getenv("RAGFLOW_LLM_BASE_URL") or "").strip()
        or None
    )
    llm_model = (
        (llm_model_arg or "").strip()
        or (os.getenv("OPENAI_MODEL") or "").strip()
        or (os.getenv("RAGFLOW_LLM_MODEL") or "").strip()
        or "gpt-4o-mini"
    )

    if not tavily_api_key:
        raise RuntimeError("缺少 TAVILY_API_KEY，请在 .env 中配置或用 --tavily-api-key 传入")
    if not llm_api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY 或 RAGFLOW_LLM_API_KEY")

    searcher = TavilySearcher(api_key=tavily_api_key)
    extractor = LLMExtractor(
        api_key=llm_api_key,
        model=llm_model,
        base_url=llm_base_url,
        temperature=0.1,
        max_tokens=4096,
    )

    print("1) 正在检索候选帖子...")
    candidates = await collect_candidates(searcher, cfg)
    print(f"   检索完成，候选 URL 数量: {len(candidates)}")

    # 尽量优先处理“经验/面经/保研”相关标题
    def rank_key(x: Dict[str, Any]) -> int:
        t = ((x.get("title") or "") + " " + (x.get("content") or "")).lower()
        score = 0
        for kw in [
            "保研",
            "经验",
            "面经",
            "夏令营",
            "预推免",
            "入营",
            "直博",
            "学硕",
            "专硕",
            "美赛",
            "数学建模",
            "icpc",
            "ccpc",
            "蓝桥杯",
            "挑战杯",
            "互联网+",
            "国奖",
            "ccf",
            "sci",
            "ei",
            "一作",
            "共同一作",
            "top",
        ]:
            if kw in t:
                score += 1
        return score

    candidates = sorted(candidates, key=rank_key, reverse=True)[: cfg.max_extract_items]

    rows: List[Dict[str, Any]] = []
    seen_url: set[str] = set()
    for idx, item in enumerate(candidates, 1):
        url = (item.get("url") or "").strip()
        if not url or url in seen_url:
            continue
        seen_url.add(url)

        try:
            rec = await extract_one(
                item,
                extractor,
                enable_vision_fallback=cfg.enable_vision_fallback,
                enable_sparse_match=cfg.enable_sparse_match,
                vision_model=cfg.vision_model or llm_model,
                max_vision_images=cfg.max_vision_images,
            )
            if not rec:
                continue
            confidence = rec.get("confidence")
            if confidence is not None and confidence < cfg.min_confidence:
                continue
            rows.append(rec)
        except Exception as e:
            print(f"   - 跳过({idx}): {url} | 错误: {e}")
            continue

        if idx % 10 == 0:
            print(f"   已处理 {idx}/{len(candidates)}，已结构化 {len(rows)} 条")
        if len(rows) >= cfg.target_count:
            break

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir)
    jsonl_path = out_dir / f"baoyan_experience_profiles_{ts}.jsonl"
    csv_path = out_dir / f"baoyan_experience_profiles_{ts}.csv"
    save_jsonl(rows, jsonl_path)
    save_csv(rows, csv_path)

    print("\n=== 完成 ===")
    print(f"结构化总条数: {len(rows)}")
    print(f"JSONL: {jsonl_path.resolve()}")
    print(f"CSV  : {csv_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量采集保研经验贴并本地结构化落盘")
    parser.add_argument("--target-count", type=int, default=150, help="目标条数，建议 100-200")
    parser.add_argument("--per-query-results", type=int, default=20, help="每个检索词返回条数")
    parser.add_argument("--max-extract-items", type=int, default=300, help="最多抽取的候选条目数")
    parser.add_argument("--max-queries", type=int, default=30, help="最多执行多少条检索 query（默认30）")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="最低保留置信度")
    parser.add_argument(
        "--search-timeout-sec",
        type=float,
        default=30.0,
        help="单次检索超时秒数，超时会自动跳过该 query",
    )
    parser.add_argument(
        "--enable-vision-fallback",
        action="store_true",
        help="文本抽取信息不足时，尝试基于帖子图片做多模态补抽",
    )
    parser.add_argument(
        "--include-school-queries",
        action="store_true",
        help="额外启用按学校名称检索（默认关闭，走全网计算机相关帖子）",
    )
    parser.add_argument(
        "--disable-sparse-match",
        action="store_true",
        help="关闭稀疏匹配（默认开启：关键词+正则补抽论文/竞赛）",
    )
    parser.add_argument(
        "--max-vision-images",
        type=int,
        default=4,
        help="每条帖子最多传入多少张图片",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default="",
        help="可选：多模态模型名；默认与 --llm-model 一致",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="输出目录（默认: ZhiYan/data）",
    )
    parser.add_argument("--tavily-api-key", type=str, default="", help="可选：直接传 Tavily Key")
    parser.add_argument("--llm-api-key", type=str, default="", help="可选：直接传 LLM Key")
    parser.add_argument("--llm-base-url", type=str, default="", help="可选：直接传 LLM Base URL")
    parser.add_argument("--llm-model", type=str, default="", help="可选：直接传 LLM 模型名")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = CollectorConfig(
        target_count=args.target_count,
        per_query_results=args.per_query_results,
        max_extract_items=args.max_extract_items,
        max_queries=args.max_queries,
        min_confidence=args.min_confidence,
        search_timeout_sec=args.search_timeout_sec,
        enable_vision_fallback=args.enable_vision_fallback,
        enable_sparse_match=(not args.disable_sparse_match),
        include_school_queries=args.include_school_queries,
        max_vision_images=args.max_vision_images,
        vision_model=args.vision_model,
        out_dir=args.out_dir,
    )
    asyncio.run(
        run(
            cfg,
            tavily_api_key_arg=args.tavily_api_key,
            llm_api_key_arg=args.llm_api_key,
            llm_base_url_arg=args.llm_base_url,
            llm_model_arg=args.llm_model,
        )
    )
