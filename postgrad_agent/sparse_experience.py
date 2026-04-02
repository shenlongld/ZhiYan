"""
经验贴稀疏匹配与归一化（纯函数，无 LLM/Tavily 依赖）。
供 collect_experience_local、model.vectorsize 共用。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

TARGET_MAJOR_KEYWORDS = [
    "计算机",
    "计算机科学",
    "计算机技术",
    "人工智能",
    "电子信息",
    "软件工程",
    "智能科学",
    "网络空间安全",
    "数据科学",
    "大数据",
    "信息与计算科学",
    "通信工程",
    "自动化",
    "控制工程",
    "机器人工程",
    "物联网工程",
    "信息安全",
    "密码学",
]
TARGET_COMPETITION_KEYWORDS = [
    "美赛",
    "MCM",
    "ICM",
    "数学建模国赛",
    "全国大学生数学建模竞赛",
    "研究生数学建模竞赛",
    "华为杯研赛建模",
    "ICPC",
    "ACM-ICPC",
    "区域赛",
    "邀请赛",
    "蓝桥杯",
    "蓝桥杯国赛",
    "蓝桥杯省赛",
    "挑战杯",
    "互联网+",
    "中国国际大学生创新大赛",
    "大创",
    "大学生创新创业训练计划",
    "全国大学生电子设计竞赛",
    "电赛",
    "全国大学生计算机系统能力大赛",
    "系统能力大赛",
    "全国大学生计算机设计大赛",
    "PAT",
    "CSP",
    "MathorCup",
    "华数杯",
    "RoboMaster",
    "RoboCup",
    "CCPC",
    "天梯赛",
    "GPLT",
    "计算机设计大赛",
    "服务外包创新创业大赛",
    "ACM",
    "国创",
    "省创",
    "国奖",
]
PAPER_VENUE_KEYWORDS = [
    "SCI一区",
    "SCI二区",
    "SCI",
    "SSCI",
    "EI",
    "中文EI",
    "北大核心",
    "中文核心",
    "CSCD",
    "CCF-A",
    "CCF-B",
    "CCF-C",
    "TPAMI",
    "TIP",
    "TNNLS",
    "TOIS",
    "TMC",
    "TKDE",
    "NeurIPS",
    "ICML",
    "ICLR",
    "AAAI",
    "IJCAI",
    "CVPR",
    "ECCV",
    "ICCV",
    "ACL",
    "NAACL",
    "COLING",
    "EMNLP",
    "KDD",
    "WWW",
    "SIGIR",
    "SIGMOD",
    "VLDB",
]
AUTHOR_ORDER_KEYWORDS = [
    "共同一作",
    "学生一作",
    "一作",
    "二作",
    "三作",
    "通讯作者",
    "共同通讯",
    "co-first",
    "first author",
]


def normalize_text(v: Any) -> str:
    return (str(v).strip() if v is not None else "")


def safe_int(v: Any) -> Optional[int]:
    if v in (None, "", "null"):
        return None
    try:
        return int(v)
    except Exception:
        return None


def build_school_college_key(blogger_school: Any, target_major: Any) -> str:
    """
    本科画像键：学校（或匿名描述）·专业方向。
    与 prompt 中「学校-学院」口径对齐：数据中常无单独学院字段，用博主学校+目标专业聚合。
    """
    school = normalize_text(blogger_school)
    major = normalize_text(target_major)
    if school and major:
        return f"{school}·{major}"
    if school:
        return f"{school}·专业未填"
    if major:
        return f"本科未知·{major}"
    return "本科未知·专业未填"


def extract_major_hits(text: str) -> List[str]:
    if not text:
        return []
    lower = text.lower()
    hits: List[str] = []
    for kw in TARGET_MAJOR_KEYWORDS:
        if kw.lower() in lower:
            hits.append(kw)
    return hits


def is_target_major_item(item: Dict[str, Any]) -> bool:
    title = item.get("title", "") or ""
    content = item.get("content", "") or ""
    raw = item.get("raw_content", "") or ""
    blob = f"{title}\n{content}\n{raw[:3000]}"
    major_hit = len(extract_major_hits(blob)) > 0
    sparse_signal = any(
        kw.lower() in blob.lower() for kw in (TARGET_COMPETITION_KEYWORDS + PAPER_VENUE_KEYWORDS)
    )
    return major_hit or sparse_signal


def normalize_competition_name(name: Any) -> Optional[str]:
    s = (str(name).strip() if name is not None else "")
    if not s:
        return None
    low = s.lower()
    mapping = [
        ("美赛", ["美赛", "mcm", "icm"]),
        ("数学建模国赛", ["数学建模国赛", "全国大学生数学建模竞赛", "高教社杯"]),
        ("研究生数学建模竞赛", ["研究生数学建模竞赛", "华为杯研赛建模"]),
        ("ICPC", ["icpc", "acm-icpc", "acm/icpc"]),
        ("CCPC", ["ccpc"]),
        ("蓝桥杯", ["蓝桥杯"]),
        ("挑战杯", ["挑战杯"]),
        ("互联网+", ["互联网+", "中国国际大学生创新大赛"]),
        ("RoboMaster", ["robomaster"]),
        ("RoboCup", ["robocup"]),
        ("天梯赛", ["天梯赛", "gplt"]),
        ("计算机设计大赛", ["计算机设计大赛"]),
        ("服务外包创新创业大赛", ["服务外包创新创业大赛"]),
    ]
    for std, aliases in mapping:
        if any(a.lower() in low for a in aliases):
            return std
    for kw in TARGET_COMPETITION_KEYWORDS:
        if kw.lower() in low:
            return kw
    return s


def normalize_competition_achievements(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = (it.get("name") or "").strip()
        level = it.get("level")
        award = it.get("award")
        std_name = normalize_competition_name(name)
        row = {
            "name": name or (std_name or ""),
            "level": level,
            "award": award,
            "normalized_name": std_name,
        }
        if row["name"] or row["normalized_name"]:
            normalized.append(row)
    return normalized


def sparse_extract_competitions(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    low = text.lower()
    out: List[Dict[str, Any]] = []
    for kw in TARGET_COMPETITION_KEYWORDS:
        if kw.lower() in low:
            level = None
            if "国际" in text or "world" in low:
                level = "国际"
            elif "国赛" in text or "国家级" in text:
                level = "国赛"
            elif "省赛" in text or "省级" in text:
                level = "省赛"
            elif "校赛" in text:
                level = "校赛"
            elif "区域赛" in text:
                level = "区域赛"
            award = None
            m_award = re.search(
                r"(金奖|银奖|铜奖|一等奖|二等奖|三等奖|特等奖|优秀奖|国家奖学金|国奖|省一|省二|省三|国一|国二|国三|F奖|M奖|H奖)",
                text,
            )
            if m_award:
                award = m_award.group(1)
            out.append(
                {
                    "name": kw,
                    "level": level,
                    "award": award,
                    "normalized_name": normalize_competition_name(kw),
                }
            )
    return normalize_competition_achievements(out)


def sparse_extract_research(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    venues = [v for v in PAPER_VENUE_KEYWORDS if v.lower() in text.lower()]
    if not venues and "论文" not in text:
        return []
    author_order = "未知"
    for a in AUTHOR_ORDER_KEYWORDS:
        if a in text:
            author_order = a
            break
    out: List[Dict[str, Any]] = []
    if venues:
        for v in venues:
            out.append(
                {
                    "type": "论文",
                    "title": None,
                    "author_order": author_order,
                    "venue_or_level": v,
                }
            )
    else:
        out.append(
            {
                "type": "论文",
                "title": None,
                "author_order": author_order,
                "venue_or_level": None,
            }
        )
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for r in out:
        key = (r.get("type"), r.get("author_order"), r.get("venue_or_level"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def _safe_percent(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if x < 0 or x > 100:
        return None
    return x


def sparse_extract_rank_fields(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rank_text": None,
        "rank_percent": None,
        "rank_num": None,
        "rank_total": None,
    }
    if not text:
        return out

    blob = text
    m_pct = re.search(
        r"(前\s*\d+(?:\.\d+)?\s*%|top\s*\d+(?:\.\d+)?\s*%)",
        blob,
        flags=re.IGNORECASE,
    )
    if m_pct:
        out["rank_text"] = m_pct.group(1).replace(" ", "")
        n = re.search(r"\d+(?:\.\d+)?", m_pct.group(1))
        if n:
            out["rank_percent"] = _safe_percent(n.group(0))

    m_frac = re.search(r"(\d+)\s*/\s*(\d+)", blob)
    if m_frac:
        a = safe_int(m_frac.group(1))
        b = safe_int(m_frac.group(2))
        if out["rank_text"] is None:
            out["rank_text"] = f"{m_frac.group(1)}/{m_frac.group(2)}"
        out["rank_num"] = a
        out["rank_total"] = b
        if a and b and b > 0 and out["rank_percent"] is None:
            out["rank_percent"] = round((a / b) * 100, 2)

    m_rk = re.search(r"(rk|rank)\s*[:：]?\s*(\d+)", blob, flags=re.IGNORECASE)
    if m_rk:
        rk = safe_int(m_rk.group(2))
        if out["rank_text"] is None:
            out["rank_text"] = f"rank{m_rk.group(2)}"
        out["rank_num"] = out["rank_num"] if out["rank_num"] is not None else rk

    if out["rank_text"] is None:
        m_gpa = re.search(
            r"(gpa|绩点)\s*[:：]?\s*(\d(?:\.\d+)?)", blob, flags=re.IGNORECASE
        )
        if m_gpa:
            out["rank_text"] = f"{m_gpa.group(1)}{m_gpa.group(2)}"

    low = blob.lower()
    if out["rank_text"] is None:
        if any(k in low for k in ["低rank", "low rank", "rank低"]):
            out["rank_text"] = "低rank"
        elif any(k in low for k in ["高rank", "high rank", "rank高", "专业第一", "rank1"]):
            out["rank_text"] = "高rank"
    return out


def merge_competitions(
    primary: List[Dict[str, Any]], extra: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    merged = normalize_competition_achievements(primary) + normalize_competition_achievements(extra)
    dedup: Dict[str, Dict[str, Any]] = {}
    for it in merged:
        name = it.get("normalized_name") or it.get("name")
        if not name:
            continue
        if name not in dedup:
            dedup[name] = it
            continue
        old = dedup[name]
        old["level"] = old.get("level") or it.get("level")
        old["award"] = old.get("award") or it.get("award")
        old["name"] = old.get("name") or it.get("name")
    return list(dedup.values())


def merge_research(
    primary: List[Dict[str, Any]], extra: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    base = primary[:] if isinstance(primary, list) else []
    for it in extra:
        if not isinstance(it, dict):
            continue
        exists = False
        for b in base:
            if (
                b.get("type") == it.get("type")
                and b.get("author_order") == it.get("author_order")
                and b.get("venue_or_level") == it.get("venue_or_level")
            ):
                exists = True
                break
        if not exists:
            base.append(it)
    return base
