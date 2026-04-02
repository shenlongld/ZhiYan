"""
把采集结果转换为深度学习友好的极简结构和多热向量。

输出：
1) model/result/compact.jsonl
   每行: {"id": "...", "成绩": "...", "比赛": [...], "论文": [...], "入营情况": {"学校-院系": true/false}}
2) model/result/vocab.json
   {"competition_vocab":[...], "journal_vocab":[...], "school_vocab":[...]}
3) model/result/vectors.jsonl
   每行: {"id":"...", "competition_vector":[...], "journal_vector":[...], "school_vector":[...]}
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_DIR = PROJECT_ROOT / "model" / "result"


SCHOOL_PATTERNS = [
    "清华大学",
    "北京大学",
    "浙江大学",
    "复旦大学",
    "上海交通大学",
    "南京大学",
    "中国科学技术大学",
    "哈尔滨工业大学",
    "西安交通大学",
    "北京航空航天大学",
    "同济大学",
    "东南大学",
    "武汉大学",
    "华中科技大学",
    "中山大学",
    "厦门大学",
    "天津大学",
    "南开大学",
]

SCHOOL_ALIASES = {
    "清华": "清华大学",
    "北大": "北京大学",
    "浙大": "浙江大学",
    "复旦": "复旦大学",
    "上交": "上海交通大学",
    "南大": "南京大学",
    "中科大": "中国科学技术大学",
    "哈工大": "哈尔滨工业大学",
    "西交": "西安交通大学",
    "北航": "北京航空航天大学",
    "同济": "同济大学",
    "东南": "东南大学",
    "武大": "武汉大学",
    "华科": "华中科技大学",
    "中大": "中山大学",
    "厦大": "厦门大学",
    "天大": "天津大学",
    "南开": "南开大学",
}

DEPT_KEYWORDS = [
    "计算机学院",
    "软件学院",
    "人工智能学院",
    "智能科学与技术学院",
    "网络空间安全学院",
    "电子信息学院",
    "信息工程学院",
    "自动化学院",
    "计算机系",
    "软件工程系",
    "人工智能系",
    "电子信息系",
    "计算机科学与技术",
    "软件工程",
    "人工智能",
    "电子信息",
    "网络空间安全",
    "信息与通信工程",
    "控制科学与工程",
]


def latest_input_file(data_dir: Path) -> Path:
    files = sorted(data_dir.glob("baoyan_experience_profiles_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"未找到采集文件: {data_dir}/baoyan_experience_profiles_*.jsonl")
    return files[0]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows


def normalize_text(v: Any) -> str:
    return (str(v).strip() if v is not None else "")


def extract_competitions(rec: Dict[str, Any]) -> tuple[List[str], List[str]]:
    items = rec.get("competition_achievements") or []
    display_out: List[str] = []
    tag_out: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = normalize_text(it.get("normalized_name")) or normalize_text(it.get("name"))
        level = normalize_text(it.get("level"))
        award = normalize_text(it.get("award"))
        if name:
            tag_out.append(name)
            if level or award:
                display_out.append(" | ".join(x for x in [name, level, award] if x))
            else:
                display_out.append(name)

    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return dedup(display_out), dedup(tag_out)


def extract_papers_and_journals(rec: Dict[str, Any]) -> tuple[List[str], List[str]]:
    research = rec.get("research_achievements") or []
    papers: List[str] = []
    journals: List[str] = []
    for it in research:
        if not isinstance(it, dict):
            continue
        typ = normalize_text(it.get("type"))
        title = normalize_text(it.get("title"))
        venue = normalize_text(it.get("venue_or_level"))
        author_order = normalize_text(it.get("author_order"))

        is_paper = ("论文" in typ) or ("CCF" in venue.upper()) or ("SCI" in venue.upper()) or ("EI" in venue.upper())
        if is_paper:
            paper_desc = " | ".join(x for x in [title or "未命名论文", author_order or "作者位次未知", venue or "期刊/会议未知"] if x)
            papers.append(paper_desc)

        if venue:
            journals.append(venue)

    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return dedup(papers), dedup(journals)


def _canonical_school_name(text: str) -> Optional[str]:
    for s in SCHOOL_PATTERNS:
        if s in text:
            return s
    for alias, full in SCHOOL_ALIASES.items():
        if alias in text:
            return full
    m = re.search(r"([\u4e00-\u9fa5]{2,12}大学)", text)
    if m:
        return m.group(1)
    return None


def _extract_dept(text: str, target_major: str = "") -> Optional[str]:
    for d in DEPT_KEYWORDS:
        if d in text:
            return d
    tm = normalize_text(target_major)
    if tm:
        for d in DEPT_KEYWORDS:
            if d in tm:
                return d
        if "/" in tm:
            return tm.split("/")[0].strip()
        return tm
    return None


def extract_school_departments(rec: Dict[str, Any]) -> List[str]:
    hits: List[str] = []
    blob = " ".join(
        [
            normalize_text(rec.get("blogger_school")),
            normalize_text(rec.get("source_title")),
            normalize_text((rec.get("camp_admission") or {}).get("detail")),
            normalize_text(rec.get("notes")),
        ]
    )
    school = _canonical_school_name(blob)
    dept = _extract_dept(blob, normalize_text(rec.get("target_major")))
    if school and dept:
        hits.append(f"{school}-{dept}")
    elif school:
        hits.append(f"{school}-未知院系")

    seen = set()
    out = []
    for x in hits:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def bool_contains(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def infer_score_from_blob(rec: Dict[str, Any]) -> Optional[str]:
    blob = " ".join(
        [
            normalize_text(rec.get("rank_text")),
            normalize_text(rec.get("source_title")),
            normalize_text(rec.get("notes")),
            normalize_text((rec.get("camp_admission") or {}).get("detail")),
        ]
    )
    if not blob:
        return None

    patterns = [
        r"前\s*\d+(?:\.\d+)?%",
        r"top\s*\d+(?:\.\d+)?%",
        r"\d+\s*/\s*\d+",
        r"(?:rk|rank)\s*\d+",
        r"专业第[一二三四五六七八九十百0-9]+",
        r"绩点\s*\d(?:\.\d+)?",
        r"gpa\s*\d(?:\.\d+)?",
    ]
    for p in patterns:
        m = re.search(p, blob, flags=re.IGNORECASE)
        if m:
            return m.group(0)

    if bool_contains(blob, ["低rank", "低 rank", "rank低", "末九低rank"]):
        return "低rank(文本线索)"
    if bool_contains(blob, ["高rank", "高 rank", "rank高", "专业第一", "rank1"]):
        return "高rank(文本线索)"
    return None


def build_admission_bool(rec: Dict[str, Any]) -> bool:
    detail = normalize_text((rec.get("camp_admission") or {}).get("detail"))
    admitted = (rec.get("camp_admission") or {}).get("admitted")
    negative = bool_contains(detail, ["未入营", "没入营", "未通过", "拒", "淘汰"])
    positive = bool(admitted) or bool_contains(detail, ["上岸", "offer", "入营", "优营"])
    if negative:
        return False
    return bool(positive)


def build_admission_map(rec: Dict[str, Any], school_depts: List[str]) -> Dict[str, bool]:
    final_admit = build_admission_bool(rec)
    if not school_depts:
        return {"未知学校-未知院系": final_admit}
    return {sd: final_admit for sd in school_depts}


def build_score_text(rec: Dict[str, Any]) -> str:
    rank_text = normalize_text(rec.get("rank_text"))
    rank_num = rec.get("rank_num")
    rank_total = rec.get("rank_total")
    rank_percent = rec.get("rank_percent")
    detail = normalize_text((rec.get("camp_admission") or {}).get("detail"))

    parts: List[str] = []
    if rank_text:
        parts.append(f"成绩:{rank_text}")
    elif rank_num is not None and rank_total is not None:
        parts.append(f"成绩:{rank_num}/{rank_total}")
    elif rank_percent is not None:
        parts.append(f"成绩:前{rank_percent}%")
    else:
        inferred = infer_score_from_blob(rec)
        if inferred:
            parts.append(f"成绩:{inferred}")
        else:
            parts.append("成绩:未知")

    if bool_contains(detail, ["预推免", "预推", "九推"]):
        parts.append("预推免:是")
    else:
        parts.append("预推免:未知")
    return " | ".join(parts)


def multi_hot(vocab_index: Dict[str, int], values: List[str]) -> List[int]:
    vec = [0] * len(vocab_index)
    for v in values:
        idx = vocab_index.get(v)
        if idx is not None:
            vec[idx] = 1
    return vec


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成简化样本与向量化词表")
    parser.add_argument("--input", type=str, default="", help="输入 jsonl，默认读取 data 最新采集文件")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="输出目录")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    in_path = Path(args.input) if args.input else latest_input_file(DEFAULT_INPUT_DIR)
    rows = read_jsonl(in_path)
    if not rows:
        raise RuntimeError(f"输入为空或无法解析: {in_path}")

    compact_rows: List[Dict[str, Any]] = []
    comp_sets: List[List[str]] = []
    journal_sets: List[List[str]] = []
    school_dept_sets: List[List[str]] = []

    for i, rec in enumerate(rows, 1):
        rid = f"r{i:06d}"
        competition_display, competition_tags = extract_competitions(rec)
        papers, journals = extract_papers_and_journals(rec)
        school_depts = extract_school_departments(rec)
        admission_map = build_admission_map(rec, school_depts)
        score_text = build_score_text(rec)

        compact_rows.append(
            {
                "id": rid,
                "成绩": score_text,
                "比赛": competition_display,
                "论文": papers,
                "入营情况": admission_map,
                "学校院系": school_depts,
                # 本科院校层级（与夏令营目标校无关），供 value / train_school 的「院校层级」维使用
                "本科层次": normalize_text(rec.get("school_tier")) or "未知",
            }
        )
        comp_sets.append(competition_tags)
        journal_sets.append(journals)
        school_dept_sets.append(school_depts)

    comp_vocab = sorted({x for xs in comp_sets for x in xs})
    journal_vocab = sorted({x for xs in journal_sets for x in xs})
    school_dept_vocab = sorted({x for xs in school_dept_sets for x in xs})

    comp_index = {v: i for i, v in enumerate(comp_vocab)}
    journal_index = {v: i for i, v in enumerate(journal_vocab)}
    school_dept_index = {v: i for i, v in enumerate(school_dept_vocab)}

    vector_rows: List[Dict[str, Any]] = []
    for row, comps, journals, school_depts in zip(compact_rows, comp_sets, journal_sets, school_dept_sets):
        vector_rows.append(
            {
                "id": row["id"],
                "competition_vector": multi_hot(comp_index, comps),
                "journal_vector": multi_hot(journal_index, journals),
                "school_dept_vector": multi_hot(school_dept_index, school_depts),
            }
        )

    compact_path = data_dir / "compact.jsonl"
    vocab_path = data_dir / "vocab.json"
    vectors_path = data_dir / "vectors.jsonl"

    write_jsonl(compact_path, compact_rows)
    write_jsonl(vectors_path, vector_rows)
    vocab_path.write_text(
        json.dumps(
            {
                "competition_vocab": comp_vocab,
                "journal_vocab": journal_vocab,
                "school_dept_vocab": school_dept_vocab,
                "competition_dim": len(comp_vocab),
                "journal_dim": len(journal_vocab),
                "school_dept_dim": len(school_dept_vocab),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== done ===")
    print(f"input:   {in_path}")
    print(f"records: {len(compact_rows)}")
    print(f"compact: {compact_path}")
    print(f"vectors: {vectors_path}")
    print(f"vocab:   {vocab_path}")


if __name__ == "__main__":
    main()

