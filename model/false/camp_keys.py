"""
从经验贴中启发式解析「院校·学院/夏令营去向」键，用于与 prompt.txt 中
「同一院校夏令营」口径对齐；数据里常无结构化字段，故仅作 best-effort。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# 标题/正文里常见简称 → 规范化前缀（不含学院时后面再补「·未知学院」）
_ALIAS_PREFIX: List[tuple[str, str]] = [
    ("中国人民大学高瓴", "中国人民大学·高瓴人工智能学院"),
    ("人大高瓴", "中国人民大学·高瓴人工智能学院"),
    ("高瓴人工智能学院", "中国人民大学·高瓴人工智能学院"),
    ("清华大学", "清华大学"),
    ("清华", "清华大学"),
    ("北京大学", "北京大学"),
    ("北大", "北京大学"),
    ("浙江大学", "浙江大学"),
    ("浙大", "浙江大学"),
    ("上海交通大学", "上海交通大学"),
    ("上交", "上海交通大学"),
    ("复旦大学", "复旦大学"),
    ("复旦", "复旦大学"),
    ("南京大学", "南京大学"),
    ("南大", "南京大学"),
    ("中国科学技术大学", "中国科学技术大学"),
    ("中科大", "中国科学技术大学"),
    ("中国科学院大学", "中国科学院大学"),
    ("国科大", "中国科学院大学"),
    ("西湖大学", "西湖大学"),
    ("同济大学", "同济大学"),
    ("同济", "同济大学"),
    ("哈尔滨工业大学", "哈尔滨工业大学"),
    ("哈工大", "哈尔滨工业大学"),
    ("华南理工大学", "华南理工大学"),
    ("华工", "华南理工大学"),
    ("东南大学", "东南大学"),
    ("东南", "东南大学"),
    ("天津大学", "天津大学"),
    ("天大", "天津大学"),
    ("北京邮电大学", "北京邮电大学"),
    ("北邮", "北京邮电大学"),
    ("中国人民大学", "中国人民大学"),
    ("人大", "中国人民大学"),
]

# 匹配「XX大学YY学院」整段
_RE_UNIV_COLLEGE = re.compile(r"([\u4e00-\u9fa5]{2,12}大学)([\u4e00-\u9fa5]{2,16}学院)")
# 仅「XX大学」且后面不紧接「学院」
_RE_UNIV_ONLY = re.compile(r"([\u4e00-\u9fa5]{2,12}大学)(?![\u4e00-\u9fa5]{0,2}学院)")

# 误匹配「XX大学」：中间夹了动词/噪声
_BAD_UNIV_SUBSTR = (
    "建议",
    "关于",
    "真的",
    "如何",
    "看待",
    "评价",
    "介绍",
    "分析",
    "参加",
    "值得",
    "哪些",
    "真的建议",
    "院情",
    "考情",
    "内容主要",
    "该帖子",
)
_BAD_UNIV_PREFIX = (
    "上岸",
    "圆梦",
    "保研至",
    "跨保",
    "跨专业",
    "通信工程四非",
    "四非",
    "低rk",
)


def _strip_noise_before_univ(name: str) -> str:
    s = name.strip()
    while s and s[0] in "0123456789０１２３４５６７８９年月届第及及在从对把被":
        s = s[1:].lstrip(" \t，,、")
    for p in _BAD_UNIV_PREFIX:
        if s.startswith(p):
            s = s[len(p) :].lstrip(" \t，,、至到")
    return s.strip()


def _is_plausible_univ(name: str) -> bool:
    if len(name) < 4 or "大学" not in name:
        return False
    if any(b in name for b in _BAD_UNIV_SUBSTR):
        return False
    # 必须以常见高校后缀结尾
    if not (
        name.endswith("大学")
        or name.endswith("学院")
        or "大学·" in name
    ):
        return False
    return True


def _blob(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    camp = row.get("camp_admission")
    if isinstance(camp, dict) and camp.get("detail"):
        parts.append(str(camp["detail"]))
    if row.get("admission_detail"):
        parts.append(str(row["admission_detail"]))
    if row.get("source_title"):
        parts.append(str(row["source_title"]))
    if row.get("notes"):
        parts.append(str(row["notes"]))
    return "\n".join(parts)


def _dedupe_preserve(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def extract_camp_keys_from_text(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    keys: List[str] = []
    for m in _RE_UNIV_COLLEGE.finditer(text):
        u, c = m.group(1), m.group(2)
        u = _strip_noise_before_univ(u)
        if not _is_plausible_univ(u):
            continue
        keys.append(f"{u}·{c}")
    if keys:
        return _dedupe_preserve(keys)
    for m in _RE_UNIV_ONLY.finditer(text):
        u = _strip_noise_before_univ(m.group(1))
        if not _is_plausible_univ(u):
            continue
        keys.append(f"{u}·学院未标注")
    if keys:
        return _dedupe_preserve(keys)
    # 简称：按别名长度降序，避免「南大」误伤「南京大学」已匹配的情况
    t = text
    for alias, full in sorted(_ALIAS_PREFIX, key=lambda x: -len(x[0])):
        if alias in t:
            if "·" in full:
                keys.append(full)
            else:
                keys.append(f"{full}·学院未标注")
            break
    return _dedupe_preserve(keys)


def camp_summer_key_for_row(row: Dict[str, Any]) -> str:
    """
    单条记录一个主键：多校命中时用「; 」连接，便于 print / 后续拆条。
    """
    found = extract_camp_keys_from_text(_blob(row))
    if not found:
        return "未解析·院校夏令营"
    if len(found) == 1:
        return found[0]
    return "; ".join(found[:5])


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="打印启发式「院校·夏令营」键及频次")
    parser.add_argument(
        "--data",
        type=str,
        default="data/baoyan_experience_profiles_20260329_134614.jsonl",
        help="jsonl 或 csv",
    )
    parser.add_argument(
        "--per-row",
        action="store_true",
        help="逐条打印 source_title 片段与解析键",
    )
    args = parser.parse_args()
    path = Path(args.data)
    if not path.is_file():
        raise SystemExit(f"文件不存在: {path.resolve()}")

    rows = load_rows(path)
    counter: Counter[str] = Counter()
    for row in rows:
        counter[camp_summer_key_for_row(row)] += 1

    print(f"=== 院校·夏令营（启发式）键 — 共 {len(counter)} 种，{len(rows)} 条样本 ===\n")
    for key, n in counter.most_common():
        print(f"{n:4d}  {key}")

    if args.per_row:
        print("\n=== 逐条（标题 + 键）===\n")
        for i, row in enumerate(rows, 1):
            title = (row.get("source_title") or "")[:70]
            k = camp_summer_key_for_row(row)
            print(f"[{i:3d}] {k}")
            print(f"      {title!r}")


if __name__ == "__main__":
    main()
