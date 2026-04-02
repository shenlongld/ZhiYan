from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_RESULT_DIR = PROJECT_ROOT / "model" / "result"
DATA_DIR = PROJECT_ROOT / "data"

TIER_SCORE = {
    "清北": 9.0,
    "华五": 8.2,
    "c9": 7.5,
    "985": 7.2,  # 用户输入友好映射
    "顶9": 7.2,
    "中九": 6.5,
    "次九": 5.6,
    "末九": 4.6,
    "其他": 3.2,
    "未知": 2.8,
}


def normalize_text(v: Any) -> str:
    return (str(v).strip() if v is not None else "")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def latest_input_file(data_dir: Path) -> Path:
    files = sorted(data_dir.glob("baoyan_experience_profiles_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"未找到输入文件: {data_dir}/baoyan_experience_profiles_*.jsonl")
    return files[0]


def parse_rank_score_from_rec(rec: Dict[str, Any]) -> Optional[float]:
    rank_percent = rec.get("rank_percent")
    if isinstance(rank_percent, (int, float)):
        rp = float(rank_percent)
        if 0 <= rp <= 100:
            return max(0.0, min(100.0, 100.0 - rp))

    rank_num = rec.get("rank_num")
    rank_total = rec.get("rank_total")
    if isinstance(rank_num, int) and isinstance(rank_total, int) and rank_total > 0 and rank_num >= 1:
        pct = (rank_num - 1) * 100.0 / rank_total
        return max(0.0, min(100.0, 100.0 - pct))

    rank_text = normalize_text(rec.get("rank_text"))
    return parse_rank_score_text(rank_text)


def parse_rank_score_text(rank_text: str) -> Optional[float]:
    t = normalize_text(rank_text)
    if not t:
        return None
    m_frac = re.search(r"(\d+)\s*/\s*(\d+)", t)
    if m_frac:
        a = int(m_frac.group(1))
        b = int(m_frac.group(2))
        if b > 0:
            pct = (a - 1) * 100.0 / b
            return max(0.0, min(100.0, 100.0 - pct))
    m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
    if m_pct:
        rp = float(m_pct.group(1))
        if 0 <= rp <= 100:
            return max(0.0, min(100.0, 100.0 - rp))
    low = t.lower()
    if "低rank" in low or "low rank" in low:
        return 25.0
    if "高rank" in low or "rank1" in low or "专业第一" in t:
        return 90.0
    return None


def build_score_feature(raw_score: Optional[float], score_mean: float, amplify: float = 1.0) -> float:
    base = raw_score if raw_score is not None else score_mean
    return max(0.0, min(100.0, base))


def compute_score_mean() -> float:
    try:
        in_path = latest_input_file(DATA_DIR)
        rows = read_jsonl(in_path)
        scores = [parse_rank_score_from_rec(r) for r in rows]
        vals = [s for s in scores if s is not None]
        if vals:
            return sum(vals) / len(vals)
    except Exception:
        pass
    return 50.0


def parse_items(v: Any) -> List[str]:
    if isinstance(v, list):
        return [normalize_text(x) for x in v if normalize_text(x)]
    s = normalize_text(v)
    if not s:
        return []
    parts = re.split(r"[，,；;、\n]+", s)
    return [normalize_text(x) for x in parts if normalize_text(x)]


def _award_coeff_from_text(text: str) -> float:
    t = normalize_text(text).lower()
    if not t:
        return 1.0
    if any(k in t for k in ["国一", "一等奖", "金奖"]):
        return 1.15
    if any(k in t for k in ["国二", "二等奖", "银奖", "m奖"]):
        return 1.08
    if any(k in t for k in ["国三", "三等奖", "铜奖", "h奖", "f奖"]):
        return 1.03
    if "省一" in t:
        return 1.06
    if "省二" in t:
        return 1.03
    if "省三" in t:
        return 1.01
    if "国奖" in t or "国家奖学金" in t:
        return 1.1
    if "省奖" in t:
        return 1.02
    return 1.0


def match_value(item: str, value_map: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not value_map:
        return None, 35.0
    low = item.lower()
    exact = [k for k in value_map if k.lower() == low]
    if exact:
        k = exact[0]
        return k, value_map[k]

    # 选择最长匹配，降低歧义
    cands = [k for k in value_map if k.lower() in low or low in k.lower()]
    if cands:
        k = sorted(cands, key=len, reverse=True)[0]
        return k, value_map[k]
    if value_map:
        return None, sum(value_map.values()) / len(value_map)
    return None, 35.0


def main() -> None:
    parser = argparse.ArgumentParser(description="简单结构化输入测试：输出条目得分与院校Bar排序")
    parser.add_argument(
        "--input-json",
        type=str,
        default="",
        help="用户输入：JSON 字符串；若该路径指向已存在的文件则按文件读取（与 --input-file 二选一）",
    )
    parser.add_argument("--input-file", type=str, default="", help="用户输入 JSON 文件路径")
    args = parser.parse_args()

    if args.input_file:
        user = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    elif args.input_json:
        raw = args.input_json.strip()
        p = Path(raw)
        if p.is_file():
            user = json.loads(p.read_text(encoding="utf-8"))
        else:
            user = json.loads(raw)
    else:
        raise RuntimeError("请通过 --input-json 或 --input-file 提供输入")

    comp_json = read_json(MODEL_RESULT_DIR / "comp_value.json")
    paper_json = read_json(MODEL_RESULT_DIR / "paper_value.json")
    tier_json_path = MODEL_RESULT_DIR / "tier_value.json"
    tier_json = read_json(tier_json_path) if tier_json_path.exists() else {"values": {}}
    school_json = read_json(MODEL_RESULT_DIR / "school_bar.json")

    comp_values: Dict[str, float] = comp_json.get("values") or {}
    paper_values: Dict[str, float] = paper_json.get("values") or {}
    tier_values: Dict[str, float] = tier_json.get("values") or {}
    groups: List[Dict[str, Any]] = school_json.get("groups") or []

    tier_text = normalize_text(user.get("院校层次") or user.get("院校层级") or user.get("本科院校层次"))
    if tier_text not in TIER_SCORE:
        tier_text = "未知"
    tier_feature = float(tier_values.get(tier_text, TIER_SCORE[tier_text] / 9.0 * 100.0))

    rank_text = normalize_text(user.get("成绩排名") or user.get("成绩") or user.get("rank"))
    score_raw = parse_rank_score_text(rank_text)
    score_mean = compute_score_mean()
    score_feature = build_score_feature(score_raw, score_mean=score_mean, amplify=1.0)

    comp_items = parse_items(user.get("竞赛获奖条目"))
    paper_items = parse_items(user.get("论文条目"))

    comp_scores: List[float] = []
    for it in comp_items:
        _, base = match_value(it, comp_values)
        # 简化：竞赛模块直接使用训练好的 value 分数做聚合
        comp_scores.append(base)

    paper_scores: List[float] = []
    for it in paper_items:
        _, base = match_value(it, paper_values)
        paper_scores.append(base)

    comp_feature = sum(comp_scores) / len(comp_scores) if comp_scores else 0.0
    paper_feature = sum(paper_scores) / len(paper_scores) if paper_scores else 0.0

    ranked = []
    for g in groups:
        w = g.get("weights") or {}
        wt = float(w.get("院校层级", 0.0))
        ws = float(w.get("成绩", 0.0))
        wc = float(w.get("竞赛", 0.0))
        wp = float(w.get("论文", 0.0))
        score = wt * tier_feature + ws * score_feature + wc * comp_feature + wp * paper_feature
        suggest = float(g.get("suggested_bar_score", 0.0))
        ranked.append(
            {
                "school_dept": g.get("school_dept"),
                "total_score": round(score, 4),
                "suggested_bar_score": round(suggest, 4),
                "predict_admit": score >= suggest,
                "bar_gap": round(score - suggest, 4),
                "acc": g.get("acc"),
            }
        )

    ranked.sort(key=lambda x: x["total_score"], reverse=True)

    output = {
        "module_scores": {
            "tier_text": tier_text,
            "院校层级分": round(tier_feature, 4),
            "rank_text": rank_text,
            "成绩原始分": None if score_raw is None else round(score_raw, 4),
            "成绩模块分": round(score_feature, 4),
            "竞赛模块分": round(comp_feature, 4),
            "论文模块分": round(paper_feature, 4),
        },
        "school_ranking": ranked,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

