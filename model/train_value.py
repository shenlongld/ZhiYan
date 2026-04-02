from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "model" / "result"

SCHOOL_TIER_BY_NAME = {
    "清华大学": "清北",
    "北京大学": "清北",
    "浙江大学": "华五",
    "复旦大学": "华五",
    "上海交通大学": "华五",
    "南京大学": "华五",
    "中国科学技术大学": "华五",
    "哈尔滨工业大学": "c9",
    "西安交通大学": "c9",
    "北京航空航天大学": "顶9",
    "同济大学": "顶9",
    "东南大学": "顶9",
    "武汉大学": "顶9",
    "华中科技大学": "顶9",
    "中山大学": "顶9",
    "厦门大学": "中九",
    "天津大学": "中九",
    "南开大学": "中九",
    "北京邮电大学": "次九",
}


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_to_100(raw_scores: Dict[str, float]) -> Dict[str, float]:
    if not raw_scores:
        return {}
    vals = list(raw_scores.values())
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {k: 50.0 for k in raw_scores}
    return {k: (v - lo) * 100.0 / (hi - lo) for k, v in raw_scores.items()}


def extract_active_items(vector: List[int], vocab: List[str]) -> List[str]:
    out: List[str] = []
    n = min(len(vector), len(vocab))
    for i in range(n):
        if vector[i]:
            out.append(vocab[i])
    return out


def school_from_school_dept(school_dept: str) -> str:
    return school_dept.split("-", 1)[0] if "-" in school_dept else school_dept


def normalize_blogger_tier(raw: Any) -> str:
    """将采集 school_tier 规范到 value 用的层级键（本科口径，非目标院校）。"""
    s = (str(raw).strip() if raw is not None else "").strip()
    if not s:
        return "未知"
    aliases = {
        "211": "其他",
        "985": "顶9",
        "双非": "其他",
        "四非": "其他",
        "普通一本": "其他",
    }
    s = aliases.get(s, s)
    if s.lower() == "c9":
        return "c9"
    allowed = {"清北", "华五", "c9", "顶9", "中九", "次九", "末九", "其他", "未知"}
    if s in allowed:
        return s
    return "未知"


def build_value_maps_from_vectors(
    compact_rows: List[Dict[str, Any]],
    vector_rows: List[Dict[str, Any]],
    vocab: Dict[str, Any],
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    List[Dict[str, Any]],
]:
    comp_vocab: List[str] = vocab.get("competition_vocab") or []
    paper_vocab: List[str] = vocab.get("journal_vocab") or []
    school_dept_vocab: List[str] = vocab.get("school_dept_vocab") or []

    compact_by_id = {r.get("id"): r for r in compact_rows if r.get("id")}
    vector_by_id = {r.get("id"): r for r in vector_rows if r.get("id")}

    school_occ: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rid, vec in vector_by_id.items():
        c = compact_by_id.get(rid)
        if not c:
            continue
        admission_map = c.get("入营情况") or {}
        if not isinstance(admission_map, dict):
            continue

        comp_items = extract_active_items(vec.get("competition_vector") or [], comp_vocab)
        paper_items = extract_active_items(vec.get("journal_vector") or [], paper_vocab)
        school_depts = extract_active_items(vec.get("school_dept_vector") or [], school_dept_vocab)
        if not school_depts:
            continue

        for sd in school_depts:
            if sd not in admission_map:
                continue
            admitted = bool(admission_map.get(sd))
            school = school_from_school_dept(sd)
            # 院校层级价值：按发帖者本科层次 school_tier，不能用目标夏令营学校名推断
            tier_item = normalize_blogger_tier(c.get("本科层次") or c.get("school_tier"))
            school_occ[school].append(
                {
                    "admitted": admitted,
                    "comp_items": comp_items,
                    "paper_items": paper_items,
                    "tier_item": tier_item,
                }
            )

    comp_raw: Dict[str, float] = defaultdict(float)
    paper_raw: Dict[str, float] = defaultdict(float)
    tier_raw: Dict[str, float] = defaultdict(float)
    comp_pos: Dict[str, int] = defaultdict(int)
    comp_neg: Dict[str, int] = defaultdict(int)
    paper_pos: Dict[str, int] = defaultdict(int)
    paper_neg: Dict[str, int] = defaultdict(int)
    tier_pos: Dict[str, int] = defaultdict(int)
    tier_neg: Dict[str, int] = defaultdict(int)
    school_dist: List[Dict[str, Any]] = []

    for school, occs in school_occ.items():
        total = len(occs)
        if total <= 0:
            continue
        pos = sum(1 for x in occs if x["admitted"])
        neg = total - pos
        difficulty = neg / total
        school_dist.append(
            {"school": school, "total": total, "admit": pos, "reject": neg, "difficulty": round(difficulty, 6)}
        )

        for occ in occs:
            sign = 1.0 if occ["admitted"] else -1.0
            step = sign * difficulty

            for item in set(occ["comp_items"]):
                comp_raw[item] += step
                if sign > 0:
                    comp_pos[item] += 1
                else:
                    comp_neg[item] += 1
            for item in set(occ["paper_items"]):
                paper_raw[item] += step
                if sign > 0:
                    paper_pos[item] += 1
                else:
                    paper_neg[item] += 1

            tier_item = occ["tier_item"]
            tier_raw[tier_item] += step
            if sign > 0:
                tier_pos[tier_item] += 1
            else:
                tier_neg[tier_item] += 1

    comp_map = normalize_to_100(dict(comp_raw))
    paper_map = normalize_to_100(dict(paper_raw))
    tier_map = normalize_to_100(dict(tier_raw))

    def build_stats(
        raw: Dict[str, float],
        norm: Dict[str, float],
        pos_cnt: Dict[str, int],
        neg_cnt: Dict[str, int],
    ) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k, v in raw.items():
            p = pos_cnt.get(k, 0)
            n = neg_cnt.get(k, 0)
            out[k] = {
                "raw_score": float(v),
                "value_score": float(norm.get(k, 50.0)),
                "sample_size": float(p + n),
                "pos_count": float(p),
                "neg_count": float(n),
            }
        return out

    comp_stats = build_stats(comp_raw, comp_map, comp_pos, comp_neg)
    paper_stats = build_stats(paper_raw, paper_map, paper_pos, paper_neg)
    tier_stats = build_stats(tier_raw, tier_map, tier_pos, tier_neg)
    school_dist.sort(key=lambda x: (x["difficulty"], x["total"]), reverse=True)
    return comp_map, paper_map, tier_map, comp_stats, paper_stats, tier_stats, school_dist


def main() -> None:
    parser = argparse.ArgumentParser(description="仅使用向量化产物训练权重")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="向量化产物目录")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    compact_rows = read_jsonl(data_dir / "compact.jsonl")
    vector_rows = read_jsonl(data_dir / "vectors.jsonl")
    vocab = read_json(data_dir / "vocab.json")
    if not compact_rows or not vector_rows:
        raise RuntimeError("缺少 compact.jsonl 或 vectors.jsonl，请先运行 vectorize.py")

    comp_map, paper_map, tier_map, comp_stats, paper_stats, tier_stats, school_dist = (
        build_value_maps_from_vectors(compact_rows, vector_rows, vocab)
    )

    write_json(
        data_dir / "comp_value.json",
        {
            "description": "同院校口径竞赛权重（向量化输入）",
            "count": len(comp_map),
            "values": {k: round(v, 4) for k, v in sorted(comp_map.items(), key=lambda kv: kv[1], reverse=True)},
            "school_distribution": school_dist,
            "metrics": {
                k: {
                    "raw_score": round(v["raw_score"], 4),
                    "sample_size": int(v["sample_size"]),
                    "pos_count": int(v["pos_count"]),
                    "neg_count": int(v["neg_count"]),
                }
                for k, v in sorted(comp_stats.items(), key=lambda kv: kv[1]["value_score"], reverse=True)
            },
        },
    )
    write_json(
        data_dir / "paper_value.json",
        {
            "description": "同院校口径论文权重（向量化输入）",
            "count": len(paper_map),
            "values": {k: round(v, 4) for k, v in sorted(paper_map.items(), key=lambda kv: kv[1], reverse=True)},
            "school_distribution": school_dist,
            "metrics": {
                k: {
                    "raw_score": round(v["raw_score"], 4),
                    "sample_size": int(v["sample_size"]),
                    "pos_count": int(v["pos_count"]),
                    "neg_count": int(v["neg_count"]),
                }
                for k, v in sorted(paper_stats.items(), key=lambda kv: kv[1]["value_score"], reverse=True)
            },
        },
    )
    write_json(
        data_dir / "tier_value.json",
        {
            "description": "同院校口径院校层级权重（向量化输入）",
            "count": len(tier_map),
            "values": {k: round(v, 4) for k, v in sorted(tier_map.items(), key=lambda kv: kv[1], reverse=True)},
            "school_distribution": school_dist,
            "metrics": {
                k: {
                    "raw_score": round(v["raw_score"], 4),
                    "sample_size": int(v["sample_size"]),
                    "pos_count": int(v["pos_count"]),
                    "neg_count": int(v["neg_count"]),
                }
                for k, v in sorted(tier_stats.items(), key=lambda kv: kv[1]["value_score"], reverse=True)
            },
        },
    )
    write_json(
        data_dir / "school_distribution.json",
        {"description": "院校样本分布（用于校验）", "schools": school_dist},
    )
    print("=== done:value(vector-only) ===")
    print(f"records: compact={len(compact_rows)}, vectors={len(vector_rows)}")


if __name__ == "__main__":
    main()
