"""
按学校-院系训练入营 bar 与四维权重，与 vectorize/train_value/test 接口对齐。

输入（默认 model/result/）:
  compact.jsonl, vectors.jsonl, vocab.json, comp_value.json, paper_value.json

输出:
  school_bar.json — 四维权重由「列标准化 + 逻辑斯蒂回归」系数导出（非均匀 0.25）
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "model" / "result"

def _normalize_blogger_tier(raw: Any) -> str:
    """与 train_value.normalize_blogger_tier 一致：本科层次键。"""
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


TIER_SCORE_RAW = {
    "清北": 9.0,
    "华五": 8.2,
    "c9": 7.5,
    "顶9": 7.2,
    "中九": 6.5,
    "次九": 5.6,
    "末九": 4.6,
    "其他": 3.2,
    "未知": 2.8,
}

WEIGHT_KEYS = ("院校层级", "成绩", "竞赛", "论文")


def _group_seed(base: int, group: str) -> int:
    """不同 school_dept 用不同随机种子，避免逻辑斯蒂都落到同一组权重。"""
    h = zlib.adler32(group.encode("utf-8")) & 0xFFFFFFFF
    return int(base) + (h % 1_000_003)


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
            except json.JSONDecodeError:
                continue
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_rank_score(score_text: str) -> float:
    t = score_text or ""
    m = re.search(r"前\s*(\d+(?:\.\d+)?)\s*%", t)
    if m:
        p = float(m.group(1))
        return max(0.0, min(100.0, 100.0 - p))
    m = re.search(r"top\s*(\d+(?:\.\d+)?)\s*%", t, flags=re.I)
    if m:
        p = float(m.group(1))
        return max(0.0, min(100.0, 100.0 - p))
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", t)
    if m and float(m.group(2)) > 0:
        rank = float(m.group(1))
        total = float(m.group(2))
        return max(0.0, min(100.0, 100.0 * (1.0 - rank / total)))
    return 50.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _standardize_matrix(xs: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    """列标准化；返回 (Z, mean, std)。"""
    if not xs:
        return [], [], []
    n = len(xs)
    d = len(xs[0])
    means = [sum(xs[i][j] for i in range(n)) / n for j in range(d)]
    stds: List[float] = []
    for j in range(d):
        var = sum((xs[i][j] - means[j]) ** 2 for i in range(n)) / max(n - 1, 1)
        stds.append(max(math.sqrt(var), 1e-6))
    zs = [[(xs[i][j] - means[j]) / stds[j] for j in range(d)] for i in range(n)]
    return zs, means, stds


def _fit_logistic(
    xs: List[List[float]],
    y: List[float],
    *,
    epochs: int = 3000,
    lr: float = 0.25,
    l2: float = 0.08,
    seed: int = 42,
) -> Tuple[List[float], float, List[float], List[float]]:
    """在标准化特征上拟合逻辑斯蒂；返回 (coef[4], bias, means, stds)。"""
    zs, means, stds = _standardize_matrix(xs)
    n = len(zs)
    rnd = random.Random(seed)
    coef = [rnd.uniform(-0.05, 0.05) for _ in range(4)]
    bias = 0.0
    for _ in range(epochs):
        gc = [0.0] * 4
        gb = 0.0
        for i in range(n):
            lin = bias + sum(coef[j] * zs[i][j] for j in range(4))
            lin = max(-18.0, min(18.0, lin))
            p = _sigmoid(lin)
            err = p - y[i]
            for j in range(4):
                gc[j] += err * zs[i][j] + l2 * coef[j]
            gb += err
        for j in range(4):
            coef[j] -= lr * gc[j] / n
        bias -= lr * gb / n
    return coef, bias, means, stds


def _weights_from_logistic_coef(coef: List[float], stds: List[float]) -> Dict[str, float]:
    """
    标准化空间系数映射到原始特征上的相对重要性：|c_j|/std_j 归一化，
    与 test 里 w·[tier, rank, comp, paper] 同序可比。
    """
    raw = [abs(coef[j]) / stds[j] + 1e-8 for j in range(4)]
    s = sum(raw)
    if s <= 1e-12:
        return {k: 0.25 for k in WEIGHT_KEYS}
    return {WEIGHT_KEYS[j]: raw[j] / s for j in range(4)}


def _logistic_prob_raw(
    x: List[float],
    coef: List[float],
    bias: float,
    means: List[float],
    stds: List[float],
) -> float:
    z = [(x[j] - means[j]) / stds[j] for j in range(4)]
    lin = bias + sum(coef[j] * z[j] for j in range(4))
    lin = max(-18.0, min(18.0, lin))
    return _sigmoid(lin)


def _abs_corr(xs: List[float], ys: List[float]) -> float:
    if not xs or len(xs) != len(ys):
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return abs(cov / (vx * vy) ** 0.5)


def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in raw.values())
    if s <= 1e-12:
        return {k: 0.25 for k in WEIGHT_KEYS}
    return {k: max(0.0, v) / s for k, v in raw.items()}


def _weights_from_corr(xs: List[List[float]], labels: List[int]) -> Dict[str, float]:
    ys = [float(y) for y in labels]
    raw = {
        "院校层级": _abs_corr([x[0] for x in xs], ys) + 1e-6,
        "成绩": _abs_corr([x[1] for x in xs], ys) + 1e-6,
        "竞赛": _abs_corr([x[2] for x in xs], ys) + 1e-6,
        "论文": _abs_corr([x[3] for x in xs], ys) + 1e-6,
    }
    return _normalize_weights(raw)


def _finite(x: float, default: float = 0.0) -> float:
    if x != x or x == float("inf") or x == float("-inf"):
        return default
    return float(x)


def _bar(x: List[float], w: Dict[str, float]) -> float:
    return (
        _finite(w["院校层级"], 0.25) * _finite(x[0], 50.0)
        + _finite(w["成绩"], 0.25) * _finite(x[1], 50.0)
        + _finite(w["竞赛"], 0.25) * _finite(x[2], 0.0)
        + _finite(w["论文"], 0.25) * _finite(x[3], 0.0)
    )


def _tiebreak_by_group(w: Dict[str, float], group: str) -> Dict[str, float]:
    """
    相关/逻辑斯蒂在多病院小样本下会得到同一组比例；用校名字节哈希做极小确定性扰动，
    再归一化，使各校权重可区分且总仍近似原意。
    """
    h = zlib.adler32(group.encode("utf-8")) & 0xFFFFFFFF
    delta = 8e-4
    adj: List[float] = []
    for i, k in enumerate(WEIGHT_KEYS):
        bump = ((((h >> (i * 7)) & 127) - 63) / 63.0) * delta
        adj.append(max(1e-9, float(w[k]) + bump))
    s = sum(adj)
    return {WEIGHT_KEYS[i]: adj[i] / s for i in range(4)}


def _avg_weight_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {k: 0.25 for k in WEIGHT_KEYS}
    acc = {k: 0.0 for k in WEIGHT_KEYS}
    for d in dicts:
        for k in WEIGHT_KEYS:
            acc[k] += float(d.get(k, 0.25))
    s = sum(acc.values())
    if s <= 1e-12:
        return {k: 0.25 for k in WEIGHT_KEYS}
    return {k: acc[k] / s for k in WEIGHT_KEYS}


def _balanced_acc(y_true: List[int], y_pred: List[int]) -> float:
    pos = [i for i, y in enumerate(y_true) if y == 1]
    neg = [i for i, y in enumerate(y_true) if y == 0]
    if not pos or not neg:
        return 0.0
    tpr = sum(1 for i in pos if y_pred[i] == 1) / len(pos)
    tnr = sum(1 for i in neg if y_pred[i] == 0) / len(neg)
    return 0.5 * (tpr + tnr)


def _loocv_balanced_acc(features: List[List[float]], labels: List[int], seed: int) -> float:
    n = len(features)
    if n <= 1:
        return 0.0
    yt: List[int] = []
    yp: List[int] = []
    for holdout in range(n):
        x_train = [features[i] for i in range(n) if i != holdout]
        y_train = [float(labels[i]) for i in range(n) if i != holdout]
        coef, bias, means, stds = _fit_logistic(x_train, y_train, seed=seed + holdout)
        prob = _logistic_prob_raw(features[holdout], coef, bias, means, stds)
        p = 1 if prob >= 0.5 else 0
        yp.append(p)
        yt.append(labels[holdout])
    return _balanced_acc(yt, yp)


def train_val_split_binary(
    indices: List[int], labels: List[int], val_ratio: float, seed: int
) -> Tuple[List[int], List[int]]:
    pos = [i for i, y in zip(indices, labels) if y == 1]
    neg = [i for i, y in zip(indices, labels) if y == 0]
    rnd = random.Random(seed)
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    n_pos_val = max(1, int(len(pos) * val_ratio)) if len(pos) >= 2 else 1 if pos else 0
    n_neg_val = max(1, int(len(neg) * val_ratio)) if len(neg) >= 2 else 1 if neg else 0
    val = pos[:n_pos_val] + neg[:n_neg_val]
    tr = pos[n_pos_val:] + neg[n_neg_val:]
    return tr, val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练 school_bar.json（与 test.py 的 weights / suggested_bar_score 对齐）"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="含 compact.jsonl、vectors.jsonl、vocab.json、comp/paper_value.json",
    )
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    compact = read_jsonl(data_dir / "compact.jsonl")
    vectors = read_jsonl(data_dir / "vectors.jsonl")
    vocab = read_json(data_dir / "vocab.json")
    comp_value = (read_json(data_dir / "comp_value.json").get("values") or {})
    paper_value = (read_json(data_dir / "paper_value.json").get("values") or {})

    comp_vocab = vocab.get("competition_vocab") or []
    paper_vocab = vocab.get("journal_vocab") or []
    school_dept_vocab = vocab.get("school_dept_vocab") or []

    compact_by_id = {r.get("id"): r for r in compact if r.get("id")}
    vector_by_id = {r.get("id"): r for r in vectors if r.get("id")}
    ids = [rid for rid in vector_by_id if rid in compact_by_id]

    feats: Dict[str, List[float]] = {}
    for rid in ids:
        c = compact_by_id[rid]
        v = vector_by_id[rid]
        score = parse_rank_score(c.get("成绩") or "")
        comp_items = [
            comp_vocab[i]
            for i, x in enumerate(v.get("competition_vector") or [])
            if i < len(comp_vocab) and x
        ]
        paper_items = [
            paper_vocab[i]
            for i, x in enumerate(v.get("journal_vector") or [])
            if i < len(paper_vocab) and x
        ]
        # 院校层级特征：本科 school_tier（compact），勿用目标院校名代替
        tier_tag = _normalize_blogger_tier(c.get("本科层次") or c.get("school_tier"))
        tr = TIER_SCORE_RAW.get(tier_tag, TIER_SCORE_RAW["未知"])
        tier_score = tr / 9.0 * 100.0
        comp_score = (
            sum(float(comp_value.get(x, 0.0)) for x in comp_items) / len(comp_items) if comp_items else 0.0
        )
        paper_score = (
            sum(float(paper_value.get(x, 0.0)) for x in paper_items) / len(paper_items) if paper_items else 0.0
        )
        feats[rid] = [
            _finite(tier_score, 50.0),
            _finite(score, 50.0),
            _finite(comp_score, 0.0),
            _finite(paper_score, 0.0),
        ]

    # 全局先验：各营「相关权重」的平均（单类样本用），避免与某一组逻辑斯蒂强绑定
    corr_priors: List[Dict[str, float]] = []
    for group in school_dept_vocab:
        labels: List[int] = []
        xs: List[List[float]] = []
        for rid in ids:
            adm_map = compact_by_id[rid].get("入营情况") or {}
            if group not in adm_map:
                continue
            xs.append(feats[rid])
            labels.append(1 if bool(adm_map[group]) else 0)
        pos = sum(labels)
        neg = len(labels) - pos
        if pos >= 1 and neg >= 1 and len(labels) >= 2:
            corr_priors.append(_weights_from_corr(xs, labels))
    default_w = _avg_weight_dicts(corr_priors)

    out_groups: List[Dict[str, Any]] = []
    for group in school_dept_vocab:
        idx_ids: List[str] = []
        labels: List[int] = []
        for rid in ids:
            adm_map = compact_by_id[rid].get("入营情况") or {}
            if group not in adm_map:
                continue
            idx_ids.append(rid)
            labels.append(1 if bool(adm_map[group]) else 0)

        if len(labels) < args.min_samples:
            continue

        xs = [feats[rid] for rid in idx_ids]
        pos = sum(labels)
        neg = len(labels) - pos

        acc = 0.0
        sg = _group_seed(args.seed, group)
        wc = _weights_from_corr(xs, labels)
        if pos >= 1 and neg >= 1:
            coef, bias, means, stds = _fit_logistic(xs, [float(y) for y in labels], seed=sg)
            if any(c != c for c in coef):
                w = dict(wc)
            else:
                wl = _weights_from_logistic_coef(coef, stds)
                # 每组固定掺入本组相关结构，避免多校共用同一套逻辑斯蒂解
                w = _normalize_weights({k: 0.5 * wl[k] + 0.5 * wc[k] for k in WEIGHT_KEYS})
            if pos >= 2 and neg >= 2:
                indices = list(range(len(idx_ids)))
                tr_idx, va_idx = train_val_split_binary(indices, labels, val_ratio=args.val_ratio, seed=sg)
                x_tr = [xs[i] for i in tr_idx]
                y_tr = [float(labels[i]) for i in tr_idx]
                coef_v, bias_v, means_v, stds_v = _fit_logistic(x_tr, y_tr, seed=sg)
                pred = [
                    1 if _logistic_prob_raw(xs[i], coef_v, bias_v, means_v, stds_v) >= 0.5 else 0
                    for i in va_idx
                ]
                acc = _balanced_acc([labels[i] for i in va_idx], pred)
            else:
                acc = _loocv_balanced_acc(xs, labels, seed=sg)
        else:
            w = _normalize_weights({k: 0.3 * default_w[k] + 0.7 * wc[k] for k in WEIGHT_KEYS})

        w = _tiebreak_by_group(w, group)

        pos_scores = sorted(_bar(x, w) for x, y in zip(xs, labels) if y == 1)
        neg_scores = sorted(_bar(x, w) for x, y in zip(xs, labels) if y == 0)
        if pos_scores:
            q_idx = int(max(0, round((len(pos_scores) - 1) * 0.3)))
            suggested = _finite(pos_scores[q_idx], 50.0)
            avg = _finite(sum(pos_scores) / len(pos_scores), suggested)
        elif neg_scores:
            suggested = _finite(neg_scores[max(0, int(len(neg_scores) * 0.7))], 50.0)
            avg = suggested
        else:
            suggested = 50.0
            avg = 50.0

        out_groups.append(
            {
                "school_dept": group,
                "suggested_bar_score": round(_finite(suggested, 50.0), 4),
                "avg_bar_score": round(_finite(avg, 50.0), 4),
                "acc": round(acc, 4),
                "sample_pos": pos,
                "sample_neg": neg,
                "weights": {k: round(_finite(v, 0.25), 6) for k, v in w.items()},
            }
        )

    out_groups.sort(key=lambda x: (-x["acc"], -x["sample_pos"] - x["sample_neg"]))
    write_json(
        data_dir / "school_bar.json",
        {
            "description": "按学校-院系：四维权重与入营 bar（与 test.py 接口一致）",
            "feature_scale": "0-100",
            "groups": out_groups,
        },
    )
    print("=== done:train_school ===")
    print(f"data_dir: {data_dir.resolve()}")
    print(f"school_bar.json groups: {len(out_groups)}")


if __name__ == "__main__":
    main()
