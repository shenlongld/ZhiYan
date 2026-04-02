"""
value.py
根据入营/未入营样本构建条目价值表。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from .vectorsize import TrainingSample


def minmax_scale(values: Dict[str, float], default: float = 0.5) -> Dict[str, float]:
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    if math.isclose(vmin, vmax):
        return {k: default for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


def build_value_tables(samples: List[TrainingSample]) -> Dict[str, Dict[str, float]]:
    """
    规则:
    - 入营条目 +1 * (未入营人数 / 总人数)
    - 未入营条目 -1 * (入营人数 / 总人数)
    最终全局归一化到 0~1。
    """
    tier_raw: Dict[str, float] = defaultdict(float)
    paper_raw: Dict[str, float] = defaultdict(float)
    competition_raw: Dict[str, float] = defaultdict(float)
    grouped: Dict[str, List[TrainingSample]] = defaultdict(list)

    for sample in samples:
        grouped[sample.program_key].append(sample)

    for group in grouped.values():
        total = len(group)
        if total == 0:
            continue
        admitted_count = sum(1 for s in group if s.admitted is True)
        reject_count = sum(1 for s in group if s.admitted is False)
        pos_scale = reject_count / total
        neg_scale = admitted_count / total

        for sample in group:
            if sample.admitted is None:
                continue
            sign = 1.0 if sample.admitted else -1.0
            scale = pos_scale if sample.admitted else neg_scale
            delta = sign * scale

            tier_raw[f"TIER_{sample.school_tier}"] += delta
            for token in sample.paper_tokens:
                paper_raw[token] += delta
            for token in sample.competition_tokens:
                competition_raw[token] += delta

    return {
        "school_tier": minmax_scale(dict(tier_raw)),
        "paper": minmax_scale(dict(paper_raw)),
        "competition": minmax_scale(dict(competition_raw)),
    }
