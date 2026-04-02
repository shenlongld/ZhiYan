"""
train.py
使用 tinyMLP 学习四维权重并生成每个项目的推荐分。
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .value import build_value_tables
from .vectorsize import TrainingSample, load_training_samples


@dataclass
class ProgramModel:
    program_key: str
    dimension_weights: Dict[str, float]
    bar_score: float
    recommend_apply_score: float
    sample_size: int


def sample_to_features(
    sample: TrainingSample,
    value_tables: Dict[str, Dict[str, float]],
) -> List[float]:
    tier_score = value_tables["school_tier"].get(f"TIER_{sample.school_tier}", 0.5)

    paper_values = [value_tables["paper"].get(t, 0.5) for t in sample.paper_tokens]
    paper_score = sum(paper_values) / len(paper_values) if paper_values else 0.0

    comp_values = [
        value_tables["competition"].get(t, 0.5) for t in sample.competition_tokens
    ]
    competition_score = sum(comp_values) / len(comp_values) if comp_values else 0.0

    rank_score = 0.5
    if sample.rank_percent is not None:
        rank_score = max(0.0, min(1.0, 1.0 - sample.rank_percent / 100.0))
    return [tier_score, paper_score, competition_score, rank_score]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def _relu(x: float) -> float:
    return x if x > 0 else 0.0


def _train_tiny_mlp(
    features: List[List[float]],
    labels: List[int],
    *,
    hidden_dim: int = 6,
    epochs: int = 400,
    lr: float = 0.02,
) -> Dict[str, Any]:
    input_dim = 4
    random.seed(42)
    w1 = [[random.uniform(-0.3, 0.3) for _ in range(hidden_dim)] for _ in range(input_dim)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [random.uniform(-0.3, 0.3) for _ in range(hidden_dim)]
    b2 = 0.0

    for _ in range(epochs):
        for x, y in zip(features, labels):
            z1 = [
                sum(x[i] * w1[i][j] for i in range(input_dim)) + b1[j]
                for j in range(hidden_dim)
            ]
            h1 = [_relu(v) for v in z1]
            z2 = sum(h1[j] * w2[j] for j in range(hidden_dim)) + b2
            pred = _sigmoid(z2)

            dz2 = pred - y
            grad_w2 = [dz2 * h1[j] for j in range(hidden_dim)]
            grad_b2 = dz2
            dh1 = [dz2 * w2[j] for j in range(hidden_dim)]
            dz1 = [dh1[j] if z1[j] > 0 else 0.0 for j in range(hidden_dim)]

            for j in range(hidden_dim):
                w2[j] -= lr * grad_w2[j]
            b2 -= lr * grad_b2

            for i in range(input_dim):
                for j in range(hidden_dim):
                    w1[i][j] -= lr * (x[i] * dz1[j])
            for j in range(hidden_dim):
                b1[j] -= lr * dz1[j]

    return {"w1": w1, "w2": w2}


def _derive_dimension_weights(network: Dict[str, Any]) -> Dict[str, float]:
    w1 = network["w1"]
    w2 = network["w2"]
    names = ["school_tier", "paper", "competition", "rank"]
    raw = {name: 0.0 for name in names}
    for i, name in enumerate(names):
        raw[name] = sum(abs(w1[i][j] * w2[j]) for j in range(len(w2)))
    total = sum(raw.values())
    if total <= 1e-12:
        return {name: 0.25 for name in names}
    return {name: raw[name] / total for name in names}


def train_program_models(
    samples: List[TrainingSample],
    value_tables: Dict[str, Dict[str, float]],
) -> Dict[str, ProgramModel]:
    grouped: Dict[str, List[TrainingSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.program_key].append(sample)

    models: Dict[str, ProgramModel] = {}
    for program_key, group in grouped.items():
        labeled = [s for s in group if s.admitted is not None]
        if len(labeled) < 6:
            weights = {
                "school_tier": 0.25,
                "paper": 0.25,
                "competition": 0.25,
                "rank": 0.25,
            }
            score_list = []
            for sample in group:
                f = sample_to_features(sample, value_tables)
                score_list.append(
                    weights["school_tier"] * f[0]
                    + weights["paper"] * f[1]
                    + weights["competition"] * f[2]
                    + weights["rank"] * f[3]
                )
        else:
            features = [sample_to_features(s, value_tables) for s in labeled]
            labels = [1 if s.admitted else 0 for s in labeled]
            network = _train_tiny_mlp(features, labels)
            weights = _derive_dimension_weights(network)
            score_list = [
                weights["school_tier"] * f[0]
                + weights["paper"] * f[1]
                + weights["competition"] * f[2]
                + weights["rank"] * f[3]
                for f in features
            ]

        score_list = sorted(score_list)
        bar_score = (sum(score_list) / len(score_list)) * 100 if score_list else 0.0
        pivot = max(0, int(len(score_list) * 0.7) - 1) if score_list else 0
        recommend_apply_score = score_list[pivot] * 100 if score_list else 50.0

        models[program_key] = ProgramModel(
            program_key=program_key,
            dimension_weights=weights,
            bar_score=round(bar_score, 2),
            recommend_apply_score=round(recommend_apply_score, 2),
            sample_size=len(group),
        )
    return models


def build_models_from_data(
    data_path: str,
    *,
    save_to: Optional[Union[str, Path]] = None,
    aggregate: str = "school_college",
    use_camp_llm: bool = False,
    camp_llm_cache_path: Optional[str] = None,
    camp_llm_progress_every: int = 20,
) -> Dict[str, Any]:
    samples = load_training_samples(
        data_path,
        aggregate=aggregate,
        use_camp_llm=use_camp_llm,
        camp_llm_cache_path=camp_llm_cache_path,
        camp_llm_progress_every=camp_llm_progress_every,
    )
    value_tables = build_value_tables(samples)
    models = train_program_models(samples, value_tables)
    artifacts: Dict[str, Any] = {
        "samples": samples,
        "value_tables": value_tables,
        "models": models,
        "aggregate": aggregate,
        "use_camp_llm": use_camp_llm,
    }
    if save_to is not None:
        from .result_io import save_pipeline_results

        save_pipeline_results(artifacts, data_path, save_to)
    return artifacts
