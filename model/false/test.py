"""
test.py
根据用户结构化画像对项目做 gap 排序。
"""
from __future__ import annotations

from typing import Any, Dict, List

from .train import ProgramModel, build_models_from_data, sample_to_features
from .vectorsize import (
    TrainingSample,
    extract_competition_tokens,
    normalize_paper_token,
    parse_float,
)


def rank_programs_for_user(
    user_profile: Dict[str, Any],
    models: Dict[str, ProgramModel],
    value_tables: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    school_tier = (user_profile.get("school_tier") or "未知").strip() or "未知"
    rank_percent = parse_float(user_profile.get("rank_percent"))

    paper_tokens: List[str] = []
    for item in user_profile.get("research_achievements", []):
        if isinstance(item, dict):
            token = normalize_paper_token(item)
            if token:
                paper_tokens.append(token)

    competition_tokens = extract_competition_tokens(
        [x for x in user_profile.get("competition_achievements", []) if isinstance(x, dict)]
    )

    pseudo_sample = TrainingSample(
        program_key="USER",
        school_tier=school_tier,
        rank_percent=rank_percent,
        admitted=None,
        paper_tokens=paper_tokens,
        competition_tokens=competition_tokens,
    )
    f = sample_to_features(pseudo_sample, value_tables)

    ranked: List[Dict[str, Any]] = []
    for program_key, model in models.items():
        w = model.dimension_weights
        user_score = (
            w["school_tier"] * f[0]
            + w["paper"] * f[1]
            + w["competition"] * f[2]
            + w["rank"] * f[3]
        ) * 100
        gap = model.recommend_apply_score - user_score
        ranked.append(
            {
                "program_key": program_key,
                "user_score": round(user_score, 2),
                "recommend_apply_score": model.recommend_apply_score,
                "gap": round(gap, 2),
                "bar_score": model.bar_score,
                "sample_size": model.sample_size,
                "weights": model.dimension_weights,
            }
        )
    ranked.sort(key=lambda x: x["gap"], reverse=True)
    return ranked


def run_user_ranking(data_path: str, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifacts = build_models_from_data(data_path)
    return rank_programs_for_user(
        user_profile=user_profile,
        models=artifacts["models"],
        value_tables=artifacts["value_tables"],
    )
