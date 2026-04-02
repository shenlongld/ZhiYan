"""
将 vectorsize / value / train 的中间与最终结果写入 model/result/。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

MODEL_RESULT_DIR = Path(__file__).resolve().parent / "result"


def _strip_prefix(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def save_pipeline_results(
    artifacts: Dict[str, Any],
    data_path: str,
    out_dir: str | Path | None = None,
) -> Path:
    out = Path(out_dir) if out_dir is not None else MODEL_RESULT_DIR
    out.mkdir(parents=True, exist_ok=True)

    samples = artifacts["samples"]
    vt = artifacts["value_tables"]
    models = artifacts["models"]

    meta = {
        "data_path": str(Path(data_path).resolve()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": len(samples),
        "aggregate": artifacts.get("aggregate", "school_college"),
        "use_camp_llm": artifacts.get("use_camp_llm", False),
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from .vectorsize import build_vector_summary

    summary = build_vector_summary(samples)
    (out / "vector_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (out / "value_tables.json").write_text(
        json.dumps(vt, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (out / "tier_value.json").write_text(
        json.dumps({"values": _strip_prefix(vt["school_tier"], "TIER_")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "paper_value.json").write_text(
        json.dumps({"values": _strip_prefix(vt["paper"], "PAPER_")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "comp_value.json").write_text(
        json.dumps(
            {"values": _strip_prefix(vt["competition"], "COMP_")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    groups = []
    for m in models.values():
        w = m.dimension_weights
        groups.append(
            {
                "school_dept": m.program_key,
                "weights": {
                    "院校层级": w["school_tier"],
                    "论文": w["paper"],
                    "竞赛": w["competition"],
                    "成绩": w["rank"],
                },
                "suggested_bar_score": m.recommend_apply_score,
                "bar_score": m.bar_score,
                "sample_size": m.sample_size,
            }
        )
    (out / "school_bar.json").write_text(
        json.dumps({"groups": groups}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    program_models = [
        {
            "program_key": m.program_key,
            "dimension_weights": m.dimension_weights,
            "bar_score": m.bar_score,
            "recommend_apply_score": m.recommend_apply_score,
            "sample_size": m.sample_size,
        }
        for m in models.values()
    ]
    (out / "program_models.json").write_text(
        json.dumps(program_models, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return out
