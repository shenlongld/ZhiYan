"""
vectorsize.py
读取经验贴数据并做向量化统计。

- aggregate=school_college：本科「学校·专业」键（与采集 school_college 一致）。
- aggregate=camp_summer：帖子中「目标院校·学院」夏令营键（prompt 任务 1）；可选 LLM（DeepSeek 等）
  抽取多校入营，否则启发式院校键 + 结构化 admitted（多校时同帖共用一条入营标签）。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_CAMP_LLM_CACHE = Path(__file__).resolve().parent / "result" / "camp_llm_cache.jsonl"

# 与 postgrad_agent 共用稀疏逻辑（无 LLM 依赖）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from postgrad_agent.sparse_experience import (
    build_school_college_key,
    merge_competitions,
    merge_research,
    normalize_text,
    sparse_extract_competitions,
    sparse_extract_rank_fields,
    sparse_extract_research,
)


@dataclass
class TrainingSample:
    """program_key：school_college 模式下为本科画像键；camp_summer 模式下为目标院校·学院夏令营键。"""

    program_key: str
    school_tier: str
    rank_percent: Optional[float]
    admitted: Optional[bool]
    paper_tokens: List[str]
    competition_tokens: List[str]


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def parse_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x < 0 or x > 100_000:
        return None
    return x


def safe_json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def normalize_paper_token(item: Dict[str, Any]) -> Optional[str]:
    venue = (item.get("venue_or_level") or "").strip().upper()
    if not venue:
        return None
    if "CCF-A" in venue or "CCF A" in venue:
        return "PAPER_CCF_A"
    if "CCF-B" in venue or "CCF B" in venue:
        return "PAPER_CCF_B"
    if "CCF-C" in venue or "CCF C" in venue:
        return "PAPER_CCF_C"
    if "SCI" in venue:
        return "PAPER_SCI"
    if "EI" in venue:
        return "PAPER_EI"
    if any(
        x in venue
        for x in (
            "NEURIPS",
            "ICML",
            "ICLR",
            "CVPR",
            "ECCV",
            "ICCV",
            "AAAI",
            "IJCAI",
            "WWW",
            "KDD",
        )
    ):
        return "PAPER_TOP_CONF"
    return f"PAPER_{venue[:24]}"


def extract_competition_tokens(items: List[Dict[str, Any]]) -> List[str]:
    tokens: List[str] = []
    for item in items:
        base_name = (item.get("normalized_name") or item.get("name") or "").strip()
        if not base_name:
            continue
        tokens.append(f"COMP_{base_name.upper()[:24]}")
    return tokens


def _admission_detail(row: Dict[str, Any]) -> str:
    if "admission_detail" in row:
        return normalize_text(row.get("admission_detail"))
    camp = row.get("camp_admission")
    if isinstance(camp, dict):
        return normalize_text(camp.get("detail"))
    return ""


def _sparse_blob_for_row(row: Dict[str, Any]) -> str:
    return "\n".join(
        [
            normalize_text(row.get("source_title")),
            normalize_text(row.get("notes")),
            _admission_detail(row),
            normalize_text(row.get("blogger_school")),
            normalize_text(row.get("target_major")),
        ]
    )


def _struct_admitted_for_row(row: Dict[str, Any]) -> Optional[bool]:
    admitted = None
    camp_admission = row.get("camp_admission")
    if isinstance(camp_admission, dict):
        admitted = parse_bool(camp_admission.get("admitted"))
    if admitted is None:
        admitted = parse_bool(row.get("camp_admitted"))
    return admitted


def _person_features_from_row(row: Dict[str, Any]) -> Tuple[
    str,
    Optional[float],
    List[str],
    List[str],
]:
    school_tier = (row.get("school_tier") or "未知").strip() or "未知"
    rank_percent = parse_float(row.get("rank_percent"))

    research_items = safe_json_loads(
        row.get("research_achievements_json", row.get("research_achievements")),
        [],
    )
    competition_items = safe_json_loads(
        row.get("competition_achievements_json", row.get("competition_achievements")),
        [],
    )
    if not isinstance(research_items, list):
        research_items = []
    if not isinstance(competition_items, list):
        competition_items = []

    blob = _sparse_blob_for_row(row)
    sparse_comp = sparse_extract_competitions(blob)
    sparse_research = sparse_extract_research(blob)
    sparse_rank = sparse_extract_rank_fields(blob)

    research_merged = merge_research(
        [x for x in research_items if isinstance(x, dict)],
        sparse_research,
    )
    competition_merged = merge_competitions(
        [x for x in competition_items if isinstance(x, dict)],
        sparse_comp,
    )

    if rank_percent is None and sparse_rank.get("rank_percent") is not None:
        rank_percent = parse_float(sparse_rank.get("rank_percent"))

    paper_tokens: List[str] = []
    for item in research_merged:
        token = normalize_paper_token(item)
        if token:
            paper_tokens.append(token)

    competition_tokens = extract_competition_tokens(competition_merged)
    return school_tier, rank_percent, paper_tokens, competition_tokens


def _expand_camp_keys_for_row(
    row: Dict[str, Any],
    *,
    aggregate: str,
    use_camp_llm: bool,
    cache: Any,
    llm_cfg: Any,
    row_index: int,
    progress_every: int,
) -> List[Tuple[str, Optional[bool]]]:
    if aggregate == "school_college":
        program_key = normalize_text(row.get("school_college"))
        if not program_key:
            program_key = build_school_college_key(
                row.get("blogger_school"), row.get("target_major")
            )
        return [(program_key, _struct_admitted_for_row(row))]

    from .camp_keys import _blob, extract_camp_keys_from_text

    struct_adm = _struct_admitted_for_row(row)

    def heuristic_pairs() -> List[Tuple[str, Optional[bool]]]:
        keys = extract_camp_keys_from_text(_blob(row))
        if not keys:
            return [("未解析·院校夏令营", struct_adm)]
        return [(k, struct_adm) for k in keys]

    if not use_camp_llm or llm_cfg is None:
        return heuristic_pairs()

    from .camp_extract_llm import (
        blob_digest,
        call_camp_extract_sync,
        camp_llm_user_blob,
        parse_llm_camp_json_array,
    )

    blob_llm = camp_llm_user_blob(row)
    digest = blob_digest(blob_llm)
    items = cache.get(digest) if cache is not None else None

    if items is None:
        if progress_every > 0 and row_index > 0 and row_index % progress_every == 0:
            print(f"[camp_llm] 已处理约 {row_index} 行…", flush=True)
        items = []
        try:
            raw = call_camp_extract_sync(blob_llm, llm_cfg)
            parsed = parse_llm_camp_json_array(raw)
            for it in parsed:
                sc = normalize_text(it.get("school_college"))
                if not sc:
                    continue
                items.append(
                    {"school_college": sc, "admitted": it.get("admitted")}
                )
        except Exception as e:
            print(f"[camp_llm] 第 {row_index} 行 API 失败，改用启发式: {e}", flush=True)

        if not items:
            items = [
                {"school_college": k, "admitted": a}
                for k, a in heuristic_pairs()
            ]

        if cache is not None:
            cache.set(digest, items)

    out: List[Tuple[str, Optional[bool]]] = []
    for it in items:
        sc = normalize_text(it.get("school_college"))
        if not sc:
            continue
        out.append((sc, parse_bool(it.get("admitted"))))
    return out if out else heuristic_pairs()


def load_training_samples(
    data_path: str,
    *,
    aggregate: str = "school_college",
    use_camp_llm: bool = False,
    camp_llm_cache_path: Optional[str] = None,
    camp_llm_progress_every: int = 20,
) -> List[TrainingSample]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    if path.suffix.lower() not in {".jsonl", ".csv"}:
        raise ValueError("仅支持 .jsonl 或 .csv 文件")
    if aggregate not in {"school_college", "camp_summer"}:
        raise ValueError("aggregate 须为 school_college 或 camp_summer")

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

    cache = None
    llm_cfg = None
    if aggregate == "camp_summer" and use_camp_llm:
        from .camp_extract_llm import CampLLMCache, LLMConfig

        llm_cfg = LLMConfig.from_env()
        cache_path = (
            Path(camp_llm_cache_path)
            if (camp_llm_cache_path or "").strip()
            else _DEFAULT_CAMP_LLM_CACHE
        )
        cache = CampLLMCache(cache_path)

    samples: List[TrainingSample] = []
    for idx, row in enumerate(rows):
        school_tier, rank_percent, paper_tokens, competition_tokens = (
            _person_features_from_row(row)
        )
        for program_key, admitted in _expand_camp_keys_for_row(
            row,
            aggregate=aggregate,
            use_camp_llm=use_camp_llm,
            cache=cache,
            llm_cfg=llm_cfg,
            row_index=idx + 1,
            progress_every=camp_llm_progress_every,
        ):
            samples.append(
                TrainingSample(
                    program_key=program_key,
                    school_tier=school_tier,
                    rank_percent=rank_percent,
                    admitted=admitted,
                    paper_tokens=paper_tokens,
                    competition_tokens=competition_tokens,
                )
            )

    if cache is not None:
        cache.flush()

    return samples


def build_vector_summary(samples: List[TrainingSample]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"admitted": 0, "not_admitted": 0, "unknown": 0}
    )
    for sample in samples:
        if sample.admitted is True:
            summary[sample.program_key]["admitted"] += 1
        elif sample.admitted is False:
            summary[sample.program_key]["not_admitted"] += 1
        else:
            summary[sample.program_key]["unknown"] += 1
    return dict(summary)
