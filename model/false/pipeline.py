"""
命令行：读数据 → 训练 → 写入 model/result/。

用法（在项目根目录）:
  python -m model.pipeline --data data/baoyan_experience_profiles_20260329_134614.jsonl
  python -m model.pipeline --data data/xxx.csv --out model/result
  # 按「目标院校夏令营」聚合 + LLM：密钥与网关读 postgrad_agent/.env（OPENAI_*）
  python -m model.pipeline --data data/xxx.jsonl --aggregate camp_summer --camp-llm
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .camp_extract_llm import load_postgrad_dotenv
from .result_io import MODEL_RESULT_DIR
from .train import build_models_from_data


def main() -> None:
    load_postgrad_dotenv()

    parser = argparse.ArgumentParser(description="保研建模：数据 → 训练 → 输出到 result")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="经验贴 jsonl 或 csv 路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help=f"输出目录，默认 {MODEL_RESULT_DIR}",
    )
    parser.add_argument(
        "--aggregate",
        choices=("school_college", "camp_summer"),
        default="school_college",
        help="聚合键：本科学校·专业 或 目标院校夏令营（prompt 任务 1）",
    )
    parser.add_argument(
        "--camp-llm",
        action="store_true",
        help="夏令营维度下用 LLM（OPENAI_API_KEY + OPENAI_BASE_URL，如 DeepSeek）抽取多校入营",
    )
    parser.add_argument(
        "--camp-llm-cache",
        type=str,
        default="",
        help="LLM 结果缓存 jsonl，默认 model/result/camp_llm_cache.jsonl",
    )
    parser.add_argument(
        "--camp-llm-progress",
        type=int,
        default=20,
        help="每处理多少行打印进度（0 关闭）",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"数据文件不存在: {data_path.resolve()}")

    out_dir = Path(args.out) if args.out.strip() else MODEL_RESULT_DIR
    cache_path = args.camp_llm_cache.strip() or None
    artifacts = build_models_from_data(
        str(data_path),
        save_to=out_dir,
        aggregate=args.aggregate,
        use_camp_llm=args.camp_llm,
        camp_llm_cache_path=cache_path,
        camp_llm_progress_every=args.camp_llm_progress,
    )

    print(f"样本数: {len(artifacts['samples'])}")
    print(f"已写入: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
