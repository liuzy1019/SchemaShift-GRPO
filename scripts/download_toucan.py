#!/usr/bin/env python3
"""下载 Toucan 数据子集用于 inspection。

Toucan-1.5M 数据集来源: https://github.com/TheAgentArk/Toucan
HuggingFace: Agent-Ark/Toucan-1.5M

可用 configs: Kimi-K2, OSS, Qwen3, SFT
- SFT: 包含完整 tool-call 轨迹，适合 inspection 和 episode 构建
- OSS: 原始 MCP 数据

本脚本下载一个可控子集（默认 5000 条），存储到 data/toucan/ 目录。
"""

import argparse
import json
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "toucan"


def download_toucan(
    subset_size: int = 5000,
    dataset_name: str = "Agent-Ark/Toucan-1.5M",
    config_name: str = "SFT",
    split: str = "train",
    seed: int = 42,
) -> Path:
    """从 HuggingFace 下载 Toucan 数据子集。

    Args:
        subset_size: 下载的样本数量
        dataset_name: HuggingFace 数据集名称
        config_name: 数据集 config (Kimi-K2, OSS, Qwen3, SFT)
        split: 数据集 split
        seed: 随机种子（用于 shuffle 后取子集）

    Returns:
        保存的文件路径
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("错误: 需要安装 datasets 库")
        print("  pip install datasets")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"正在从 HuggingFace 加载 {dataset_name} (config={config_name}) ...")
    print(f"  split: {split}")
    print(f"  subset_size: {subset_size}")

    # 加载数据集
    ds = load_dataset(dataset_name, config_name, split=split)
    print(f"  总样本数: {len(ds)}")

    # shuffle 并取子集
    if subset_size < len(ds):
        ds_subset = ds.shuffle(seed=seed).select(range(subset_size))
        print(f"  随机采样 {subset_size} 条")
    else:
        ds_subset = ds
        print(f"  数据集不足 {subset_size} 条，使用全部 {len(ds)} 条")

    # 保存为 JSONL
    config_tag = config_name.lower()
    output_path = DATA_DIR / f"toucan_{config_tag}_subset_{len(ds_subset)}.jsonl"
    print(f"正在保存到 {output_path} ...")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds_subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"完成! 保存了 {len(ds_subset)} 条到 {output_path}")

    # 同时保存数据集 schema 信息
    schema_path = DATA_DIR / f"schema_info_{config_tag}.json"
    schema_info = {
        "dataset_name": dataset_name,
        "config_name": config_name,
        "total_samples": len(ds),
        "subset_size": len(ds_subset),
        "columns": list(ds.column_names),
        "features": {k: str(v) for k, v in ds.features.items()},
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2, ensure_ascii=False)
    print(f"Schema 信息保存到 {schema_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="下载 Toucan 数据子集")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=5000,
        help="下载的样本数量 (默认: 5000)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Agent-Ark/Toucan-1.5M",
        help="HuggingFace 数据集名称",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="SFT",
        choices=["Kimi-K2", "OSS", "Qwen3", "SFT"],
        help="数据集 config (默认: SFT)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集 split (默认: train)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    args = parser.parse_args()

    download_toucan(
        subset_size=args.subset_size,
        dataset_name=args.dataset,
        config_name=args.config,
        split=args.split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
