#!/usr/bin/env python3
"""下载 ToolACE 多轮 tool-use 数据集。

从 HuggingFace 下载 Team-ACE/ToolACE 数据集到 data/toolace/raw/。
"""

import json
import sys
from pathlib import Path
from loguru import logger


def main():
    output_dir = Path(__file__).resolve().parent.parent / "data" / "toolace" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("正在从 HuggingFace 加载 Team-ACE/ToolACE ...")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("需要 datasets 库: pip install datasets")
        sys.exit(1)

    # ToolACE 数据集
    ds = load_dataset("Team-ACE/ToolACE", split="train")
    logger.info(f"加载完成: {len(ds)} 条样本")
    logger.info(f"字段: {list(ds.column_names)}")

    # 保存为 JSON Lines 格式
    output_path = output_dir / "toolace_train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(ds):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if i < 3:
                logger.info(f"  样本 {i} keys: {list(sample.keys())}")
                # 打印部分内容帮助调试格式
                for k, v in sample.items():
                    s = repr(v)
                    if len(s) > 200:
                        s = s[:200] + "..."
                    logger.info(f"    {k}: {s}")

    logger.info(f"保存完成: {output_path} ({len(ds)} 条)")

    # 同时保存一份元信息
    meta = {
        "source": "Team-ACE/ToolACE",
        "split": "train",
        "num_samples": len(ds),
        "columns": list(ds.column_names),
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"元信息: {meta_path}")


if __name__ == "__main__":
    main()
