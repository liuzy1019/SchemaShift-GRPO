"""SFT Cold-Start 训练脚本。

用 SFT 数据让 Qwen3-4B 学会 MCP tool_call 格式。

目标：
  - 让模型学会 <tool_call>...</tool_call> 格式
  - 让模型学会正确的 JSON 参数填充
  - 让模型学会并行调用语法
  - 让模型学会 <final_answer>...</final_answer> 格式
  - 不追求泛化能力（那是 RL 阶段的目标）

超参设计依据（基于数据集精确 tokenize 分析）：
  - 数据集：9146 条样本
  - Token 长度分布（精确）：mean=1036, median=931, p95=1943, p99=2207, max=2478
  - max_seq_length=2560 覆盖 >99% 样本（仅 2.9% 超过 2048）
  - 8×L20 44GB GPU，4B bf16 全参数 + DeepSpeed ZeRO-2
  - effective_batch_size=128（per_device=4 × grad_accum=4 × 8 GPU）
  - 3 epochs → ~214 total steps
  - 显存估算：模型8GB + ZeRO-2 optimizer 4GB + activations ~10GB ≈ 22GB/44GB（充裕）

使用方式：
  # 全参数微调（8×L20 44GB，推荐）
  torchrun --nproc_per_node=8 scripts/sft_cold_start.py \\
      --deepspeed configs/ds_zero2.json

  # 如果 evaluation OOM，优先降低 eval batch 或关闭中途 eval
  torchrun --nproc_per_node=8 scripts/sft_cold_start.py \\
      --eval-batch-size 1 --eval-steps 100 --save-steps 100 \\
      --deepspeed configs/ds_zero2.json

  # 全参数微调（8×A10 24GB，需 batch=1）
  torchrun --nproc_per_node=8 scripts/sft_cold_start.py \\
      --batch-size 1 --grad-accum 8 --deepspeed configs/ds_zero2.json

  # LoRA 微调（单卡或显存不足时）
  python scripts/sft_cold_start.py --lora

  # smoke 模式（快速验证）
  python scripts/sft_cold_start.py --smoke
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from loguru import logger
except ModuleNotFoundError:
    class _FallbackLogger:
        def info(self, message): print(message)
        def warning(self, message): print(f"WARNING: {message}")
        def error(self, message): print(f"ERROR: {message}")
        def remove(self): pass
        def add(self, *args, **kwargs): pass

    logger = _FallbackLogger()


# ============================================================
# 分布式环境工具
# ============================================================


def _get_rank() -> int:
    """获取当前进程的全局 rank（兼容 torchrun / deepspeed launcher）。"""
    # torchrun 设置 RANK；deepspeed launcher 也设置 RANK
    return int(os.environ.get("RANK", 0))


def _get_local_rank() -> int:
    """获取当前进程的 local rank。"""
    return int(os.environ.get("LOCAL_RANK", 0))


def _is_main_process() -> bool:
    """判断当前进程是否为主进程（rank 0）。"""
    return _get_rank() == 0


def _setup_logging_for_distributed():
    """配置 loguru 使其仅在 rank 0 输出，避免分布式训练时日志重复。

    同时控制 DeepSpeed 和 transformers 的日志级别，
    确保终端只显示 rank 0 的进度条和日志，不被其他 rank 干扰。
    """
    if not _is_main_process():
        logger.remove()  # 移除默认 stderr handler
        # 添加一个 null handler 避免 loguru 报错
        logger.add(lambda msg: None, level="CRITICAL")
        # 抑制 DeepSpeed 在非主进程的日志输出
        os.environ["DEEPSPEED_LOG_LEVEL"] = "WARNING"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["DATASETS_VERBOSITY"] = "error"
        try:
            import datasets
            import transformers

            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
        except Exception:
            pass


# ============================================================
# 配置
# ============================================================


@dataclass
class SFTConfig:
    """SFT Cold-Start 配置。

    超参设计依据（精确 tokenize 分析 + 显存估算）：
      - 数据集 9146 条，token 长度 mean=1036, p95=1943, p99=2207, max=2478
      - 硬件 8×L20 44GB
      - Qwen3-4B BF16 ~8GB，ZeRO-2 optimizer 分片后 ~4GB/卡
        batch=4,seq=2560: 激活 ~10GB → 总峰值 ~22GB < 44GB（充裕）
      - effective_batch=128 → per_device=4 × grad_accum=4 × 8GPU
      - 3 epochs × 8689 / 128 ≈ 204 total steps
      - lr=2e-5 cosine，warmup 5%（~10 steps）
      - eval batch 默认更保守，避免 Qwen3 大 vocab logits 在 eval loss 计算时 OOM
    """
    # 模型
    model_name_or_path: str = "models/Qwen3-4B"
    # 数据
    train_data_path: str = "data/sft/sft_train.jsonl"
    eval_split_ratio: float = 0.05  # 5% 作为 eval set（~458 条）
    # 输出
    output_dir: str = "outputs/sft_cold_start_4b"
    # 训练超参（基于数据集分析）
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # effective_batch = 4×4×8GPU = 128
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05  # ~20 steps warmup
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 2560  # p99=2207, 2560 覆盖 >99%
    # 优化
    bf16: bool = True
    gradient_checkpointing: bool = True
    # 日志与保存
    logging_steps: int = 5
    eval_steps: int = 50  # 每 50 步 eval（总 ~204 步，约 4 次 eval）
    save_steps: int = 999999  # 不保存中间 checkpoint，只保留最终权重
    save_total_limit: int = 1  # 只保留最后一个 checkpoint
    eval_strategy: str = "steps"  # "steps" / "epoch" / "no"
    report_to: str = "tensorboard"
    dataloader_num_workers: int = 4
    # Smoke 模式
    smoke: bool = False
    smoke_max_samples: int = 100
    smoke_max_steps: int = 30
    # LoRA（可选，显存不足时使用）
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # DeepSpeed
    deepspeed: Optional[str] = None  # DeepSpeed 配置文件路径
    # 随机种子
    seed: int = 42


# ============================================================
# 数据加载
# ============================================================


def load_sft_data(path: str, max_samples: Optional[int] = None) -> list[dict]:
    """加载 SFT JSONL 数据。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append(item)
                if max_samples and len(data) >= max_samples:
                    break
    return data


def split_train_eval(data: list[dict], eval_ratio: float, seed: int = 42):
    """按比例划分 train/eval，保持 action_type 分布。"""
    import random
    from collections import defaultdict

    # 按 action_type 分组
    type_groups = defaultdict(list)
    for item in data:
        atype = item.get("metadata", {}).get("action_type", "unknown")
        type_groups[atype].append(item)

    rng = random.Random(seed)
    train_data, eval_data = [], []

    for atype, items in type_groups.items():
        rng.shuffle(items)
        n_eval = max(1, int(len(items) * eval_ratio))
        eval_data.extend(items[:n_eval])
        train_data.extend(items[n_eval:])

    rng.shuffle(train_data)
    rng.shuffle(eval_data)
    return train_data, eval_data


# ============================================================
# 训练主函数
# ============================================================


def run_sft(config: SFTConfig) -> dict:
    """运行 SFT cold-start 训练。返回训练指标字典。"""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    # 分布式环境下仅 rank 0 输出日志，避免 8 个进程重复打印
    _setup_logging_for_distributed()

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("SFT Cold-Start 训练")
    logger.info("=" * 60)
    logger.info(f"  模型: {config.model_name_or_path}")
    logger.info(f"  数据: {config.train_data_path}")
    logger.info(f"  输出: {config.output_dir}")
    logger.info(f"  模式: {'smoke' if config.smoke else 'full'}")
    logger.info(f"  LoRA: {config.use_lora}")
    logger.info(f"  max_seq_length: {config.max_seq_length}")
    logger.info(f"  epochs: {config.num_train_epochs}")
    logger.info(f"  per_device_batch: {config.per_device_train_batch_size}")
    logger.info(f"  per_device_eval_batch: {config.per_device_eval_batch_size}")
    logger.info(f"  grad_accum: {config.gradient_accumulation_steps}")
    logger.info(f"  lr: {config.learning_rate}")
    logger.info(f"  scheduler: {config.lr_scheduler_type}")
    logger.info(f"  warmup_ratio: {config.warmup_ratio}")
    logger.info(f"  eval_strategy: {config.eval_strategy}")
    logger.info(f"  eval_steps: {config.eval_steps}")
    logger.info(f"  save_steps: {config.save_steps}")
    logger.info(f"  logging_steps: {config.logging_steps}")
    logger.info(f"  report_to: {config.report_to}")
    logger.info(f"  dataloader_num_workers: {config.dataloader_num_workers}")

    # 创建输出目录
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载数据
    max_samples = config.smoke_max_samples if config.smoke else None
    raw_data = load_sft_data(config.train_data_path, max_samples=max_samples)
    logger.info(f"加载 {len(raw_data)} 条 SFT 样本")

    if not raw_data:
        logger.error("没有加载到任何数据，退出")
        return {}

    # 划分 train/eval
    if config.smoke:
        train_data = raw_data
        eval_data = raw_data[:10]  # smoke 模式用前 10 条做 eval
    else:
        train_data, eval_data = split_train_eval(
            raw_data, config.eval_split_ratio, config.seed
        )

    logger.info(f"Train: {len(train_data)} 条, Eval: {len(eval_data)} 条")

    # 统计 action_type 分布
    from collections import Counter
    train_types = Counter(
        item.get("metadata", {}).get("action_type", "unknown") for item in train_data
    )
    eval_types = Counter(
        item.get("metadata", {}).get("action_type", "unknown") for item in eval_data
    )
    logger.info(f"Train action_type 分布: {dict(train_types)}")
    logger.info(f"Eval action_type 分布: {dict(eval_types)}")

    # 构建 HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float32,
    }

    if config.use_lora:
        from peft import LoraConfig, get_peft_model

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs,
        )
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if _is_main_process():
            model.print_trainable_parameters()
        if config.gradient_checkpointing:
            model.enable_input_require_grads()
        logger.info(f"LoRA 配置: r={config.lora_r}, alpha={config.lora_alpha}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs,
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"全参数微调: {trainable_params/1e9:.2f}B / {total_params/1e9:.2f}B 参数")

    # 训练参数
    max_steps = config.smoke_max_steps if config.smoke else -1
    num_epochs = 1 if config.smoke else config.num_train_epochs
    # 默认 save_steps 很大，通常只在训练结束后手动 save_model 到 final/。
    # 用户显式传较小 --save-steps 时，Trainer 会额外保存中间 checkpoint。
    save_strategy = "steps" if config.save_steps > 0 else "no"

    # trl 0.29.x 使用 SFTConfig（继承自 TrainingArguments）统一配置
    training_args = TRLSFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=save_strategy,
        eval_steps=config.eval_steps if not config.smoke else 10,
        save_steps=config.save_steps if config.save_steps > 0 else 1,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=False,
        max_steps=max_steps,
        report_to=config.report_to if not config.smoke else "none",
        logging_dir=str(output_path / "logs"),
        remove_unused_columns=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        deepspeed=config.deepspeed,
        disable_tqdm=not _is_main_process(),
        log_on_each_node=False,
        # SFT 特有参数（trl 0.29.x）
        max_length=config.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    # SFT Trainer（trl 0.29.x）
    # 数据集含 messages 字段，trl 会自动用 apply_chat_template 处理
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 训练
    logger.info("开始 SFT 训练...")
    if not _is_main_process():
        # 非主进程禁用 transformers 的 logging 避免干扰进度条
        import transformers
        transformers.utils.logging.set_verbosity_error()
    train_result = trainer.train()

    # 保存最终模型（Trainer 内部已处理分布式，仅 rank 0 实际保存）
    trainer.save_model(str(output_path / "final"))
    if _is_main_process():
        tokenizer.save_pretrained(str(output_path / "final"))
    logger.info(f"最终模型已保存到 {output_path / 'final'}")

    # 最终 eval
    if config.eval_strategy != "no":
        eval_metrics = trainer.evaluate()
        logger.info(f"最终 eval_loss: {eval_metrics.get('eval_loss', 'N/A')}")
    else:
        eval_metrics = {}
        logger.info("eval_strategy=no，跳过最终 eval")

    # 汇总训练指标
    train_metrics = train_result.metrics
    elapsed = time.time() - start_time

    summary = {
        "config": {
            "model": config.model_name_or_path,
            "use_lora": config.use_lora,
            "lora_r": config.lora_r if config.use_lora else None,
            "max_seq_length": config.max_seq_length,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "per_device_eval_batch_size": config.per_device_eval_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "eval_strategy": config.eval_strategy,
            "eval_steps": config.eval_steps,
            "save_steps": config.save_steps,
            "lr_scheduler_type": config.lr_scheduler_type,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "max_grad_norm": config.max_grad_norm,
            "bf16": config.bf16,
            "gradient_checkpointing": config.gradient_checkpointing,
            "seed": config.seed,
            "smoke": config.smoke,
        },
        "data": {
            "total_samples": len(raw_data),
            "train_samples": len(train_data),
            "eval_samples": len(eval_data),
            "train_action_type_distribution": dict(train_types),
            "eval_action_type_distribution": dict(eval_types),
        },
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "timing": {
            "total_seconds": elapsed,
            "total_minutes": elapsed / 60,
        },
    }

    # 保存训练报告（仅主进程写文件，避免多 rank 竞争）
    if _is_main_process():
        report_path = output_path / "training_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"训练报告已保存到 {report_path}")

    # 打印最终摘要
    logger.info("=" * 60)
    logger.info("训练完成摘要")
    logger.info("=" * 60)
    logger.info(f"  train_loss: {train_metrics.get('train_loss', 'N/A')}")
    logger.info(f"  eval_loss: {eval_metrics.get('eval_loss', 'N/A')}")
    logger.info(f"  train_runtime: {train_metrics.get('train_runtime', 0):.1f}s")
    logger.info(f"  total_time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"  samples/sec: {train_metrics.get('train_samples_per_second', 'N/A')}")
    logger.info(f"  global_steps: {trainer.state.global_step}")
    logger.info("=" * 60)

    # Smoke 模式额外验证（仅主进程执行，避免多 rank 重复推理）
    if config.smoke and _is_main_process():
        _smoke_validation(model, tokenizer, raw_data[:5], config)

    return summary


def _smoke_validation(model, tokenizer, samples: list[dict], config: SFTConfig) -> None:
    """Smoke 模式下的快速验证：检查模型能否生成正确格式。"""
    import torch
    from src.reward.action_parser import parse_action

    logger.info("\n" + "=" * 60)
    logger.info("Smoke Validation: 检查模型输出格式")
    logger.info("=" * 60)

    model.eval()
    format_valid_count = 0
    report = {
        "num_samples": len(samples),
        "format_valid_count": 0,
        "format_valid_rate": 0.0,
        "samples": [],
    }

    for i, sample in enumerate(samples):
        messages = sample["messages"]
        prompt_messages = messages[:-1]

        input_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed = parse_action(generated_text, strict=True)

        status = "✅" if parsed.parseable else "❌"
        format_valid_count += 1 if parsed.parseable else 0
        report["samples"].append({
            "index": i,
            "expected": messages[-1]["content"][:200],
            "generated": generated_text[:200],
            "parseable": parsed.parseable,
            "parsed_action_type": parsed.action_type,
            "error_detail": parsed.error_detail,
        })

        logger.info(f"\n  Sample {i}:")
        logger.info(f"    Expected: {messages[-1]['content'][:80]}...")
        logger.info(f"    Generated: {generated_text[:80]}...")
        logger.info(f"    Parsed type: {parsed.action_type}")
        logger.info(f"    Format valid: {status}")

    report["format_valid_count"] = format_valid_count
    report["format_valid_rate"] = format_valid_count / len(samples) if samples else 0.0

    logger.info(f"\n  Format valid rate: {format_valid_count}/{len(samples)} = {report['format_valid_rate']:.1%}")
    logger.info("=" * 60)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "smoke_validation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Smoke validation report saved to {report_path}")


# ============================================================
# CLI 入口
# ============================================================


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(_PROJECT_ROOT) / config_path
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"SFT YAML config must be a mapping: {config_path}")
    return data


def _flatten_sft_yaml(data: dict[str, Any]) -> dict[str, Any]:
    """Map structured YAML sections onto argparse defaults."""
    mapping = {
        ("model", "name_or_path"): "model",
        ("data", "train_path"): "data",
        ("data", "eval_split_ratio"): "eval_split",
        ("output", "dir"): "output",
        ("training", "epochs"): "epochs",
        ("training", "per_device_train_batch_size"): "batch_size",
        ("training", "per_device_eval_batch_size"): "eval_batch_size",
        ("training", "gradient_accumulation_steps"): "grad_accum",
        ("training", "learning_rate"): "lr",
        ("training", "lr_scheduler_type"): "lr_scheduler_type",
        ("training", "warmup_ratio"): "warmup_ratio",
        ("training", "weight_decay"): "weight_decay",
        ("training", "max_grad_norm"): "max_grad_norm",
        ("training", "max_seq_length"): "max_seq_length",
        ("training", "bf16"): "bf16",
        ("training", "gradient_checkpointing"): "gradient_checkpointing",
        ("logging", "logging_steps"): "logging_steps",
        ("logging", "eval_strategy"): "eval_strategy",
        ("logging", "eval_steps"): "eval_steps",
        ("logging", "save_steps"): "save_steps",
        ("logging", "save_total_limit"): "save_total_limit",
        ("logging", "report_to"): "report_to",
        ("runtime", "dataloader_num_workers"): "dataloader_num_workers",
        ("runtime", "deepspeed"): "deepspeed",
        ("runtime", "seed"): "seed",
        ("smoke", "enabled"): "smoke",
        ("smoke", "max_samples"): "smoke_max_samples",
        ("smoke", "max_steps"): "smoke_max_steps",
        ("lora", "enabled"): "lora",
        ("lora", "r"): "lora_r",
        ("lora", "alpha"): "lora_alpha",
        ("lora", "dropout"): "lora_dropout",
        ("lora", "target_modules"): "lora_target_modules",
    }
    defaults: dict[str, Any] = {}
    for (section, key), dest in mapping.items():
        section_data = data.get(section, {})
        if isinstance(section_data, dict) and key in section_data:
            defaults[dest] = section_data[key]
    return defaults


def _default_sft_cli_values() -> dict[str, Any]:
    return {
        "model": _env_str("SFT_MODEL", "models/Qwen3-4B"),
        "data": _env_str("SFT_DATA", "data/sft/sft_train.jsonl"),
        "output": _env_str("SFT_OUTPUT", "outputs/sft_cold_start_4b"),
        "smoke": _env_bool("SFT_SMOKE", False),
        "lora": _env_bool("SFT_LORA", False),
        "epochs": _env_int("SFT_EPOCHS", 3),
        "batch_size": _env_int("SFT_BATCH_SIZE", 4),
        "eval_batch_size": _env_int("SFT_EVAL_BATCH_SIZE", 1),
        "grad_accum": _env_int("SFT_GRAD_ACCUM", 4),
        "lr": _env_float("SFT_LR", 2e-5),
        "lr_scheduler_type": _env_str("SFT_LR_SCHEDULER_TYPE", "cosine"),
        "warmup_ratio": _env_float("SFT_WARMUP_RATIO", 0.05),
        "weight_decay": _env_float("SFT_WEIGHT_DECAY", 0.01),
        "max_grad_norm": _env_float("SFT_MAX_GRAD_NORM", 1.0),
        "max_seq_length": _env_int("SFT_MAX_SEQ_LENGTH", 2560),
        "eval_split": _env_float("SFT_EVAL_SPLIT", 0.05),
        "eval_strategy": _env_str("SFT_EVAL_STRATEGY", "steps"),
        "eval_steps": _env_int("SFT_EVAL_STEPS", 50),
        "save_steps": _env_int("SFT_SAVE_STEPS", 999999),
        "save_total_limit": _env_int("SFT_SAVE_TOTAL_LIMIT", 1),
        "logging_steps": _env_int("SFT_LOGGING_STEPS", 5),
        "report_to": _env_str("SFT_REPORT_TO", "tensorboard"),
        "dataloader_num_workers": _env_int("SFT_DATALOADER_NUM_WORKERS", 4),
        "smoke_max_samples": _env_int("SFT_SMOKE_MAX_SAMPLES", 100),
        "smoke_max_steps": _env_int("SFT_SMOKE_MAX_STEPS", 30),
        "lora_r": _env_int("SFT_LORA_R", 64),
        "lora_alpha": _env_int("SFT_LORA_ALPHA", 128),
        "lora_dropout": _env_float("SFT_LORA_DROPOUT", 0.05),
        "lora_target_modules": _env_str("SFT_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),
        "seed": _env_int("SFT_SEED", 42),
        "bf16": _env_bool("SFT_BF16", True),
        "gradient_checkpointing": _env_bool("SFT_GRADIENT_CHECKPOINTING", True),
        "deepspeed": os.environ.get("SFT_DEEPSPEED"),
    }


def main():
    """命令行入口。"""
    import argparse

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=os.environ.get("SFT_CONFIG"))
    pre_args, _ = pre_parser.parse_known_args()
    defaults = _default_sft_cli_values()
    defaults.update(_flatten_sft_yaml(_load_yaml_config(pre_args.config)))

    parser = argparse.ArgumentParser(description="SFT Cold-Start 训练")
    parser.add_argument(
        "--config",
        default=pre_args.config,
        help="YAML 配置文件；命令行参数会覆盖 YAML",
    )
    parser.add_argument(
        "--model",
        default=defaults["model"],
        help="模型路径或 HuggingFace ID",
    )
    parser.add_argument(
        "--data",
        default=defaults["data"],
        help="SFT 训练数据路径",
    )
    parser.add_argument(
        "--output",
        default=defaults["output"],
        help="输出目录",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=bool(defaults["smoke"]),
        help="Smoke 模式（少量数据快速验证）",
    )
    parser.add_argument("--no-smoke", dest="smoke", action="store_false", help="禁用 smoke 模式")
    parser.add_argument(
        "--lora",
        action="store_true",
        default=bool(defaults["lora"]),
        help="使用 LoRA 训练（显存不足时）",
    )
    parser.add_argument("--no-lora", dest="lora", action="store_false", help="禁用 LoRA")
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults["epochs"],
        help="训练轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults["batch_size"],
        help="每设备 batch size（L20 44GB + Qwen3-4B 推荐 4）",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=defaults["eval_batch_size"],
        help="每设备 eval batch size（Qwen3 大 vocab eval loss 较耗显存，默认 1）",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=defaults["grad_accum"],
        help="梯度累积步数（8卡时推荐 4，effective_batch=4×4×8=128）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=defaults["lr"],
        help="学习率",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        default=defaults["lr_scheduler_type"],
        help="学习率调度器类型",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=defaults["warmup_ratio"],
        help="warmup 比例",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=defaults["weight_decay"],
        help="weight decay",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=defaults["max_grad_norm"],
        help="梯度裁剪阈值",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=defaults["max_seq_length"],
        help="最大序列长度（基于精确 tokenize 分析：p99=2207, max=2478）",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=defaults["eval_split"],
        help="Eval 集比例",
    )
    parser.add_argument(
        "--eval-strategy",
        choices=["steps", "epoch", "no"],
        default=defaults["eval_strategy"],
        help="Eval 策略；显存紧张时可设 no 跳过中途和最终 eval",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=defaults["eval_steps"],
        help="每多少步 eval 一次（eval-strategy=steps 时生效）",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=defaults["save_steps"],
        help="每多少步保存 checkpoint；默认很大，等价于只保存 final/",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=defaults["save_total_limit"],
        help="最多保留 checkpoint 数量",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=defaults["logging_steps"],
        help="每多少步打印一次训练日志",
    )
    parser.add_argument(
        "--report-to",
        default=defaults["report_to"],
        help='Trainer report_to，例如 "tensorboard" 或 "none"',
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=defaults["dataloader_num_workers"],
        help="DataLoader worker 数",
    )
    parser.add_argument(
        "--smoke-max-samples",
        type=int,
        default=defaults["smoke_max_samples"],
        help="smoke 模式最多使用多少样本",
    )
    parser.add_argument(
        "--smoke-max-steps",
        type=int,
        default=defaults["smoke_max_steps"],
        help="smoke 模式最大训练步数",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=defaults["lora_r"],
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=defaults["lora_alpha"],
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=defaults["lora_dropout"],
        help="LoRA dropout",
    )
    parser.add_argument(
        "--lora-target-modules",
        default=defaults["lora_target_modules"],
        help="逗号分隔的 LoRA target modules",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults["seed"],
        help="随机种子",
    )
    parser.add_argument(
        "--bf16",
        dest="bf16",
        action="store_true",
        default=bool(defaults["bf16"]),
        help="启用 bf16",
    )
    parser.add_argument(
        "--no-bf16",
        dest="bf16",
        action="store_false",
        help="禁用 bf16",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=bool(defaults["gradient_checkpointing"]),
        help="启用 gradient checkpointing",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="禁用 gradient checkpointing",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=defaults["deepspeed"],
        help="DeepSpeed 配置文件路径（推荐 configs/ds_zero2.json）",
    )
    args = parser.parse_args()
    if isinstance(args.lora_target_modules, (list, tuple)):
        lora_target_modules = [str(item) for item in args.lora_target_modules]
    else:
        lora_target_modules = [
            item.strip() for item in str(args.lora_target_modules).split(",") if item.strip()
        ]

    config = SFTConfig(
        model_name_or_path=args.model,
        train_data_path=args.data,
        output_dir=args.output,
        smoke=args.smoke,
        use_lora=args.lora,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_seq_length=args.max_seq_length,
        eval_split_ratio=args.eval_split,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        smoke_max_samples=args.smoke_max_samples,
        smoke_max_steps=args.smoke_max_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        deepspeed=args.deepspeed,
    )

    run_sft(config)


if __name__ == "__main__":
    main()
