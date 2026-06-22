"""验证 SFT checkpoint 的生成格式质量。"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.reward.action_parser import ParsedAction, parse_action


def load_samples(path: Path, max_samples: int, seed: int) -> list[dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:max_samples]


def same_arguments(predicted: ParsedAction, expected: ParsedAction) -> bool:
    if expected.action_type != "tool_call":
        return True
    if len(predicted.tool_calls) != len(expected.tool_calls):
        return False
    for pred_call, exp_call in zip(predicted.tool_calls, expected.tool_calls):
        if pred_call.get("arguments", {}) != exp_call.get("arguments", {}):
            return False
    return True


def same_tool_names(predicted: ParsedAction, expected: ParsedAction) -> bool:
    if expected.action_type != "tool_call":
        return True
    pred_names = [call.get("name") for call in predicted.tool_calls]
    exp_names = [call.get("name") for call in expected.tool_calls]
    return pred_names == exp_names


def format_preview(text: str, limit: int = 240) -> str:
    text = " ".join(text.strip().split())
    return text[:limit] + ("..." if len(text) > limit else "")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data)
    model_path = Path(args.model)
    samples = load_samples(data_path, args.max_samples, args.seed)

    logger.info(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    counters = {
        "total": 0,
        "tool_call_total": 0,
        "parseable": 0,
        "action_type_match": 0,
        "tool_name_match": 0,
        "arguments_match": 0,
        "exact_match": 0,
    }
    examples = []

    for idx, sample in enumerate(samples):
        messages = sample["messages"]
        prompt_messages = messages[:-1]
        expected_text = messages[-1]["content"]
        expected = parse_action(expected_text, strict=True)

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted = parse_action(generated_text, strict=True)

        is_tool_call = expected.action_type == "tool_call"
        tool_name_match = same_tool_names(predicted, expected)
        arguments_match = same_arguments(predicted, expected)

        counters["total"] += 1
        counters["tool_call_total"] += int(is_tool_call)
        counters["parseable"] += int(predicted.parseable)
        counters["action_type_match"] += int(predicted.action_type == expected.action_type)
        counters["tool_name_match"] += int(is_tool_call and tool_name_match)
        counters["arguments_match"] += int(is_tool_call and arguments_match)
        counters["exact_match"] += int(predicted.parseable and generated_text.strip() == expected_text.strip())

        record = {
            "index": idx,
            "metadata": sample.get("metadata", {}),
            "expected_action_type": expected.action_type,
            "predicted_action_type": predicted.action_type,
            "parseable": predicted.parseable,
            "tool_name_match": tool_name_match if is_tool_call else None,
            "arguments_match": arguments_match if is_tool_call else None,
            "expected": format_preview(expected_text),
            "generated": format_preview(generated_text),
            "error_detail": predicted.error_detail,
        }
        examples.append(record)
        logger.info(
            f"[{idx + 1}/{len(samples)}] parse={predicted.parseable} "
            f"type={record['predicted_action_type']} "
            f"tool={record['tool_name_match']} args={record['arguments_match']}"
        )

    total = max(counters["total"], 1)
    tool_total = max(counters["tool_call_total"], 1)
    rates = {
        "parseable_rate": counters["parseable"] / total,
        "action_type_rate": counters["action_type_match"] / total,
        "tool_name_rate": counters["tool_name_match"] / tool_total,
        "arguments_rate": counters["arguments_match"] / tool_total,
        "exact_rate": counters["exact_match"] / total,
    }
    report = {
        "model": str(model_path),
        "data": str(data_path),
        "max_samples": args.max_samples,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "counters": counters,
        "rates": rates,
        "examples": examples,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"报告已保存: {output_path}")

    logger.info("=" * 60)
    for key, value in rates.items():
        logger.info(f"{key}: {value:.2%}")
    logger.info("=" * 60)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 SFT checkpoint 生成质量")
    # 路径
    parser.add_argument("--model", default="outputs/sft_cold_start_4b/final")
    parser.add_argument("--data", default="data/sft/sft_train.jsonl")
    parser.add_argument("--output", default="outputs/sft_cold_start_4b/generation_report.json")
    # 生成
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
