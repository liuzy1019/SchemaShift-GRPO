#!/usr/bin/env python3
"""Check for packages that are known to conflict with SchemaShift-GRPO."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import subprocess
import sys

from packaging.version import Version


EXPECTED_EXACT = {
    "accelerate": "1.13.0",
    "datasets": "4.8.4",
    "deepspeed": "0.19.2",
    "compressed-tensors": "0.11.0",
    "flashinfer-python": "0.6.4",
    "huggingface-hub": "0.36.2",
    "hydra-core": "1.3.3",
    "einops": "0.8.0",
    "lm-format-enforcer": "0.11.3",
    "numpy": "1.26.4",
    "omegaconf": "2.3.0",
    "packaging": "25.0",
    "pandas": "2.2.3",
    "peft": "0.18.1",
    "protobuf": "5.29.6",
    "pyarrow": "23.0.1",
    "pydantic": "2.12.5",
    "pyyaml": "6.0.3",
    "pytest": "9.1.0",
    "ray": "2.54.1",
    "safetensors": "0.7.0",
    "scipy": "1.13.0",
    "tensordict": "0.10.0",
    "tokenizers": "0.22.2",
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "torchvision": "0.23.0",
    "transformers": "4.57.6",
    "trl": "0.29.1",
    "vllm": "0.11.0",
    "wandb": "0.26.0",
    "xformers": "0.0.32.post1",
    "xgrammar": "0.1.25",
}

KNOWN_CONFLICT_PACKAGES = {
    "fastvideo": "pins old transformers/accelerate/peft/huggingface_hub versions",
    "hpsv2": "pins protobuf<4 and pytest==7.2.0",
    "pytest-split": "pytest-split 0.8.0 requires pytest<8",
    "decord": "video-only dependency that is unsupported on this platform",
}

FLASH_ATTN_EXPECTED = "2.7.3"


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def run_pip_check() -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return result.returncode, output


def uninstall_known_conflicts(packages: list[str]) -> None:
    if not packages:
        return
    print("Removing known conflicting packages:", ", ".join(packages))
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", *packages],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="uninstall known SchemaShift-incompatible packages before checking",
    )
    parser.add_argument(
        "--fix-only",
        action="store_true",
        help="only uninstall known SchemaShift-incompatible packages, then exit",
    )
    args = parser.parse_args()

    installed_conflicts = [
        package for package in KNOWN_CONFLICT_PACKAGES if installed_version(package) is not None
    ]
    if args.fix or args.fix_only:
        uninstall_known_conflicts(installed_conflicts)
        if args.fix_only:
            return 0
        installed_conflicts = [
            package for package in KNOWN_CONFLICT_PACKAGES if installed_version(package) is not None
        ]

    failures: list[str] = []
    warnings: list[str] = []

    for package, reason in KNOWN_CONFLICT_PACKAGES.items():
        version = installed_version(package)
        if version is not None:
            failures.append(f"{package}=={version} is installed: {reason}")

    for package, expected in EXPECTED_EXACT.items():
        version = installed_version(package)
        if version is None:
            failures.append(f"{package} is missing; expected {expected}")
        else:
            # 比较 base version，忽略 local segment（如 +cu128）
            installed_base = str(Version(version).base_version)
            expected_base = str(Version(expected).base_version)
            if installed_base != expected_base:
                failures.append(f"{package}=={version}; expected {expected}")

    flash_attn = installed_version("flash-attn")
    if flash_attn is None:
        warnings.append("flash-attn is not installed; training may need xformers/SDPA fallback")
    else:
        installed_base = str(Version(flash_attn).base_version)
        expected_base = str(Version(FLASH_ATTN_EXPECTED).base_version)
        if installed_base != expected_base:
            failures.append(f"flash-attn=={flash_attn}; expected {FLASH_ATTN_EXPECTED}")

    pip_check_code, pip_check_output = run_pip_check()
    if pip_check_code != 0:
        failures.append("pip check failed:\n" + pip_check_output)

    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")

    if failures:
        print("Dependency conflicts found:")
        for item in failures:
            print(f"  - {item}")
        print("\nFor a project-owned environment, run:")
        print("  python scripts/check_dependency_conflicts.py --fix")
        return 1

    print("Dependency check passed: SchemaShift-GRPO stack is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
