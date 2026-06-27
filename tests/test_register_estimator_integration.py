"""register_estimator.py monkey-patch 集成测试。

构造最小 fake DataProto，验证：
1. patched_compute_advantage 把 non_tensor_batch 透传给 livemcp_grpo estimator;
2. 端到端 advantages 计算成功且为 finite tensor;
3. 非 livemcp 路径不会调用我们的 estimator。
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

verl_ray = pytest.importorskip("verl.trainer.ppo.ray_trainer")


class _FakeDataProto:
    def __init__(self, batch, non_tensor_batch):
        self.batch = dict(batch)
        self.non_tensor_batch = non_tensor_batch


@pytest.fixture()
def patched_module():
    mod = importlib.import_module("verl.trainer.ppo.ray_trainer")
    original = mod.compute_advantage
    from src.training.register_estimator import register_livemcp_estimator
    ok = register_livemcp_estimator({"use_livemcp": True})
    assert ok
    assert mod.compute_advantage is not original
    yield mod
    mod.compute_advantage = original


def _make_data(bsz=9):
    rewards = torch.tensor(
        [0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    ).unsqueeze(-1)
    response_mask = torch.ones(bsz, 1)
    levels = np.array(
        ["none"] * 3 + ["mild"] * 3 + ["strong"] * 3, dtype=object
    )
    group_id = np.array(["task_a"] * bsz, dtype=object)
    uids = np.array([f"task_a___{lv}" for lv in levels], dtype=object)
    return _FakeDataProto(
        batch={
            "token_level_rewards": rewards,
            "response_mask": response_mask,
        },
        non_tensor_batch={
            "perturbation_level": levels,
            "group_id": group_id,
            "uid": uids,
        },
    )


def test_patch_routes_livemcp_to_estimator(patched_module):
    data = _make_data()
    out = patched_module.compute_advantage(data, "livemcp_grpo")
    assert "advantages" in out.batch
    advantages = out.batch["advantages"]
    assert advantages.shape == (9, 1)
    assert torch.all(torch.isfinite(advantages))
    assert advantages.abs().sum().item() > 1e-3


def test_patch_passes_non_tensor_batch(patched_module, monkeypatch):
    captured = {}

    def fake_estimator(**kwargs):
        captured["non_tensor_batch"] = kwargs.get("non_tensor_batch")
        captured["index"] = kwargs.get("index")
        n = kwargs["token_level_rewards"].shape[0]
        zero = torch.zeros(n, 1)
        return zero, zero

    from verl.trainer.ppo import core_algos
    monkeypatch.setattr(
        core_algos, "get_adv_estimator_fn", lambda name: fake_estimator
    )

    data = _make_data()
    patched_module.compute_advantage(data, "livemcp_grpo")

    assert captured["non_tensor_batch"] is not None
    assert "perturbation_level" in captured["non_tensor_batch"]
    assert "group_id" in captured["non_tensor_batch"]
    assert captured["index"] is not None
    assert len(captured["index"]) == 9


def test_patch_does_not_break_other_estimators(patched_module, monkeypatch):
    """patch 后非 livemcp 路径必须走原 verl 实现，不会调用我们的 estimator。"""
    sentinel_calls = {"n": 0}

    def fake_estimator(**kwargs):
        sentinel_calls["n"] += 1
        n = kwargs["token_level_rewards"].shape[0]
        return torch.zeros(n, 1), torch.zeros(n, 1)

    from verl.trainer.ppo import core_algos
    monkeypatch.setattr(
        core_algos, "get_adv_estimator_fn", lambda name: fake_estimator
    )

    data = _make_data()
    # livemcp_grpo 应路由到自定义 estimator
    patched_module.compute_advantage(data, "livemcp_grpo")
    assert sentinel_calls["n"] == 1

    # 非 livemcp_grpo 走原 verl 路径，fake 不应被再次调用
    data2 = _make_data()
    try:
        patched_module.compute_advantage(data2, "grpo")
    except Exception:
        pass  # 原路径是否成功取决于 verl 内部依赖，这里只关心是否绕过了 fake
    assert sentinel_calls["n"] == 1, (
        "非 livemcp_grpo 路径不应触发自定义 estimator"
    )


def test_estimator_consumes_perturbation_level(patched_module):
    data = _make_data()
    out = patched_module.compute_advantage(data, "livemcp_grpo")
    advantages = out.batch["advantages"].squeeze(-1)
    assert abs(advantages.sum().item()) < 1e-4
