"""Reference advantage implementations for unit testing.

These functions extract the core stratified-advantage logic from
livemcp_grpo_estimator.py into pure functions that can be tested
independently of verl/Ray infrastructure.

Production path: livemcp_grpo_estimator.compute_livemcp_grpo_advantage()
Test path:      livemcp_advantage.compute_livemcp_advantages()
"""

from __future__ import annotations

from collections import defaultdict

import torch


def compute_standard_grpo_advantages(
    rewards: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Standard GRPO: z-score within batch.

    If all rewards are identical, returns zeros.
    """
    if rewards.numel() < 2:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std < epsilon:
        return torch.zeros_like(rewards)
    return (rewards - mean) / std


def compute_livemcp_advantages(
    rewards: torch.Tensor,
    levels: list[str],
    scenario_types: list[str] | None = None,
    beta: float = 0.25,
    epsilon: float = 1e-6,
    min_stratum_size: int = 3,
) -> torch.Tensor:
    """2D stratified advantage: per-(level, scenario) z-score + beta * global_z.

    Args:
        rewards: (n,) tensor of per-sample scalar rewards.
        levels: perturbation levels, same length as rewards.
        scenario_types: optional scenario types. If None, treated as single_step.
        beta: weight of global residual (0 = pure stratum, 1 = pure global).
        epsilon: std floor to avoid division by zero.
        min_stratum_size: minimum samples in a stratum for z-score with std.

    Returns:
        (n,) tensor of advantages.
    """
    if len(rewards) != len(levels):
        raise ValueError(
            f"length mismatch: rewards={len(rewards)}, levels={len(levels)}"
        )

    if scenario_types is None:
        scenario_types = ["single_step"] * len(rewards)

    if len(scenario_types) != len(rewards):
        raise ValueError(
            f"length mismatch: rewards={len(rewards)}, "
            f"scenario_types={len(scenario_types)}"
        )

    bsz = len(rewards)
    advantages = torch.zeros(bsz, device=rewards.device)

    # Group by task_id — when not provided, treat all as one group
    # (test callers pass per-task batches directly)
    task_ids = ["default"] * bsz
    task2indices = defaultdict(list)
    for i, tid in enumerate(task_ids):
        task2indices[tid].append(i)

    with torch.no_grad():
        for tid, indices in task2indices.items():
            idx_tensor = torch.tensor(indices, device=rewards.device)
            group_scores = rewards[idx_tensor]
            group_levels = [levels[i] for i in indices]
            group_scenarios = [scenario_types[i] for i in indices]
            n_group = len(group_scores)

            # Per-(level, scenario) strata
            stratum2local = defaultdict(list)
            scenario2local = defaultdict(list)
            for local_i, (level, scenario) in enumerate(
                zip(group_levels, group_scenarios)
            ):
                stratum2local[(level, scenario)].append(local_i)
                scenario2local[scenario].append(local_i)

            strat_advs = torch.zeros(n_group, device=rewards.device)

            for stratum_key, loc_indices in stratum2local.items():
                loc_tensor = torch.tensor(loc_indices, device=rewards.device)
                stratum_scores = group_scores[loc_tensor]
                n_stratum = len(loc_indices)

                if n_stratum >= min_stratum_size:
                    s_mean = stratum_scores.mean()
                    s_std = stratum_scores.std(unbiased=False).clamp(min=epsilon)
                    strat_advs[loc_tensor] = (stratum_scores - s_mean) / s_std
                elif n_stratum == 2:
                    strat_advs[loc_tensor] = stratum_scores - stratum_scores.mean()
                elif n_stratum == 1:
                    # Fallback to scenario-level
                    scenario = stratum_key[1]
                    sc_indices = scenario2local[scenario]
                    if len(sc_indices) >= 2:
                        sc_tensor = torch.tensor(sc_indices, device=rewards.device)
                        sc_scores = group_scores[sc_tensor]
                        sc_mean = sc_scores.mean()
                        if len(sc_indices) >= min_stratum_size:
                            sc_std = sc_scores.std(unbiased=False).clamp(min=epsilon)
                            strat_advs[loc_tensor] = (
                                stratum_scores - sc_mean
                            ) / sc_std
                        else:
                            strat_advs[loc_tensor] = stratum_scores - sc_mean
                    else:
                        # Fallback to group-level
                        g_mean = group_scores.mean()
                        if n_group >= min_stratum_size:
                            g_std = group_scores.std(unbiased=False).clamp(
                                min=epsilon
                            )
                            strat_advs[loc_tensor] = (
                                stratum_scores - g_mean
                            ) / g_std
                        else:
                            strat_advs[loc_tensor] = stratum_scores - g_mean

            # Global z-score residual
            group_mean = group_scores.mean()
            if n_group >= 2:
                group_std = group_scores.std(unbiased=False).clamp(min=epsilon)
                global_z = (group_scores - group_mean) / group_std
            else:
                global_z = group_scores - group_mean

            advantages[idx_tensor] = strat_advs + beta * global_z

    return advantages


def compute_stratified_advantage(
    rewards: torch.Tensor,
    levels: list[str],
    scenario_types: list[str],
    beta: float = 0.25,
    epsilon: float = 1e-6,
    min_stratum_size: int = 3,
) -> torch.Tensor:
    """Alias for compute_livemcp_advantages with explicit scenario_types required.

    This is the 2D stratified version — both perturbation_level and scenario_type
    are used for stratum grouping.
    """
    return compute_livemcp_advantages(
        rewards=rewards,
        levels=levels,
        scenario_types=scenario_types,
        beta=beta,
        epsilon=epsilon,
        min_stratum_size=min_stratum_size,
    )


__all__ = [
    "compute_livemcp_advantages",
    "compute_standard_grpo_advantages",
    "compute_stratified_advantage",
]
