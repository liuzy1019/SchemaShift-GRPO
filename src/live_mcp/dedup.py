"""Jaccard-based deduplication for LLM teacher generated tasks.

PROVE uses Jaccard similarity threshold of 0.70 on oracle call sequences
to remove near-duplicate training tasks. This improves data diversity.
"""

from __future__ import annotations

from typing import Iterable

from src.live_mcp.types import LiveTask


def jaccard_similarity(a: LiveTask, b: LiveTask) -> float:
    """Jaccard similarity between two tasks' oracle tool call signatures.

    Each task is represented as the set of (tool_name, frozenset(key=value))
    pairs from its oracle program.  This captures both *which* tools are
    called and *with what arguments*, so tasks targeting different entities
    or using different parameter values are correctly distinguished.

    Returns a float in [0.0, 1.0].
    """
    sig_a = _call_signatures(a)
    sig_b = _call_signatures(b)

    if not sig_a and not sig_b:
        return 1.0  # both empty → treat as duplicate
    if not sig_a or not sig_b:
        return 0.0

    intersection = sig_a & sig_b
    union = sig_a | sig_b
    return len(intersection) / len(union)


def dedup_tasks(
    tasks: Iterable[LiveTask],
    threshold: float = 0.70,
) -> list[LiveTask]:
    """Greedy deduplication: keep first occurrence, discard subsequent similar tasks.

    Only compares tasks within the same domain — cross-domain Jaccard is
    always 0 (different tool sets) so cross-domain comparison is wasteful
    and never triggers dedup.

    Preserves insertion order.  For each task, if any previously kept task
    *in the same domain* has Jaccard similarity >= *threshold*, it is skipped.
    """
    kept: list[LiveTask] = []
    for task in tasks:
        task_domain = task.target_servers[0] if task.target_servers else ""
        is_dup = False
        for kept_task in kept:
            kept_domain = kept_task.target_servers[0] if kept_task.target_servers else ""
            if task_domain != kept_domain:
                continue
            if jaccard_similarity(task, kept_task) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(task)
    return kept


# ── helpers ──────────────────────────────────────────────────────────


def _call_signatures(task: LiveTask) -> set[tuple[str, frozenset[str]]]:
    """Build set of (tool_name, frozenset(key=value)) pairs from oracle calls.

    Includes argument *values* so that tasks targeting different entities
    or using different parameter values produce distinct signatures.
    This allows Jaccard similarity to correctly distinguish:
      - Different entities: transfer(from=savings, to=checking) ≠ transfer(from=checking, to=savings)
      - Same entity:   two tasks with identical tool calls → true duplicate → dedup
    """
    sigs: set[tuple[str, frozenset[str]]] = set()
    for call in task.oracle_program.calls:
        parts: list[str] = []
        if call.arguments:
            for k, v in sorted(call.arguments.items()):
                parts.append(f"{k}={v}")
        sigs.add((call.tool_name, frozenset(parts)))
    return sigs
