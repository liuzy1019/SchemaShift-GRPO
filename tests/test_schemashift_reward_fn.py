"""Tests for the verl custom reward entrypoint."""

import json

from src.reward.schemashift_reward_fn import compute_score


def test_compute_score_returns_only_scalar_or_string_diagnostics():
    ground_truth = {
        "oracle_actions": [
            {
                "action_type": "tool_call",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "arguments": {"city": "Beijing", "unit": "celsius"},
                    }
                ],
            }
        ],
        "episode_type": "single_step",
    }
    output = (
        '<tool_call>{"name": "get_weather", '
        '"arguments": {"city": "Beijing", "unit": "celsius"}}</tool_call>'
    )

    result = compute_score(
        data_source="schemashift",
        solution_str=output,
        ground_truth=json.dumps(ground_truth),
        extra_info={"perturbation_level": "none", "scenario_type": "single_step"},
    )

    assert "components" not in result
    assert result["score"] > 1.0
    assert result["exact_success"] == 1.0
    assert result["component_format"] == 1.0
    assert all(
        isinstance(value, (float, str))
        for value in result.values()
    )


def test_compute_score_returns_fixed_keys_for_mixed_outcomes():
    tool_gt = {
        "oracle_actions": [
            {
                "action_type": "tool_call",
                "tool_calls": [{"name": "get_weather", "arguments": {"city": "Beijing"}}],
            }
        ],
        "episode_type": "call_only",
    }
    final_gt = {
        "oracle_actions": [
            {
                "action_type": "final_answer",
                "final_answer": "The answer is 42.",
            }
        ],
        "episode_type": "no_tool",
    }

    results = [
        compute_score(
            data_source="schemashift",
            solution_str='<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>',
            ground_truth=json.dumps(tool_gt),
            extra_info={},
        ),
        compute_score(
            data_source="schemashift",
            solution_str="<final_answer>The answer is 42.</final_answer>",
            ground_truth=json.dumps(final_gt),
            extra_info={},
        ),
        compute_score(
            data_source="schemashift",
            solution_str="",
            ground_truth="{not-json",
            extra_info={},
        ),
    ]

    key_sets = [set(result) for result in results]
    assert key_sets[0] == key_sets[1] == key_sets[2]
    assert all(
        isinstance(value, (float, str))
        for result in results
        for value in result.values()
    )
