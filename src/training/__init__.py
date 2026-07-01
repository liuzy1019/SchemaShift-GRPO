# Heavy modules are lazy-loaded so that lightweight consumers (oval_reward_fn,
# test_runner, etc.) can import livemcp_hyperparams without triggering the full
# torch / verl / transformers import chain.
#
# Consumers that need the full training stack should import submodules directly:
#     from src.training.advantage_core import compute_standard_grpo_advantages
# rather than relying on __init__.py re-exports.


def __getattr__(name: str):
    """Lazy-import heavy training submodules on first access."""
    _LAZY_MAP = {
        "compute_livemcp_grpo_advantage": ".livemcp_grpo_estimator",
        "register_livemcp_estimator": ".register_estimator",
        "_normalize_livemcp_non_tensor_batch": ".register_estimator",
        "TrainerConfig": ".trainer_config",
        "ExperimentManager": ".trainer_config",
        "resolve_gpu_info": ".trainer_config",
        "print_config_summary": ".trainer_config",
        "update_lambda_safe": ".hooks",
        "normalize_livemcp_non_tensor_batch": ".hooks",
        "compute_per_group_stratified_advantages": ".advantage_core",
        "compute_livemcp_advantages": ".advantage_core",
        "compute_stratified_advantage": ".advantage_core",
        "compute_standard_grpo_advantages": ".advantage_core",
    }
    if name in _LAZY_MAP:
        import importlib
        mod = importlib.import_module(_LAZY_MAP[name], __package__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compute_livemcp_grpo_advantage",
    "register_livemcp_estimator",
    "TrainerConfig",
    "ExperimentManager",
    "resolve_gpu_info",
    "print_config_summary",
    "_normalize_livemcp_non_tensor_batch",
    "update_lambda_safe",
    "normalize_livemcp_non_tensor_batch",
    "compute_per_group_stratified_advantages",
    "compute_livemcp_advantages",
    "compute_stratified_advantage",
    "compute_standard_grpo_advantages",
]
