"""ReplayMCPExecutor — 确定性 replay 环境执行器。

给定 EpisodeSeed + model action，通过 oracle_trace 匹配确定性返回 tool_output。
不走真实 MCP server，纯 replay。

核心逻辑：
  1. 接收 model 的 parsed action
  2. 与当前 step 的 oracle 做匹配（tool_name + arguments）
  3. 匹配成功 → 释放 oracle 的 replay_observation
  4. 匹配失败 → 返回 error observation（模拟 MCP server 拒绝）
  5. 维护 step pointer，支持 reset

设计约束：
  - 无状态副作用（不修改 EpisodeSeed）
  - 线程安全（每个 executor 实例独立）
  - 支持 parallel_tool_call
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.data.episode_schema import ActionType, EpisodeSeed, OracleStep


@dataclass
class ExecutionResult:
    """单步执行结果。"""
    observation: str                    # 返回给模型的 observation
    observations: list[str] = field(default_factory=list)  # parallel 时的多个 observation
    matched: bool = False               # 是否与 oracle 匹配
    done: bool = False                  # episode 是否结束
    step_idx: int = 0                   # 当前 step 序号
    match_detail: str = ""              # 匹配细节（诊断用）
    oracle_action_type: str = ""        # oracle 期望的 action type


@dataclass
class MatchConfig:
    """匹配配置。"""
    # 工具名匹配时是否使用 name_map
    name_map: dict[str, str] = field(default_factory=dict)
    # 参数值匹配时是否使用 enum_map
    enum_map: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    # 是否要求参数值精确匹配（False 时只检查 key 覆盖）
    strict_values: bool = False
    # 工具名匹配是否大小写敏感
    case_sensitive_name: bool = False


class ReplayMCPExecutor:
    """确定性 replay 执行器。

    Usage:
        executor = ReplayMCPExecutor(episode_seed)
        result = executor.step(parsed_action)
        if result.done:
            trajectory_reward = ...
    """

    def __init__(
        self,
        episode: EpisodeSeed,
        match_config: Optional[MatchConfig] = None,
        error_template: str = "Error: tool execution failed. {detail}",
    ):
        self.episode = episode
        self.match_config = match_config or MatchConfig()
        self.error_template = error_template
        self._step_ptr = 0
        self._done = False
        self._history: list[ExecutionResult] = []

    @property
    def current_step(self) -> int:
        return self._step_ptr

    @property
    def done(self) -> bool:
        return self._done

    @property
    def total_steps(self) -> int:
        return len(self.episode.oracle_trace)

    @property
    def history(self) -> list[ExecutionResult]:
        return self._history

    def reset(self) -> None:
        """重置执行器状态。"""
        self._step_ptr = 0
        self._done = False
        self._history = []

    def step(self, action: dict[str, Any]) -> ExecutionResult:
        """执行一步。

        Args:
            action: 解析后的 action dict，格式：
                {
                    "action_type": "tool_call" | "final_answer" | ...,
                    "tool_calls": [{"name": ..., "arguments": ...}],  # tool_call 时
                    "content": "...",  # final_answer/report_error/ask_clarification 时
                }

        Returns:
            ExecutionResult。
        """
        if self._done:
            result = ExecutionResult(
                observation="",
                done=True,
                step_idx=self._step_ptr,
                match_detail="episode already done",
            )
            return result

        # 超出 oracle trace 长度 → 强制结束
        if self._step_ptr >= self.total_steps:
            self._done = True
            result = ExecutionResult(
                observation="",
                done=True,
                step_idx=self._step_ptr,
                match_detail="exceeded oracle trace length",
            )
            self._history.append(result)
            return result

        oracle_step = self.episode.oracle_trace[self._step_ptr]
        model_action_type = action.get("action_type", "")

        # 终止类 action（final_answer / report_error / ask_clarification）→ 直接结束
        if model_action_type in ("final_answer", "report_error", "ask_clarification"):
            self._done = True
            matched = (model_action_type == oracle_step.action_type)
            result = ExecutionResult(
                observation="",
                matched=matched,
                done=True,
                step_idx=self._step_ptr,
                match_detail=f"terminal action: model={model_action_type}, oracle={oracle_step.action_type}",
                oracle_action_type=oracle_step.action_type,
            )
            self._history.append(result)
            return result

        # tool_call 类 action
        if model_action_type in ("tool_call", "parallel_tool_call"):
            return self._handle_tool_call(action, oracle_step)

        # 未知 action type → 结束
        self._done = True
        result = ExecutionResult(
            observation="",
            matched=False,
            done=True,
            step_idx=self._step_ptr,
            match_detail=f"unknown action_type: {model_action_type}",
            oracle_action_type=oracle_step.action_type,
        )
        self._history.append(result)
        return result

    def _handle_tool_call(self, action: dict[str, Any], oracle_step: OracleStep) -> ExecutionResult:
        """处理 tool_call 类 action。"""
        model_calls = action.get("tool_calls", [])

        # oracle 期望的也是 tool_call 类
        if oracle_step.action_type == ActionType.TOOL_CALL.value:
            return self._match_single_tool_call(model_calls, oracle_step)
        elif oracle_step.action_type == ActionType.PARALLEL_TOOL_CALL.value:
            return self._match_parallel_tool_call(model_calls, oracle_step)
        else:
            # oracle 期望非 tool_call（如 final_answer），但模型调了工具
            # 返回 error observation，不推进 step pointer，但标记 mismatch
            obs = self.error_template.format(
                detail=f"Expected {oracle_step.action_type}, got tool_call"
            )
            # 不推进 step，给模型一次纠错机会？
            # 设计决策：mismatch 时直接结束，避免无限循环
            self._done = True
            result = ExecutionResult(
                observation=obs,
                matched=False,
                done=True,
                step_idx=self._step_ptr,
                match_detail=f"action_type mismatch: model=tool_call, oracle={oracle_step.action_type}",
                oracle_action_type=oracle_step.action_type,
            )
            self._history.append(result)
            return result

    def _match_single_tool_call(
        self,
        model_calls: list[dict],
        oracle_step: OracleStep,
    ) -> ExecutionResult:
        """匹配单个 tool_call。"""
        # 取模型的第一个 call（忽略多余的）
        if not model_calls:
            obs = self.error_template.format(detail="No tool call provided")
            self._done = True
            result = ExecutionResult(
                observation=obs,
                matched=False,
                done=True,
                step_idx=self._step_ptr,
                match_detail="empty tool_calls list",
                oracle_action_type=oracle_step.action_type,
            )
            self._history.append(result)
            return result

        model_call = model_calls[0]
        matched = self._call_matches_oracle(
            model_call,
            oracle_step.tool_name,
            oracle_step.arguments,
        )

        if matched:
            # 释放 replay observation，推进 step
            obs = oracle_step.replay_observation
            self._step_ptr += 1
            # 如果已到最后一步，标记 done
            done = self._step_ptr >= self.total_steps
            if done:
                self._done = True
            result = ExecutionResult(
                observation=obs,
                matched=True,
                done=done,
                step_idx=self._step_ptr - 1,
                match_detail="exact match",
                oracle_action_type=oracle_step.action_type,
            )
        else:
            # 不匹配 → 返回 error，结束
            detail = self._mismatch_detail(model_call, oracle_step.tool_name, oracle_step.arguments)
            obs = self.error_template.format(detail=detail)
            self._done = True
            result = ExecutionResult(
                observation=obs,
                matched=False,
                done=True,
                step_idx=self._step_ptr,
                match_detail=detail,
                oracle_action_type=oracle_step.action_type,
            )

        self._history.append(result)
        return result

    def _match_parallel_tool_call(
        self,
        model_calls: list[dict],
        oracle_step: OracleStep,
    ) -> ExecutionResult:
        """匹配 parallel tool_call。"""
        oracle_calls = oracle_step.calls
        oracle_observations = oracle_step.replay_observations

        if not model_calls:
            obs = self.error_template.format(detail="No tool calls provided for parallel step")
            self._done = True
            result = ExecutionResult(
                observation=obs,
                matched=False,
                done=True,
                step_idx=self._step_ptr,
                match_detail="empty tool_calls for parallel step",
                oracle_action_type=oracle_step.action_type,
            )
            self._history.append(result)
            return result

        # 按 match_mode 匹配
        if oracle_step.match_mode == "ordered":
            matched = self._match_ordered(model_calls, oracle_calls)
        else:
            matched = self._match_set(model_calls, oracle_calls)

        if matched:
            self._step_ptr += 1
            done = self._step_ptr >= self.total_steps
            if done:
                self._done = True
            # 拼接所有 observations
            combined_obs = "\n---\n".join(oracle_observations) if oracle_observations else ""
            result = ExecutionResult(
                observation=combined_obs,
                observations=oracle_observations,
                matched=True,
                done=done,
                step_idx=self._step_ptr - 1,
                match_detail="parallel match",
                oracle_action_type=oracle_step.action_type,
            )
        else:
            detail = (
                f"parallel mismatch: model has {len(model_calls)} calls, "
                f"oracle has {len(oracle_calls)} calls"
            )
            obs = self.error_template.format(detail=detail)
            self._done = True
            result = ExecutionResult(
                observation=obs,
                matched=False,
                done=True,
                step_idx=self._step_ptr,
                match_detail=detail,
                oracle_action_type=oracle_step.action_type,
            )

        self._history.append(result)
        return result

    def _match_ordered(self, model_calls: list[dict], oracle_calls: list[dict]) -> bool:
        """顺序匹配 parallel calls。"""
        if len(model_calls) != len(oracle_calls):
            return False
        for mc, oc in zip(model_calls, oracle_calls):
            if not self._call_matches_oracle(mc, oc["tool_name"], oc.get("arguments", {})):
                return False
        return True

    def _match_set(self, model_calls: list[dict], oracle_calls: list[dict]) -> bool:
        """无序匹配 parallel calls（multiset）。"""
        if len(model_calls) != len(oracle_calls):
            return False

        remaining_oracle = list(range(len(oracle_calls)))
        for mc in model_calls:
            found = False
            for i, oc_idx in enumerate(remaining_oracle):
                oc = oracle_calls[oc_idx]
                if self._call_matches_oracle(mc, oc["tool_name"], oc.get("arguments", {})):
                    remaining_oracle.pop(i)
                    found = True
                    break
            if not found:
                return False
        return True

    def _call_matches_oracle(
        self,
        model_call: dict,
        oracle_name: str,
        oracle_args: dict,
    ) -> bool:
        """检查单个 model call 是否匹配 oracle。"""
        model_name = model_call.get("name", "")
        model_args = model_call.get("arguments", {})

        # 工具名匹配（通过 name_map）
        canonical_model = self.match_config.name_map.get(model_name, model_name)
        canonical_oracle = self.match_config.name_map.get(oracle_name, oracle_name)

        if self.match_config.case_sensitive_name:
            if canonical_model != canonical_oracle:
                return False
        else:
            if canonical_model.lower() != canonical_oracle.lower():
                return False

        # 参数匹配
        if self.match_config.strict_values:
            return self._args_match_strict(model_args, oracle_args, oracle_name)
        else:
            # 宽松模式：只检查 key 覆盖率 >= 100%
            return self._args_match_keys(model_args, oracle_args)

    def _args_match_keys(self, model_args: dict, oracle_args: dict) -> bool:
        """宽松匹配：所有 oracle required keys 都存在于 model args 中。"""
        if not oracle_args:
            return True
        if not isinstance(model_args, dict):
            return False
        return set(oracle_args.keys()).issubset(set(model_args.keys()))

    def _args_match_strict(
        self,
        model_args: dict,
        oracle_args: dict,
        tool_name: str,
    ) -> bool:
        """严格匹配：key 完全一致 + value 匹配（考虑 enum_map）。"""
        if not isinstance(model_args, dict):
            return not oracle_args
        if set(model_args.keys()) != set(oracle_args.keys()):
            return False

        for key, oracle_val in oracle_args.items():
            model_val = model_args.get(key)
            # enum_map 映射
            model_val = self._map_enum(tool_name, key, model_val)
            if not self._value_eq(model_val, oracle_val):
                return False
        return True

    def _map_enum(self, tool_name: str, param_name: str, value: Any) -> Any:
        """通过 enum_map 映射值。"""
        if not self.match_config.enum_map:
            return value
        tool_map = self.match_config.enum_map.get(tool_name, {})
        param_map = tool_map.get(param_name, {})
        if isinstance(value, str) and value in param_map:
            return param_map[value]
        return value

    @staticmethod
    def _value_eq(a: Any, b: Any) -> bool:
        """值相等比较（宽松类型）。"""
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        # 字符串
        if isinstance(a, str) and isinstance(b, str):
            return a.strip().lower() == b.strip().lower()
        # 数值
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-9
        # bool
        if isinstance(a, bool) and isinstance(b, bool):
            return a == b
        # list（无序）
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            remaining = list(b)
            for item in a:
                found = False
                for i, r_item in enumerate(remaining):
                    if ReplayMCPExecutor._value_eq(item, r_item):
                        remaining.pop(i)
                        found = True
                        break
                if not found:
                    return False
            return True
        # dict
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(ReplayMCPExecutor._value_eq(a[k], b[k]) for k in a)
        # fallback: 字符串化比较
        return str(a).strip().lower() == str(b).strip().lower()

    def _mismatch_detail(self, model_call: dict, oracle_name: str, oracle_args: dict) -> str:
        """生成 mismatch 诊断信息。"""
        model_name = model_call.get("name", "")
        model_args = model_call.get("arguments", {})

        canonical_model = self.match_config.name_map.get(model_name, model_name)
        canonical_oracle = self.match_config.name_map.get(oracle_name, oracle_name)

        if canonical_model.lower() != canonical_oracle.lower():
            return f"tool_name mismatch: model={model_name}, expected={oracle_name}"

        # name 匹配但 args 不匹配
        missing_keys = set(oracle_args.keys()) - set(model_args.keys()) if isinstance(model_args, dict) else set(oracle_args.keys())
        if missing_keys:
            return f"missing argument keys: {sorted(missing_keys)}"

        return "argument values mismatch"
