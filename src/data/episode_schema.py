"""Episode Seed Schema 定义。

定义 EpisodeSeed 的正式数据结构，作为整个训练框架的核心数据格式。
所有从 Toucan/ToolACE 构建的训练数据最终都转换为此格式。

设计依据: mcp_tools_rl_project_plan.md §7

EpisodeSeed 是 replay online rollout 的输入单元：
  - 对模型不可见：oracle_trace
  - 对模型可见：initial_messages + tools_snapshot
  - 环境使用：oracle_trace 驱动 replay 匹配和 observation 释放
  - 验证器使用：oracle_trace + verifier config 计算 reward
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ============================================================
# 枚举定义
# ============================================================


class ActionType(str, Enum):
    """动作类型。"""
    TOOL_CALL = "tool_call"
    PARALLEL_TOOL_CALL = "parallel_tool_call"
    FINAL_ANSWER = "final_answer"
    ASK_CLARIFICATION = "ask_clarification"
    REPORT_ERROR = "report_error"


class EpisodeType(str, Enum):
    """Episode 类型（§8.1）。"""
    CALL_ONLY = "call_only"              # user + tools -> tool_call -> done
    CALL_THEN_FINAL = "call_then_final"  # tool_call -> observation -> final_answer
    CALL_THEN_CALL = "call_then_call"    # tool_call -> observation -> next tool_call
    NO_TOOL = "no_tool"                  # user + tools -> final/clarify
    ERROR_OUTPUT = "error_output"        # tool_call -> error -> report/fallback


class PerturbationTag(str, Enum):
    """扰动标签（§8.2）。"""
    SCHEMA_SHIFT = "schema_shift"
    DISTRACTOR = "distractor"
    OUTPUT_NOISE = "output_noise"
    MISSING_INFO = "missing_info"
    IRRELEVANT_TOOL = "irrelevant_tool"


class VerifierType(str, Enum):
    """验证器类型。"""
    EXACT = "exact"          # 精确匹配
    ENTITY = "entity"        # 实体级匹配
    JUDGE = "judge"          # LLM judge（后续阶段）


class DataSource(str, Enum):
    """数据来源。"""
    TOUCAN = "toucan"
    TOOLACE = "toolace"
    APIGEN_MT = "apigen_mt"
    SYNTHETIC = "synthetic"


# ============================================================
# Oracle Step 定义
# ============================================================


@dataclass
class VerifierConfig:
    """单步验证器配置。

    控制该步骤如何被验证。
    """
    type: str = "exact"                          # VerifierType
    name_map: dict[str, str] = field(default_factory=dict)   # perturbed -> canonical
    enum_map: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    # 是否允许参数值的宽松匹配
    lenient_values: bool = False
    # final_answer 匹配时的 expected entities
    expected_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"type": self.type}
        if self.name_map:
            d["name_map"] = self.name_map
        if self.enum_map:
            d["enum_map"] = self.enum_map
        if self.lenient_values:
            d["lenient_values"] = True
        if self.expected_entities:
            d["expected_entities"] = self.expected_entities
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VerifierConfig":
        return cls(
            type=d.get("type", "exact"),
            name_map=d.get("name_map", {}),
            enum_map=d.get("enum_map", {}),
            lenient_values=d.get("lenient_values", False),
            expected_entities=d.get("expected_entities", []),
        )


@dataclass
class OracleStep:
    """单步 oracle action（§7）。

    表示 oracle trace 中的一个决策步骤。
    """
    step: int                                    # 步骤序号（从 0 开始）
    action_type: str                             # ActionType value
    # tool_call 字段
    tool_name: str = ""                          # 工具名
    arguments: dict[str, Any] = field(default_factory=dict)  # 参数
    # parallel_tool_call 字段（§7.1）
    calls: list[dict[str, Any]] = field(default_factory=list)  # [{"tool_name": ..., "arguments": ...}]
    match_mode: str = "set"                      # "set" | "ordered"
    # replay observation
    replay_observation: str = ""                 # 单步 replay 观测
    replay_observations: list[str] = field(default_factory=list)  # 并行步 replay 观测列表
    # final_answer / ask_clarification / report_error 字段
    expected_content: str = ""                   # 期望的文本内容
    expected_entities: list[str] = field(default_factory=list)  # 期望包含的实体
    # 验证器配置
    verifier: VerifierConfig = field(default_factory=VerifierConfig)

    def to_dict(self) -> dict:
        """序列化为 dict。"""
        d: dict[str, Any] = {
            "step": self.step,
            "action_type": self.action_type,
        }

        if self.action_type == ActionType.TOOL_CALL.value:
            d["tool_name"] = self.tool_name
            d["arguments"] = self.arguments
            if self.replay_observation:
                d["replay_observation"] = self.replay_observation

        elif self.action_type == ActionType.PARALLEL_TOOL_CALL.value:
            d["calls"] = self.calls
            d["match_mode"] = self.match_mode
            if self.replay_observations:
                d["replay_observations"] = self.replay_observations

        elif self.action_type == ActionType.FINAL_ANSWER.value:
            if self.expected_content:
                d["expected_content"] = self.expected_content
            if self.expected_entities:
                d["expected_entities"] = self.expected_entities

        elif self.action_type in (ActionType.ASK_CLARIFICATION.value, ActionType.REPORT_ERROR.value):
            if self.expected_content:
                d["expected_content"] = self.expected_content

        d["verifier"] = self.verifier.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OracleStep":
        """从 dict 反序列化。"""
        verifier = VerifierConfig.from_dict(d.get("verifier", {}))
        return cls(
            step=d["step"],
            action_type=d["action_type"],
            tool_name=d.get("tool_name", ""),
            arguments=d.get("arguments", {}),
            calls=d.get("calls", []),
            match_mode=d.get("match_mode", "set"),
            replay_observation=d.get("replay_observation", ""),
            replay_observations=d.get("replay_observations", []),
            expected_content=d.get("expected_content", ""),
            expected_entities=d.get("expected_entities", []),
            verifier=verifier,
        )

    def validate(self) -> list[str]:
        """验证 step 的合法性，返回错误列表。"""
        errors = []

        if self.step < 0:
            errors.append(f"step must be >= 0, got {self.step}")

        valid_types = [t.value for t in ActionType]
        if self.action_type not in valid_types:
            errors.append(f"invalid action_type: {self.action_type}")

        if self.action_type == ActionType.TOOL_CALL.value:
            if not self.tool_name:
                errors.append("tool_call step must have tool_name")

        if self.action_type == ActionType.PARALLEL_TOOL_CALL.value:
            if not self.calls:
                errors.append("parallel_tool_call step must have non-empty calls")
            for i, call in enumerate(self.calls):
                if "tool_name" not in call:
                    errors.append(f"parallel call[{i}] missing tool_name")

        return errors


# ============================================================
# EpisodeSeed 定义
# ============================================================


@dataclass
class SplitKeys:
    """用于数据划分的 key（训练/评测分离）。"""
    server: str = ""           # MCP server 名
    domain: str = ""           # 领域（weather, finance, etc.）
    tool_names: list[str] = field(default_factory=list)  # 涉及的工具名

    def to_dict(self) -> dict:
        d = {}
        if self.server:
            d["server"] = self.server
        if self.domain:
            d["domain"] = self.domain
        if self.tool_names:
            d["tool_names"] = self.tool_names
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SplitKeys":
        return cls(
            server=d.get("server", ""),
            domain=d.get("domain", ""),
            tool_names=d.get("tool_names", []),
        )


@dataclass
class EpisodeSeed:
    """Episode Seed — 训练框架的核心数据单元。

    一个 EpisodeSeed 包含 replay online rollout 所需的全部信息：
      - initial_messages: 模型可见的初始对话上下文
      - tools_snapshot: 模型可见的工具 schema 列表
      - oracle_trace: 环境和验证器使用的正确行动序列（模型不可见）

    设计原则:
      - oracle_trace 对模型完全隐藏
      - tools_snapshot 可能包含 distractor 工具
      - 支持 schema_shift 后的 name_map/enum_map 还原
    """
    # 唯一标识
    episode_id: str
    # 数据来源
    source: str                                  # DataSource value
    # 场景标签
    episode_type: str                            # EpisodeType value
    scenario_tags: list[str] = field(default_factory=list)  # PerturbationTag values
    # MCP server 信息
    mcp_servers: list[str] = field(default_factory=list)
    # 模型可见内容
    tools_snapshot: list[dict] = field(default_factory=list)  # 工具 schema 列表
    initial_messages: list[dict] = field(default_factory=list)  # 初始对话
    # Oracle trace（模型不可见）
    oracle_trace: list[OracleStep] = field(default_factory=list)
    # 约束
    max_turns: int = 3                           # 最大决策轮数
    # 数据划分 key
    split_keys: SplitKeys = field(default_factory=SplitKeys)
    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为 JSON-compatible dict。"""
        return {
            "episode_id": self.episode_id,
            "source": self.source,
            "episode_type": self.episode_type,
            "scenario_tags": self.scenario_tags,
            "mcp_servers": self.mcp_servers,
            "tools_snapshot": self.tools_snapshot,
            "initial_messages": self.initial_messages,
            "oracle_trace": [step.to_dict() for step in self.oracle_trace],
            "max_turns": self.max_turns,
            "split_keys": self.split_keys.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeSeed":
        """从 dict 反序列化。"""
        oracle_trace = [OracleStep.from_dict(s) for s in d.get("oracle_trace", [])]
        split_keys = SplitKeys.from_dict(d.get("split_keys", {}))
        return cls(
            episode_id=d["episode_id"],
            source=d.get("source", "toucan"),
            episode_type=d.get("episode_type", "call_only"),
            scenario_tags=d.get("scenario_tags", []),
            mcp_servers=d.get("mcp_servers", []),
            tools_snapshot=d.get("tools_snapshot", []),
            initial_messages=d.get("initial_messages", []),
            oracle_trace=oracle_trace,
            max_turns=d.get("max_turns", 3),
            split_keys=split_keys,
            metadata=d.get("metadata", {}),
        )

    def validate(self) -> list[str]:
        """验证 EpisodeSeed 的完整性和合法性。

        Returns:
            错误列表。空列表表示验证通过。
        """
        errors = []

        # 必要字段
        if not self.episode_id:
            errors.append("episode_id is required")
        if not self.source:
            errors.append("source is required")
        if not self.episode_type:
            errors.append("episode_type is required")

        # episode_type 合法性
        valid_types = [t.value for t in EpisodeType]
        if self.episode_type and self.episode_type not in valid_types:
            errors.append(f"invalid episode_type: {self.episode_type}")

        # tools_snapshot 非空（除了 no_tool 场景也需要有工具列表）
        if not self.tools_snapshot:
            errors.append("tools_snapshot must not be empty")

        # initial_messages 至少有 user message
        if not self.initial_messages:
            errors.append("initial_messages must not be empty")
        elif not any(m.get("role") == "user" for m in self.initial_messages):
            errors.append("initial_messages must contain at least one user message")

        # oracle_trace 非空
        if not self.oracle_trace:
            errors.append("oracle_trace must not be empty")

        # oracle_trace 步骤验证
        for i, step in enumerate(self.oracle_trace):
            step_errors = step.validate()
            for e in step_errors:
                errors.append(f"oracle_trace[{i}]: {e}")

            # 步骤序号连续性
            if step.step != i:
                errors.append(f"oracle_trace[{i}].step should be {i}, got {step.step}")

        # max_turns 合理性
        if self.max_turns < 1:
            errors.append(f"max_turns must be >= 1, got {self.max_turns}")

        # episode_type 与 oracle_trace 一致性
        if self.oracle_trace:
            self._validate_type_trace_consistency(errors)

        return errors

    def _validate_type_trace_consistency(self, errors: list[str]) -> None:
        """验证 episode_type 与 oracle_trace 的一致性。"""
        trace_types = [s.action_type for s in self.oracle_trace]

        if self.episode_type == EpisodeType.CALL_ONLY.value:
            # 应该只有 tool_call / parallel_tool_call
            for t in trace_types:
                if t not in (ActionType.TOOL_CALL.value, ActionType.PARALLEL_TOOL_CALL.value):
                    errors.append(
                        f"call_only episode should only have tool_call steps, "
                        f"found {t}"
                    )
                    break

        elif self.episode_type == EpisodeType.NO_TOOL.value:
            # 不应该有 tool_call
            for t in trace_types:
                if t in (ActionType.TOOL_CALL.value, ActionType.PARALLEL_TOOL_CALL.value):
                    errors.append(
                        f"no_tool episode should not have tool_call steps, "
                        f"found {t}"
                    )
                    break

    @property
    def decision_turns(self) -> int:
        """Oracle trace 中的决策轮数。"""
        return len(self.oracle_trace)

    @property
    def has_parallel_calls(self) -> bool:
        """是否包含并行调用。"""
        return any(
            s.action_type == ActionType.PARALLEL_TOOL_CALL.value
            for s in self.oracle_trace
        )

    @property
    def tool_names_used(self) -> list[str]:
        """Oracle trace 中使用的所有工具名。"""
        names = []
        for step in self.oracle_trace:
            if step.action_type == ActionType.TOOL_CALL.value:
                names.append(step.tool_name)
            elif step.action_type == ActionType.PARALLEL_TOOL_CALL.value:
                for call in step.calls:
                    names.append(call.get("tool_name", ""))
        return names

    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_jsonl_line(self) -> str:
        """序列化为单行 JSON（用于 JSONL 文件）。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ============================================================
# 批量 I/O 工具
# ============================================================


def save_episodes(episodes: list[EpisodeSeed], path: str | Path) -> int:
    """保存 episode seeds 到 JSONL 文件。

    Args:
        episodes: EpisodeSeed 列表。
        path: 输出文件路径。

    Returns:
        写入的条数。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(ep.to_jsonl_line() + "\n")
            count += 1

    return count


def load_episodes(path: str | Path) -> list[EpisodeSeed]:
    """从 JSONL 文件加载 episode seeds。

    Args:
        path: JSONL 文件路径。

    Returns:
        EpisodeSeed 列表。
    """
    path = Path(path)
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                episodes.append(EpisodeSeed.from_dict(d))
    return episodes


def validate_episodes(episodes: list[EpisodeSeed]) -> dict[str, Any]:
    """批量验证 episode seeds。

    Returns:
        验证报告 dict。
    """
    total = len(episodes)
    valid = 0
    invalid = 0
    all_errors: list[dict] = []

    for ep in episodes:
        errors = ep.validate()
        if errors:
            invalid += 1
            all_errors.append({
                "episode_id": ep.episode_id,
                "errors": errors,
            })
        else:
            valid += 1

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "valid_rate": valid / total if total > 0 else 0.0,
        "errors": all_errors[:50],  # 只保留前 50 个错误样本
    }
