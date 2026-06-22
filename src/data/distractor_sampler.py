"""Distractor Tool 采样器。

从 ToolACE 工具池中为每个样本采样语义相近的干扰工具。

设计要点（mcp_tools_rl_project_plan.md §7）：
  - 不与 GT tool 同名
  - 不破坏 GT oracle
  - 尽量语义相近（同领域），而不是随机噪声工具
  - GT tool 与 distractor 使用同一扰动强度分布
  - distractor 中至少一部分必须与 GT 一样经过 perturbation

采样策略：
  1. 领域匹配：优先从同领域工具中采样
  2. 参数结构相似：优先选参数数量/类型相近的工具
  3. 名称相似：优先选名称有共同词根的工具

同强度扰动（mcp_tools_rl_project_plan.md §7 组合矩阵）：
  - sample_with_perturbation() 方法对 distractor 施加与 GT 相同级别的扰动
  - 避免 shortcut：模型不能通过"被改名的就是答案"来区分 GT 和 distractor
"""

import copy
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

# 延迟导入避免循环依赖
_SchemaPerturber = None


def _get_perturber_class():
    """延迟导入 SchemaPerturber。"""
    global _SchemaPerturber
    if _SchemaPerturber is None:
        from src.envs.schema_perturber import SchemaPerturber
        _SchemaPerturber = SchemaPerturber
    return _SchemaPerturber


# 领域关键词映射（与 compute_full_stats.py 一致）
DOMAIN_KEYWORDS = {
    "finance": ["stock", "market", "trading", "finance", "crypto", "sec", "invest", "bank", "payment", "price", "currency"],
    "travel": ["travel", "flight", "hotel", "booking", "trip", "airport", "reservation"],
    "weather": ["weather", "climate", "temperature", "forecast"],
    "file": ["file", "directory", "folder", "download", "upload", "storage", "document"],
    "message": ["message", "email", "sms", "chat", "notification", "mail", "send"],
    "search": ["search", "query", "find", "lookup", "get", "retrieve", "fetch"],
    "math": ["math", "calculate", "compute", "convert", "unit", "formula"],
    "social": ["social", "post", "tweet", "comment", "like", "follow", "user", "profile"],
    "code": ["code", "compile", "debug", "git", "repo", "api", "function", "program"],
    "health": ["health", "medical", "doctor", "patient", "drug", "symptom"],
}


@dataclass
class ToolEntry:
    """工具池中的一条工具记录。"""
    name: str
    schema: dict
    domain: str
    param_count: int
    param_types: set = field(default_factory=set)
    name_words: set = field(default_factory=set)  # 工具名拆分后的词集合


@dataclass
class DistractorConfig:
    """Distractor 采样配置。"""
    min_distractors: int = 2
    max_distractors: int = 5
    domain_weight: float = 0.5  # 领域匹配权重
    param_weight: float = 0.3  # 参数结构相似权重
    name_weight: float = 0.2  # 名称相似权重
    seed: int = 42


class DistractorSampler:
    """Distractor 工具采样器。

    从工具池中为每个样本采样语义相近的干扰工具。

    Usage:
        sampler = DistractorSampler.from_decision_steps("data/toolace/processed/decision_steps.jsonl")
        distractors = sampler.sample(gt_tools=["get_weather"], num_distractors=3)
    """

    def __init__(self, config: Optional[DistractorConfig] = None):
        self.config = config or DistractorConfig()
        self.rng = random.Random(self.config.seed)
        self.tool_pool: dict[str, ToolEntry] = {}  # name -> ToolEntry
        self.domain_index: dict[str, list[str]] = defaultdict(list)  # domain -> [tool_names]
        self._built = False

    @classmethod
    def from_decision_steps(
        cls,
        data_path: str | Path,
        config: Optional[DistractorConfig] = None,
    ) -> "DistractorSampler":
        """从 decision_steps.jsonl 构建工具池。

        Args:
            data_path: decision_steps.jsonl 路径。
            config: 采样配置。

        Returns:
            初始化好的 DistractorSampler。
        """
        sampler = cls(config=config)
        sampler.build_pool(data_path)
        return sampler

    def build_pool(self, data_path: str | Path) -> None:
        """从 decision_steps.jsonl 构建工具池。"""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        logger.info(f"构建工具池: {data_path}")
        seen_names = set()

        with open(data_path) as f:
            for line in f:
                d = json.loads(line)
                for tool in d.get("tools", []):
                    name = tool.get("name", "")
                    if not name or name in seen_names:
                        continue
                    seen_names.add(name)

                    entry = self._make_entry(tool)
                    self.tool_pool[name] = entry
                    self.domain_index[entry.domain].append(name)

        self._built = True
        logger.info(
            f"工具池构建完成: {len(self.tool_pool)} 个工具, "
            f"{len(self.domain_index)} 个领域"
        )
        for domain, tools in sorted(self.domain_index.items(), key=lambda x: -len(x[1])):
            logger.debug(f"  {domain}: {len(tools)} 个工具")

    def sample(
        self,
        gt_tool_names: list[str],
        num_distractors: Optional[int] = None,
        exclude_names: Optional[set[str]] = None,
    ) -> list[dict]:
        """为给定的 GT 工具采样 distractor 工具。

        Args:
            gt_tool_names: ground truth 工具名列表。
            num_distractors: 采样数量。None 时从 config 范围内随机。
            exclude_names: 额外排除的工具名集合。

        Returns:
            distractor 工具 schema 列表。
        """
        if not self._built:
            raise RuntimeError("工具池未构建，请先调用 build_pool()")

        if num_distractors is None:
            num_distractors = self.rng.randint(
                self.config.min_distractors,
                self.config.max_distractors,
            )

        # 排除 GT 工具和额外排除的工具
        exclude = set(gt_tool_names)
        if exclude_names:
            exclude.update(exclude_names)

        # 计算候选工具的相似度分数
        candidates = []
        for name, entry in self.tool_pool.items():
            if name in exclude:
                continue
            score = self._compute_similarity(gt_tool_names, entry)
            candidates.append((name, score))

        if not candidates:
            logger.warning(f"无可用 distractor 候选（GT: {gt_tool_names}）")
            return []

        # 按分数排序，从 top-K 中随机采样（避免总是选最相似的）
        candidates.sort(key=lambda x: -x[1])
        top_k = min(len(candidates), num_distractors * 3)
        top_candidates = candidates[:top_k]

        # 从 top-K 中随机选 num_distractors 个
        selected = self.rng.sample(
            top_candidates,
            min(num_distractors, len(top_candidates)),
        )

        # 返回工具 schema
        result = []
        for name, score in selected:
            schema = copy.deepcopy(self.tool_pool[name].schema)
            result.append(schema)

        return result

    def sample_with_perturbation(
        self,
        gt_tool_names: list[str],
        perturbation_level: str = "none",
        num_distractors: Optional[int] = None,
        exclude_names: Optional[set[str]] = None,
        perturbation_seed: Optional[int] = None,
    ) -> tuple[list[dict], dict]:
        """采样 distractor 并对其施加与 GT 同强度的扰动。

mcp_tools_rl_project_plan.md §7 要求：GT tool 与 distractor 使用同一扰动强度分布，
        避免模型通过"被改名的就是答案"来 shortcut。

        Args:
            gt_tool_names: ground truth 工具名列表。
            perturbation_level: 扰动级别 ("none" / "mild" / "strong")。
            num_distractors: 采样数量。
            exclude_names: 额外排除的工具名集合。
            perturbation_seed: 扰动随机种子。

        Returns:
            (perturbed_distractor_schemas, perturbation_info)
            perturbation_info 包含 name_map 和 enum_map（用于诊断）。
        """
        # 先采样原始 distractor
        raw_distractors = self.sample(
            gt_tool_names=gt_tool_names,
            num_distractors=num_distractors,
            exclude_names=exclude_names,
        )

        if not raw_distractors or perturbation_level == "none":
            return raw_distractors, {"name_map": {}, "enum_map": {}, "level": "none"}

        # 对 distractor 施加同级别扰动
        SchemaPerturber = _get_perturber_class()
        seed = perturbation_seed or (self.config.seed + 1000)
        perturber = SchemaPerturber(seed=seed)

        try:
            perturbed = perturber.perturb(raw_distractors, level=perturbation_level)
            info = {
                "name_map": dict(perturber.name_map),
                "enum_map": copy.deepcopy(perturber.enum_map),
                "level": perturbation_level,
                "num_perturbed": len(perturbed),
            }
            return perturbed, info
        except (ValueError, AssertionError) as e:
            logger.debug(f"Distractor 扰动失败 (level={perturbation_level}): {e}")
            # 扰动失败时返回原始 distractor
            return raw_distractors, {"name_map": {}, "enum_map": {}, "level": "none", "error": str(e)}

    def sample_with_metadata(
        self,
        gt_tool_names: list[str],
        num_distractors: Optional[int] = None,
        exclude_names: Optional[set[str]] = None,
    ) -> tuple[list[dict], dict]:
        """采样 distractor 并返回元数据（用于日志和诊断）。

        Returns:
            (distractor_schemas, metadata_dict)
        """
        if num_distractors is None:
            num_distractors = self.rng.randint(
                self.config.min_distractors,
                self.config.max_distractors,
            )

        exclude = set(gt_tool_names)
        if exclude_names:
            exclude.update(exclude_names)

        candidates = []
        for name, entry in self.tool_pool.items():
            if name in exclude:
                continue
            score = self._compute_similarity(gt_tool_names, entry)
            candidates.append((name, score, entry.domain))

        if not candidates:
            return [], {"num_candidates": 0}

        candidates.sort(key=lambda x: -x[1])
        top_k = min(len(candidates), num_distractors * 3)
        top_candidates = candidates[:top_k]

        selected = self.rng.sample(
            top_candidates,
            min(num_distractors, len(top_candidates)),
        )

        schemas = [copy.deepcopy(self.tool_pool[name].schema) for name, _, _ in selected]

        # 元数据
        gt_domains = set()
        for gt_name in gt_tool_names:
            if gt_name in self.tool_pool:
                gt_domains.add(self.tool_pool[gt_name].domain)

        metadata = {
            "num_candidates": len(candidates),
            "num_selected": len(selected),
            "gt_domains": list(gt_domains),
            "distractor_names": [name for name, _, _ in selected],
            "distractor_domains": [domain for _, _, domain in selected],
            "similarity_scores": [score for _, score, _ in selected],
            "domain_overlap_ratio": sum(
                1 for _, _, d in selected if d in gt_domains
            ) / max(1, len(selected)),
        }

        return schemas, metadata

    def _compute_similarity(self, gt_tool_names: list[str], candidate: ToolEntry) -> float:
        """计算候选工具与 GT 工具集的相似度分数。

        综合三个维度：
          - 领域匹配（同领域得分高）
          - 参数结构相似（参数数量和类型相近）
          - 名称相似（共同词根）
        """
        if not gt_tool_names:
            return 0.0

        # 获取 GT 工具的信息
        gt_entries = [self.tool_pool[n] for n in gt_tool_names if n in self.tool_pool]
        if not gt_entries:
            return self.rng.random() * 0.1  # 无法比较时给小随机分

        # 1. 领域匹配
        gt_domains = {e.domain for e in gt_entries}
        domain_score = 1.0 if candidate.domain in gt_domains else 0.0
        # "other" 域的工具给一个小分（避免 other 域工具永远不被选）
        if candidate.domain == "other" and "other" in gt_domains:
            domain_score = 0.5

        # 2. 参数结构相似
        avg_gt_params = sum(e.param_count for e in gt_entries) / len(gt_entries)
        param_diff = abs(candidate.param_count - avg_gt_params)
        param_score = max(0.0, 1.0 - param_diff / 5.0)  # 差 5 个参数以上得 0

        # 参数类型重合
        gt_types = set()
        for e in gt_entries:
            gt_types.update(e.param_types)
        if gt_types and candidate.param_types:
            type_overlap = len(gt_types & candidate.param_types) / max(
                len(gt_types | candidate.param_types), 1
            )
            param_score = 0.6 * param_score + 0.4 * type_overlap

        # 3. 名称相似（共同词根）
        gt_words = set()
        for e in gt_entries:
            gt_words.update(e.name_words)
        if gt_words and candidate.name_words:
            name_overlap = len(gt_words & candidate.name_words) / max(
                len(gt_words | candidate.name_words), 1
            )
        else:
            name_overlap = 0.0
        name_score = name_overlap

        # 加权综合
        total = (
            self.config.domain_weight * domain_score
            + self.config.param_weight * param_score
            + self.config.name_weight * name_score
        )

        # 加入少量随机性避免总是选同一批
        total += self.rng.random() * 0.05

        return total

    def _make_entry(self, tool: dict) -> ToolEntry:
        """从 tool schema 构建 ToolEntry。"""
        name = tool.get("name", "")
        params = tool.get("parameters", {}).get("properties", {})
        param_count = len(params)
        param_types = set()
        for p_spec in params.values():
            if isinstance(p_spec, dict):
                ptype = p_spec.get("type", "")
                if isinstance(ptype, str) and ptype:
                    param_types.add(ptype)

        domain = self._classify_domain(name, tool.get("description", ""))
        name_words = self._extract_name_words(name)

        return ToolEntry(
            name=name,
            schema=tool,
            domain=domain,
            param_count=param_count,
            param_types=param_types,
            name_words=name_words,
        )

    @staticmethod
    def _classify_domain(name: str, description: str) -> str:
        """根据工具名和描述分类领域。"""
        text = f"{name} {description}".lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return domain
        return "other"

    @staticmethod
    def _extract_name_words(name: str) -> set[str]:
        """从工具名中提取词集合。"""
        # camelCase → snake_case
        snake = re.sub(
            r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])',
            '_', name
        ).lower()
        words = set(w for w in re.split(r'[_\s\-]+', snake) if len(w) > 1)
        return words

    def get_pool_stats(self) -> dict:
        """返回工具池统计信息。"""
        return {
            "total_tools": len(self.tool_pool),
            "domains": {d: len(tools) for d, tools in self.domain_index.items()},
            "avg_params": sum(e.param_count for e in self.tool_pool.values()) / max(1, len(self.tool_pool)),
        }
