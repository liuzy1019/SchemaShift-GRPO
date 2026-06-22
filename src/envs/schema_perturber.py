"""
Schema 扰动引擎。

对 BFCL Tool Definition 做受控扰动，生成语义等价但表面形式不同的 schema 变体。
支持三档扰动强度（轻度/中度/重度）和四种扰动类型（工具名/描述/参数/枚举值）。
"""

import copy
import random
import re
from typing import Optional
from loguru import logger


# ──────────────────────────────────────────────
# 内置同义词映射表
# ──────────────────────────────────────────────

DEFAULT_SYNONYM_MAP = {
    # ── 工具名 → 同义词 ──
    "tool_name": {
        # 动词前缀
        "search": ["find", "query", "lookup", "retrieve", "fetch"],
        "get": ["retrieve", "fetch", "obtain", "acquire"],
        "find": ["search", "locate", "discover", "lookup"],
        "create": ["add", "new", "make", "generate", "build"],
        "add": ["create", "insert", "append", "attach"],
        "delete": ["remove", "erase", "clear", "drop", "purge"],
        "remove": ["delete", "erase", "clear", "strip"],
        "update": ["modify", "change", "edit", "set", "adjust", "refresh"],
        "modify": ["update", "change", "edit", "alter", "revise"],
        "book": ["reserve", "order", "schedule", "arrange"],
        "cancel": ["void", "terminate", "revoke", "stop"],
        "list": ["show", "display", "enumerate", "view"],
        "show": ["display", "list", "view", "reveal"],
        "set": ["configure", "adjust", "define", "specify"],
        "check": ["verify", "validate", "inspect", "examine", "review"],
        "send": ["transmit", "dispatch", "forward", "submit"],
        "read": ["load", "open", "retrieve", "view"],
        "write": ["save", "store", "record", "dump"],
        "start": ["begin", "launch", "initiate", "activate"],
        "stop": ["halt", "terminate", "end", "cease"],
        "calculate": ["compute", "evaluate", "determine", "estimate"],
        "convert": ["transform", "translate", "transcode", "change"],
        "filter": ["refine", "narrow", "sift"],
        "sort": ["order", "arrange", "organize", "rank"],
        "merge": ["combine", "unite", "fuse", "join"],
        "split": ["divide", "separate", "partition"],
        "enable": ["activate", "allow", "permit", "turn_on"],
        "disable": ["deactivate", "block", "forbid", "turn_off"],
    },
    # ── 描述用词 → 同义词 ──
    # 用于描述文本中的关键词替换
    "description": {
        "search": "lookup",
        "find": "locate",
        "retrieve": "fetch",
        "available": "accessible",
        "return": "provide",
        "information": "data",
        "details": "particulars",
        "specified": "given",
        "current": "present",
        "specific": "particular",
        "valid": "legitimate",
        "related": "associated",
        "existing": "current",
        "between": "among",
        "using": "via",
        "based on": "according to",
        "perform": "execute",
    },
    # ── 枚举值 → 同义词 ──
    "enum": {
        "economy": ["standard", "coach", "regular", "main_cabin"],
        "business": ["executive", "premium", "business_class"],
        "first": ["first_class", "luxury", "premier", "deluxe"],
        "pending": ["awaiting", "unresolved", "open"],
        "confirmed": ["verified", "approved", "validated"],
        "cancelled": ["voided", "terminated", "revoked"],
        "completed": ["finished", "done", "resolved", "processed"],
        "active": ["enabled", "live", "operational"],
        "inactive": ["disabled", "offline", "suspended"],
        "success": ["ok", "successful", "passed"],
        "failure": ["error", "failed", "unsuccessful"],
        "high": ["elevated", "major", "critical"],
        "medium": ["moderate", "average", "normal"],
        "low": ["minor", "minimal", "trivial"],
    },
}


# ──────────────────────────────────────────────
# 扰动强度级别
# ──────────────────────────────────────────────

PerturbationLevel = str
LEVEL_NONE: PerturbationLevel = "none"
LEVEL_MILD: PerturbationLevel = "mild"
LEVEL_MODERATE: PerturbationLevel = "moderate"
LEVEL_STRONG: PerturbationLevel = "strong"

ALL_LEVELS = [LEVEL_NONE, LEVEL_MILD, LEVEL_MODERATE, LEVEL_STRONG]
TRAINING_LEVELS = [LEVEL_NONE, LEVEL_MILD, LEVEL_STRONG]


# ──────────────────────────────────────────────
# 扰动器
# ──────────────────────────────────────────────


class SchemaPerturber:
    """Schema 扰动引擎。

    对 BFCL 的 tool definition 做受控扰动，维护扰动映射记录。
    所有操作是确定性的（给定 seed），不依赖外部模型。

    Example:
        >>> perturber = SchemaPerturber(seed=42)
        >>> functions = [{"name": "search_flights", "description": "Search flights", ...}]
        >>> perturbed = perturber.perturb(functions, level="mild")
        >>> perturber.name_map
        {"find_flights": "search_flights"}
    """

    def __init__(
        self,
        synonym_map: Optional[dict] = None,
        seed: int = 42,
    ):
        """初始化扰动器。

        Args:
            synonym_map: 自定义同义词映射表。None 使用内置默认表。
            seed: 随机种子，确保扰动可复现。
        """
        self.synonym_map = synonym_map or copy.deepcopy(DEFAULT_SYNONYM_MAP)
        self.rng = random.Random(seed)

        # 扰动映射记录：{perturbed_name: original_name}
        self.name_map: dict[str, str] = {}
        # 枚举值映射：{func_name: {param_name: {perturbed_enum_value: original_enum_value}}}
        self.enum_map: dict[str, dict[str, dict[str, str]]] = {}

        logger.info(
            f"SchemaPerturber 初始化完成，seed={seed}"
        )

    def perturb(
        self,
        functions: list[dict],
        level: PerturbationLevel,
    ) -> list[dict]:
        """对 tool definition 列表应用扰动。

        支持两种格式：
        - flat: {"name": ..., "description": ..., "parameters": ...}
        - wrapped (OpenAI/Toucan): {"type": "function", "function": {"name": ..., ...}}

        Args:
            functions: 原始 tool definition 列表。
            level: 扰动强度。可选: "none", "mild", "moderate", "strong"。

        Returns:
            扰动后的 tool definition 列表。

        Raises:
            ValueError: 不支持的扰动强度。
            AssertionError: 扰动安全性检查失败。
        """
        if level == LEVEL_NONE:
            return copy.deepcopy(functions)

        if level not in ALL_LEVELS:
            raise ValueError(f"不支持的扰动强度: {level}，可选: {ALL_LEVELS}")

        perturbed = copy.deepcopy(functions)
        types_to_apply = self._get_types_for_level(level)

        for item in perturbed:
            # 支持 wrapped 和 flat 两种格式
            func = item.get("function", item) if "function" in item else item
            original_name = func.get("name", "")
            if not original_name:
                continue

            if "tool_name" in types_to_apply:
                self._perturb_tool_name(func)

            if "description" in types_to_apply:
                self._perturb_description(func)

            if "enum" in types_to_apply:
                self._perturb_enums(func, original_name)

            # 记录映射
            new_name = func.get("name", "")
            if new_name != original_name and original_name:
                self.name_map[new_name] = original_name

        # 安全性检查（提取 flat func 进行检查）
        original_funcs = [f.get("function", f) if "function" in f else f for f in functions]
        perturbed_funcs = [f.get("function", f) if "function" in f else f for f in perturbed]
        self._validate_perturbation(original_funcs, perturbed_funcs)

        logger.debug(
            f"扰动完成: level={level}, "
            f"functions={len(perturbed)}, "
            f"name_map新增={len(self.name_map)}"
        )
        return perturbed

    def _get_types_for_level(self, level: PerturbationLevel) -> list[str]:
        """根据扰动强度返回要启用的扰动类型列表。"""
        mapping = {
            LEVEL_MILD: ["tool_name"],
            LEVEL_MODERATE: ["tool_name", "description"],
            LEVEL_STRONG: ["tool_name", "description", "enum"],
        }
        return mapping.get(level, [])

    def _perturb_tool_name(self, func: dict) -> None:
        """扰动工具名：拆分词干 + 同义替换。"""
        name = func.get("name", "")
        if not name:
            return

        # 按下划线或驼峰拆分（保留连续大写缩写如 URL）
        # 先处理 camelCase → snake_case：小写后跟大写、大写后跟大写在跟小写的交界处插入下划线
        words = re.sub(
            r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])',
            '_',
            name,
        ).lower().split('_')
        words = [w for w in words if w]

        synonym_map_tn = self.synonym_map.get("tool_name", {})
        new_words = []
        for w in words:
            w_lower = w.lower()
            if w_lower in synonym_map_tn:
                candidates = synonym_map_tn[w_lower]
                # 避免选到自身
                filtered = [c for c in candidates if c != w_lower]
                if filtered:
                    new_words.append(self.rng.choice(filtered))
                else:
                    new_words.append(w)
            else:
                new_words.append(w)

        new_name = "_".join(new_words).lower()
        func["name"] = new_name

    def _perturb_description(self, func: dict) -> None:
        """扰动工具描述：关键词同义替换。"""
        desc = func.get("description", "")
        if not desc:
            return

        synonym_map_desc = self.synonym_map.get("description", {})
        enum_map = self.synonym_map.get("enum", {})
        for old_word, new_word in synonym_map_desc.items():
            if old_word in desc.lower() and new_word:
                # 只替换第一次出现，替换太多可能改变语义
                pattern = re.compile(re.escape(old_word), re.IGNORECASE)
                desc = pattern.sub(new_word, desc, count=1)

        # 扰动主 description 中的枚举值
        if enum_map:
            desc = self._perturb_description_enums(desc, enum_map)
        func["description"] = desc

        # 同样处理每个参数的 description
        parameters = func.get("parameters", {})
        for param_spec in parameters.get("properties", {}).values():
            param_desc = param_spec.get("description", "")
            if not param_desc:
                continue
            for old_word, new_word in synonym_map_desc.items():
                if old_word in param_desc.lower() and new_word:
                    pattern = re.compile(re.escape(old_word), re.IGNORECASE)
                    param_desc = pattern.sub(new_word, param_desc, count=1)
            if enum_map:
                param_desc = self._perturb_description_enums(param_desc, enum_map)
            param_spec["description"] = param_desc

    @staticmethod
    def _find_enum_patterns(desc: str) -> list[tuple[str, list[str]]]:
        """从 description 文本中提取枚举值模式。

        匹配以下格式：
        - "Available values: value1, value2, value3."
        - "Options: value1, value2, value3"
        - "Supported: value1/value2/value3"
        - "(value1/value2/value3)"
        - "[value1, value2, value3]"
        - "one of: value1, value2, value3"

        Returns:
            [(matched_text, [values]), ...] 每个匹配的文本片段和提取的枚举值列表
        """
        patterns = []
        # 模式1: "key: v1, v2, v3" 或 "key: v1/v2/v3"
        m = re.search(
            r'(?:available\s+)?(?:values?|options?|choices?|supported|one\s+of)\s*[:：]\s*'
            r'([a-zA-Z_][\w\s]*(?:,\s*[a-zA-Z_][\w\s]*)*)',
            desc, re.IGNORECASE
        )
        if m:
            text = m.group(0)
            values_str = m.group(1)
            # 按逗号或斜杠分割
            values = [v.strip().lower() for v in re.split(r'[,/]', values_str) if v.strip()]
            patterns.append((text, values))

        # 模式2: "(v1/v2/v3)" 括弧形式
        for m in re.finditer(r'\(([a-zA-Z_][\w\s]*(?:/[a-zA-Z_][\w\s]*)+)\)', desc):
            text = m.group(0)
            values_str = m.group(1)
            values = [v.strip().lower() for v in values_str.split('/') if v.strip()]
            patterns.append((text, values))

        # 模式3: "[v1, v2, v3]" 方括号形式
        for m in re.finditer(r'\[([a-zA-Z_][\w\s]*(?:,\s*[a-zA-Z_][\w\s]*)+)\]', desc):
            text = m.group(0)
            values_str = m.group(1)
            values = [v.strip().lower() for v in re.split(r'[,]', values_str) if v.strip()]
            patterns.append((text, values))

        return patterns

    def _perturb_description_enums(self, desc: str, enum_map: dict[str, list[str]]) -> str:
        """替换 description 中的枚举值为同义词。

        注意：这里只做文本替换让描述与 perturbed enum 一致，
        不记录到 self.enum_map（判分映射由 _perturb_enums 负责）。
        """
        patterns = self._find_enum_patterns(desc)
        for matched_text, values in patterns:
            new_text = matched_text
            for val in values:
                if val in enum_map:
                    candidates = [c for c in enum_map[val] if c != val]
                    if candidates:
                        chosen = self.rng.choice(candidates)
                        new_text = re.sub(
                            rf'\b{re.escape(val)}\b',
                            chosen,
                            new_text,
                            count=1,
                            flags=re.IGNORECASE,
                        )
            desc = desc.replace(matched_text, new_text)
        return desc

    def _perturb_enums(self, func: dict, original_func_name: str) -> None:
        """扰动枚举值。同时记录 perturbed->original 映射（按 func_name + param_name 精确记录）。

        注意：enum_map 的 key 使用 original function name，因为评估时会先
        通过 name_map resolve 回 original name 再查 enum_map。
        """
        parameters = func.get("parameters", {})
        properties = parameters.get("properties", {})

        synonym_map_enum = self.synonym_map.get("enum", {})
        for param_name, param_spec in properties.items():
            if "enum" in param_spec:
                new_enums = []
                for e in param_spec["enum"]:
                    if not isinstance(e, str):
                        new_enums.append(e)
                        continue
                    e_lower = e.lower()
                    if e_lower in synonym_map_enum:
                        candidates = synonym_map_enum[e_lower]
                        filtered = [c for c in candidates if c != e_lower]
                        if filtered:
                            chosen = self.rng.choice(filtered)
                            new_enums.append(chosen)
                            # 精确记录：original_func_name -> param_name -> {perturbed: original}
                            if original_func_name not in self.enum_map:
                                self.enum_map[original_func_name] = {}
                            if param_name not in self.enum_map[original_func_name]:
                                self.enum_map[original_func_name][param_name] = {}
                            self.enum_map[original_func_name][param_name][chosen] = e
                        else:
                            new_enums.append(e)
                    else:
                        new_enums.append(e)
                param_spec["enum"] = new_enums

    def _validate_perturbation(
        self, original: list[dict], perturbed: list[dict]
    ) -> None:
        """扰动安全性检查。"""
        # 1. 检查工具名冲突
        pert_names = {f["name"] for f in perturbed}
        if len(pert_names) != len(perturbed):
            raise ValueError(
                f"扰动导致工具名冲突: {len(pert_names)} 个唯一名 vs "
                f"{len(perturbed)} 条定义"
            )

        # 2. 检查必需参数字段不变
        for o, p in zip(original, perturbed):
            o_required = set(o.get("parameters", {}).get("required", []))
            p_required = set(p.get("parameters", {}).get("required", []))
            if o_required != p_required:
                raise ValueError(
                    f"扰动改变了必需参数: {o['name']} "
                    f"{o_required} -> {p_required}"
                )

        # 3. 检查参数类型不变
        for o, p in zip(original, perturbed):
            o_params = o.get("parameters", {}).get("properties", {})
            p_params = p.get("parameters", {}).get("properties", {})
            common_keys = set(o_params.keys()) & set(p_params.keys())
            for key in common_keys:
                o_type = o_params[key].get("type")
                p_type = p_params[key].get("type")
                if o_type != p_type:
                    raise ValueError(
                        f"扰动改变了参数类型: {o['name']}.{key} "
                        f"{o_type} -> {p_type}"
                    )

    def get_reverse_name_map(self) -> dict[str, str]:
        """获取反向映射表（原版名 → 扰动名）。

        Returns:
            {original_name: perturbed_name} 字典。
        """
        return {v: k for k, v in self.name_map.items()}


def generate_level_distribution(
    group_size: int, seed: Optional[int] = None
) -> list[PerturbationLevel]:
    """按 1:1:1 比例生成扰动强度分布。

    Args:
        group_size: 总 rollout 数。
        seed: 随机种子，用于确定性 shuffle。None 时使用模块级 random。

    Returns:
        扰动强度列表，长度 = group_size。

    Raises:
        ValueError: group_size 不是 3 的倍数。
    """
    if group_size % 3 != 0:
        raise ValueError(
            f"group_size ({group_size}) 必须是 3 的倍数，"
            f"建议使用 9 (3:3:3)"
        )

    per_level = group_size // 3
    distribution = (
        [LEVEL_NONE] * per_level
        + [LEVEL_MILD] * per_level
        + [LEVEL_STRONG] * per_level
    )
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(distribution)

    logger.debug(
        f"扰动强度分布生成: group_size={group_size}, "
        f"none={distribution.count(LEVEL_NONE)}, "
        f"mild={distribution.count(LEVEL_MILD)}, "
        f"strong={distribution.count(LEVEL_STRONG)}"
    )
    return distribution
