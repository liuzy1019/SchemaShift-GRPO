"""公共 AST 匹配逻辑。

eval、reward、agent loop 三处共享此模块，避免训练 loop 依赖 eval 主文件。
"""


def _normalize_value(v) -> str:
    """AST 类型宽松匹配：将值归一化为可比较的字符串。

    BFCL 官方 AST 评估允许以下等价：
    - "1" == 1 (字符串数字 vs 整数)
    - "True" == True (字符串布尔 vs 布尔)
    - "None" == None
    - "[1, 2]" == [1, 2] (字符串列表 vs 列表)
    """
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    if isinstance(v, list):
        return str(v)
    s = str(v)
    try:
        num = float(s)
        if num == int(num) and "." not in s:
            return str(int(num))
        return str(num)
    except (ValueError, TypeError):
        pass
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        s = s[1:-1]
    return s


def _args_match(agent_args: dict, gt_args: dict) -> bool:
    """AST 宽松匹配：比较两个参数字典。"""
    if set(agent_args.keys()) != set(gt_args.keys()):
        return False
    for k in gt_args:
        if _normalize_value(agent_args[k]) != _normalize_value(gt_args[k]):
            return False
    return True


def map_enum_values(
    func_name: str,
    args: dict,
    enum_map: dict,
) -> dict:
    """对参数值做 enum 映射，同时检查合法性。

    enum_map 支持两种格式：
    1. 精确格式（推荐）: {func_name: {param_name: {perturbed_val: original_val}}}
    2. 扁平格式（兼容旧数据）: {perturbed_val: original_val}

    映射规则：
    - 模型输出 perturbed enum value → 映射回 original（判对）
    - 模型输出 original enum value（在 perturbed schema 下非法）→ 标记为无效（判错）
    - 非 enum 参数值不受影响

    Args:
        func_name: 当前函数名（已 resolve 为 original name）。
        args: 模型输出的参数字典。
        enum_map: enum 映射表。

    Returns:
        映射后的参数字典。
    """
    if not enum_map:
        return args

    # 判断格式：如果第一个 value 是 dict，则为精确格式
    first_val = next(iter(enum_map.values()), None) if enum_map else None
    is_nested = isinstance(first_val, dict)

    if is_nested:
        return _map_enum_nested(func_name, args, enum_map)
    else:
        return _map_enum_flat(args, enum_map)


def _map_enum_nested(func_name: str, args: dict, enum_map: dict) -> dict:
    """精确格式：按 func_name + param_name 限定映射范围。"""
    func_enums = enum_map.get(func_name, {})
    if not func_enums:
        return args

    mapped = {}
    for k, v in args.items():
        param_enums = func_enums.get(k)
        if not param_enums:
            mapped[k] = v
            continue

        v_str = str(v)
        # reverse: {original_val: perturbed_val}
        reverse = {orig: pert for pert, orig in param_enums.items()}

        if v_str in param_enums:
            # 合法 perturbed value → 映射回 original
            mapped[k] = param_enums[v_str]
        elif v_str in reverse:
            # 模型输出了 original value，perturbed schema 下非法
            mapped[k] = f"__INVALID_ENUM_{v_str}__"
        else:
            mapped[k] = v
    return mapped


def _map_enum_flat(args: dict, enum_map: dict) -> dict:
    """扁平格式（兼容旧数据）：全局映射 + 合法性检查。"""
    reverse = {v: k for k, v in enum_map.items()}
    mapped = {}
    for k, v in args.items():
        v_str = str(v)
        if v_str in enum_map:
            # 合法 perturbed value → 映射回 original
            mapped[k] = enum_map[v_str]
        elif v_str in reverse:
            # original value 在 perturbed schema 下非法
            mapped[k] = f"__INVALID_ENUM_{v_str}__"
        else:
            mapped[k] = v
    return mapped
