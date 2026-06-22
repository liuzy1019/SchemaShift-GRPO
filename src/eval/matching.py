"""公共 AST 匹配逻辑。

eval、reward、agent loop 三处共享此模块，避免训练 loop 依赖 eval 主文件。
"""
import ast
import re
from typing import Any


# ── _parse_bfcl_native_args 相关常量 ──
_PARSE_MAX_INPUT_LEN = 8192
_PARSE_MAX_ARGS = 64
_PARSE_MAX_KEY_LEN = 64
_PARSE_MAX_LITERAL_LEN = 4096
_PARSE_NAME_MAX_SCAN = 256
_IDENT_RE = re.compile(r"^[a-zA-Z_]\w{0,63}$")


def _looks_like_number(s: str) -> bool:
    """判断字符串是否像数字字面量。"""
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_bfcl_native_args(func_call_str: str) -> tuple[str, dict[str, Any]]:
    """解析 BFCL 原生格式的函数调用字符串，bounded linear parser。

    设计目标：永不卡死、无 O(N²) 路径、超长/畸形输入直接降级返回。
    任何分支失败都返回 ("", {}) 或 (name, {}), 由调用方走 fallback。
    """
    if not func_call_str or len(func_call_str) > _PARSE_MAX_INPUT_LEN:
        return "", {}

    # ---- Step 1: 函数名 ----
    head = func_call_str[:_PARSE_NAME_MAX_SCAN]
    name_match = re.match(r"([a-zA-Z_][\w.]{0,127})\s*\(", head)
    if not name_match:
        return "", {}
    name = name_match.group(1).rstrip(".")
    if not name:
        return "", {}

    # ---- Step 2: 定位匹配的 ')' ----
    args_start = func_call_str.find("(")
    if args_start < 0:
        return name, {}
    paren_depth = 0
    args_end = -1
    for j in range(args_start, len(func_call_str)):
        c = func_call_str[j]
        if c == "(":
            paren_depth += 1
        elif c == ")":
            paren_depth -= 1
            if paren_depth == 0:
                args_end = j
                break
    if args_end < 0:
        return name, {}
    args_part = func_call_str[args_start + 1:args_end]
    if not args_part or not args_part.strip():
        return name, {}

    # ---- Step 3: 切段 ----
    segments: list[tuple[str, int]] = []
    n = len(args_part)
    i = 0
    seg_start = 0
    eq_offset = -1
    quote = ""
    bracket = 0
    brace = 0
    paren = 0
    while i < n:
        if len(segments) >= _PARSE_MAX_ARGS:
            return name, {}
        c = args_part[i]
        if quote:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            i += 1
            continue
        if c == "[":
            bracket += 1
        elif c == "]" and bracket > 0:
            bracket -= 1
        elif c == "{":
            brace += 1
        elif c == "}" and brace > 0:
            brace -= 1
        elif c == "(":
            paren += 1
        elif c == ")" and paren > 0:
            paren -= 1
        elif c == "=" and bracket == 0 and brace == 0 and paren == 0 and eq_offset < 0:
            eq_offset = i - seg_start
        elif c == "," and bracket == 0 and brace == 0 and paren == 0:
            segments.append((args_part[seg_start:i], eq_offset))
            seg_start = i + 1
            eq_offset = -1
        i += 1
    tail = args_part[seg_start:n]
    if tail.strip():
        segments.append((tail, eq_offset))

    # ---- Step 4: 分 kwarg/positional ----
    args: dict[str, Any] = {}
    positional_idx = 0
    for seg_text, eq_off in segments:
        seg = seg_text.strip()
        if not seg:
            continue
        if eq_off >= 0:
            raw_key = seg_text[:eq_off]
            raw_val = seg_text[eq_off + 1:]
            key_stripped = raw_key.strip()
            if (
                len(key_stripped) <= _PARSE_MAX_KEY_LEN
                and _IDENT_RE.match(key_stripped)
            ):
                args[key_stripped] = raw_val.strip()
                continue
        args[f"_pos_{positional_idx}"] = seg
        positional_idx += 1

    # ---- Step 5: 字面量回填 ----
    for k in list(args.keys()):
        v = args[k]
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s or len(s) > _PARSE_MAX_LITERAL_LEN:
            continue
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            try:
                args[k] = ast.literal_eval(s)
                continue
            except (ValueError, SyntaxError):
                args[k] = s[1:-1]
                continue
        if s[0] in "[{" or s in ("True", "False", "None") or _looks_like_number(s):
            try:
                args[k] = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                pass
    return name, args


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
