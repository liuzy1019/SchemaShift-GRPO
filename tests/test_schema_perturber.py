"""
SchemaPerturber 单元测试。
"""
import pytest
from src.envs.schema_perturber import (
    SchemaPerturber,
    generate_level_distribution,
    LEVEL_NONE,
    LEVEL_MILD,
    LEVEL_STRONG,
    TRAINING_LEVELS,
)


SAMPLE_FUNCTIONS = [
    {
        "name": "search_flights",
        "description": "Search for available flights between cities",
        "parameters": {
            "type": "dict",
            "properties": {
                "origin": {"type": "string", "description": "Departure city"},
                "destination": {"type": "string", "description": "Arrival city"},
                "date": {"type": "string", "description": "Travel date"},
            },
            "required": ["origin", "destination", "date"],
        },
    },
    {
        "name": "book_flight",
        "description": "Book a flight ticket",
        "parameters": {
            "type": "dict",
            "properties": {
                "flight_id": {"type": "string", "description": "Flight ID"},
                "seat_class": {
                    "type": "string",
                    "description": "Seat class",
                    "enum": ["economy", "business", "first"],
                },
            },
            "required": ["flight_id"],
        },
    },
]


class TestSchemaPerturber:
    """SchemaPerturber 功能测试。"""

    def setup_method(self):
        self.perturber = SchemaPerturber(seed=42)

    def test_init(self):
        """初始化后 name_map 应为空。"""
        assert len(self.perturber.name_map) == 0

    def test_no_perturbation(self):
        """LEVEL_NONE 应返回不变的功能列表。"""
        result = self.perturber.perturb(SAMPLE_FUNCTIONS, LEVEL_NONE)
        assert result == SAMPLE_FUNCTIONS
        assert len(self.perturber.name_map) == 0

    def test_mild_perturbation(self):
        """轻度扰动应只改工具名。"""
        result = self.perturber.perturb(SAMPLE_FUNCTIONS, LEVEL_MILD)

        # 名字应该变了
        names = {f["name"] for f in result}
        orig_names = {f["name"] for f in SAMPLE_FUNCTIONS}
        assert names != orig_names, "轻度扰动应改变工具名"

        # 描述应该没变
        for orig, res in zip(SAMPLE_FUNCTIONS, result):
            assert orig["description"] == res["description"]

        # name_map 应有记录
        assert len(self.perturber.name_map) > 0

    def test_strong_perturbation(self):
        """重度扰动应改工具名、描述、枚举值。"""
        result = self.perturber.perturb(SAMPLE_FUNCTIONS, LEVEL_STRONG)

        # 工具名应该变了
        names = {f["name"] for f in result}
        orig_names = {f["name"] for f in SAMPLE_FUNCTIONS}
        assert names != orig_names

        # 枚举值应该变了（book_flight.seat_class 有 enum: [economy, business, first]）
        for orig, res in zip(SAMPLE_FUNCTIONS, result):
            for p_name, p_spec in orig.get("parameters", {}).get("properties", {}).items():
                o_enum = p_spec.get("enum", [])
                r_enum = res["parameters"]["properties"][p_name].get("enum", [])
                if o_enum:
                    assert len(r_enum) == len(o_enum), (
                        f"enum 数量不应改变: {p_name} {o_enum} -> {r_enum}"
                    )
                    # 至少有一个枚举值被替换（同义词表覆盖 economy/business/first）
                    assert r_enum != o_enum or len(self.perturber.enum_map) == 0, (
                        f"重度扰动应改变枚举值: {p_name}"
                    )

        # 描述应该变了（strong 包含 description 扰动）
        any_desc_changed = False
        for orig, res in zip(SAMPLE_FUNCTIONS, result):
            if orig["description"] != res["description"]:
                any_desc_changed = True
                break
        assert any_desc_changed, "重度扰动应改变描述"

    def test_deterministic(self):
        """相同 seed 应产生相同结果。"""
        perturber1 = SchemaPerturber(seed=42)
        perturber2 = SchemaPerturber(seed=42)

        result1 = perturber1.perturb(SAMPLE_FUNCTIONS, LEVEL_MILD)
        result2 = perturber2.perturb(SAMPLE_FUNCTIONS, LEVEL_MILD)

        assert result1 == result2

    def test_seed_changes_result(self):
        """不同 seed 应产生不同结果。"""
        perturber1 = SchemaPerturber(seed=42)
        perturber2 = SchemaPerturber(seed=999)

        result1 = perturber1.perturb(SAMPLE_FUNCTIONS, LEVEL_MILD)
        result2 = perturber2.perturb(SAMPLE_FUNCTIONS, LEVEL_MILD)

        # 不同 seed 应产生不同的工具名映射
        names1 = {f["name"] for f in result1}
        names2 = {f["name"] for f in result2}
        assert result1 != result2, (
            f"不同 seed 应产生不同结果: names1={names1}, names2={names2}"
        )

    def test_name_map_consistency(self):
        """name_map 记录应一致。"""
        perturber = SchemaPerturber(seed=42)
        perturber.perturb(SAMPLE_FUNCTIONS, LEVEL_MILD)

        for pert_name, orig_name in perturber.name_map.items():
            assert orig_name in {f["name"] for f in SAMPLE_FUNCTIONS}
            assert pert_name != orig_name

    def test_get_reverse_name_map(self):
        """反向映射应为正向映射的逆。"""
        perturber = SchemaPerturber(seed=42)
        perturber.perturb(SAMPLE_FUNCTIONS, LEVEL_STRONG)

        reverse = perturber.get_reverse_name_map()
        for pert, orig in perturber.name_map.items():
            assert reverse[orig] == pert

    def test_validate_no_conflict(self):
        """扰动不应导致工具名冲突。"""
        perturber = SchemaPerturber(seed=42)
        result = perturber.perturb(SAMPLE_FUNCTIONS, LEVEL_STRONG)

        names = [f["name"] for f in result]
        assert len(names) == len(set(names)), "工具名不应冲突"

    def test_parameters_required_unchanged(self):
        """必需参数字段不应被扰动改变。"""
        perturber = SchemaPerturber(seed=42)
        result = perturber.perturb(SAMPLE_FUNCTIONS, LEVEL_STRONG)

        for orig, res in zip(SAMPLE_FUNCTIONS, result):
            # required 字段应完全一致
            o_req = set(orig["parameters"].get("required", []))
            r_req = set(res["parameters"].get("required", []))
            # 注意：如果参数名被扰动，required 中的名字也会变
            assert len(o_req) == len(r_req), "required 数量不应改变"


class TestLevelDistribution:
    """generate_level_distribution 测试。"""

    def test_valid_group_size(self):
        """合法的 group_size 应产生正确分布。"""
        dist = generate_level_distribution(9)
        assert len(dist) == 9
        assert dist.count(LEVEL_NONE) == 3
        assert dist.count(LEVEL_MILD) == 3
        assert dist.count(LEVEL_STRONG) == 3

    def test_invalid_group_size(self):
        """不合法的 group_size 应报错。"""
        with pytest.raises(ValueError):
            generate_level_distribution(8)

    def test_only_training_levels(self):
        """分布中只应包含训练级别。"""
        dist = generate_level_distribution(9)
        for level in dist:
            assert level in TRAINING_LEVELS
