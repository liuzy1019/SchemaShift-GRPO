from src.live_mcp.schema_registry import SchemaRegistry


def test_schema_registry_validates_required_type_and_enum():
    registry = SchemaRegistry()
    registry.register_tools(
        "s",
        [
            {
                "name": "paint",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "string", "enum": ["red"]},
                        "count": {"type": "integer"},
                    },
                    "required": ["color", "count"],
                },
            }
        ],
    )
    valid = registry.validate_arguments("paint", {"color": "red", "count": 1})
    invalid = registry.validate_arguments("paint", {"color": "blue", "count": "1"})
    assert valid.valid is True
    assert invalid.valid is False
    assert invalid.type_errors == ["count"]
    assert invalid.enum_errors == ["color"]
