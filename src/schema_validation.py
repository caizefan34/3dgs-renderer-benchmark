"""Small JSON Schema validator for committed benchmark artifacts.

It supports the schema features used by this repository: type, required,
properties, items, enum, minimum, and additionalProperties.
"""
from __future__ import annotations

import json
from typing import Any, Mapping


class SchemaValidationError(ValueError):
    pass


_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value: Any, expected) -> bool:
    if isinstance(expected, list):
        return any(_type_ok(value, item) for item in expected)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, _TYPE_MAP[expected])


def validate_schema(instance: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    if "type" in schema and not _type_ok(instance, schema["type"]):
        raise SchemaValidationError(f"{path}: expected {schema['type']}, got {type(instance).__name__}")
    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaValidationError(f"{path}: {instance!r} is not in enum")
    if "minimum" in schema and instance < schema["minimum"]:
        raise SchemaValidationError(f"{path}: {instance!r} is below minimum {schema['minimum']}")

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                raise SchemaValidationError(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                validate_schema(value, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise SchemaValidationError(f"{path}: unexpected key {key!r}")
    elif isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            validate_schema(item, schema["items"], f"{path}[{index}]")


def validate_json_file(json_path: str, schema_path: str) -> None:
    with open(json_path, encoding="utf-8") as handle:
        instance = json.load(handle)
    with open(schema_path, encoding="utf-8") as handle:
        schema = json.load(handle)
    validate_schema(instance, schema)

