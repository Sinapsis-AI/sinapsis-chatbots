# -*- coding: utf-8 -*-
import re
from typing import Any

from sinapsis_chatbots_base.helpers.llm_keys import MCPKeys


def make_tools_anthropic_compatible(tools: list[dict]) -> list[dict]:
    """
    Make tool input schemas compatible with Anthropic API.

    Only fixes input_schema property names that contain invalid characters
    like < > which are common in Twilio MCP tools.

    Args:
        tools (list[dict]): Original tools from MCP servers

    Returns:
        list[dict]: Tools with compatible input schemas
    """
    if not tools:
        return tools

    compatible_tools = []

    for tool in tools:
        compatible_tool = tool.copy()

        if MCPKeys.input_schema in tool and isinstance(tool[MCPKeys.input_schema], dict):
            compatible_tool[MCPKeys.input_schema] = _fix_input_schema(tool[MCPKeys.input_schema])

        compatible_tools.append(compatible_tool)

    return compatible_tools


def _fix_input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Fix input schema properties to be Anthropic API compatible."""
    if MCPKeys.properties not in schema:
        return schema

    fixed_schema = schema.copy()
    fixed_properties = {}

    for prop_name, prop_definition in schema[MCPKeys.properties].items():
        compatible_prop_name = _make_property_compatible(prop_name)
        fixed_properties[compatible_prop_name] = prop_definition

    fixed_schema[MCPKeys.properties] = fixed_properties

    if MCPKeys.required in schema and isinstance(schema[MCPKeys.required], list):
        fixed_schema[MCPKeys.required] = [
            _make_property_compatible(required_prop) for required_prop in schema[MCPKeys.required]
        ]

    return fixed_schema


def _make_property_compatible(property_name: str) -> str:
    """Make a property name compatible with Anthropic's requirements."""
    compatible_name = property_name

    compatible_name = compatible_name.replace("<", "_lt")
    compatible_name = compatible_name.replace(">", "_gt")
    compatible_name = compatible_name.replace("<=", "_lte")
    compatible_name = compatible_name.replace(">=", "_gte")

    compatible_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", compatible_name)

    if len(compatible_name) > 64:
        compatible_name = compatible_name[:64]

    return compatible_name
