# -*- coding: utf-8 -*-
import re
from typing import Any

from sinapsis_chatbots_base.helpers.llm_keys import MCPKeys
from sinapsis_core.utils.logging_utils import sinapsis_logger


def make_tools_anthropic_compatible(tools: list[dict] | dict | None) -> list[dict]:
    """Make tool input schemas compatible with Anthropic API.

    Only fixes input_schema property names that contain invalid characters
    like < > which are common in Twilio MCP tools.

    Args:
        tools (list[dict] | dict | None): The list of tools, potentially invalid,
            from an MCP source.

    Returns:
        list[dict]: Tools with compatible input schemas
    """
    compatible_tools: list[dict] = []

    if not tools:
        sinapsis_logger.warning("Received 'tools' input that was None or empty. Returning empty tool list.")
        return compatible_tools

    if not isinstance(tools, list):
        sinapsis_logger.warning(
            f"Received 'tools' input of invalid type {type(tools)}. Expected a list. Returning empty tool list."
        )
        return compatible_tools

    for tool in tools:
        compatible_tool = tool.copy()

        if MCPKeys.input_schema in tool and isinstance(tool[MCPKeys.input_schema], dict):
            compatible_tool[MCPKeys.input_schema] = _fix_input_schema(tool[MCPKeys.input_schema])

        compatible_tools.append(compatible_tool)

    return compatible_tools


def _fix_input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Fix input schema properties to be Anthropic API compatible.

    Args:
        schema (dict[str, Any]): The JSON schema for a single tool's `input_schema`.

    Returns:
        dict[str, Any]: A new schema dictionary with sanitized `properties` keys
            and `required` list values.
    """
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
    """Make a property name compatible with Anthropic's requirements.

    Args:
        property_name (str): The raw property name.

    Returns:
        str: The sanitized, API-compatible property name.
    """
    compatible_name = property_name

    compatible_name = compatible_name.replace("<=", "_lte")
    compatible_name = compatible_name.replace(">=", "_gte")
    compatible_name = compatible_name.replace("<", "_lt")
    compatible_name = compatible_name.replace(">", "_gt")

    compatible_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", compatible_name)

    if len(compatible_name) > 64:
        compatible_name = compatible_name[:64]

    return compatible_name
