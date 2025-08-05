# -*- coding: utf-8 -*-
"""Helper functions for MCP tool handling."""

import json
import re
from typing import Any

from sinapsis_chatbots_base.helpers.llm_keys import MCPKeys
from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_llama_cpp.helpers.mcp_constants import MCPConstants


def make_tools_llama_compatible(tools: list[dict]) -> list[dict]:
    """Convert MCP tools to llama-cpp compatible format."""
    llama_tools = []
    for tool in tools:
        llama_tool = {
            MCPKeys.tool_type: MCPKeys.function,
            MCPKeys.function: {
                MCPKeys.name: tool[MCPKeys.name],
                MCPKeys.description: tool.get(MCPKeys.description, ""),
                MCPKeys.parameters: tool.get(MCPKeys.input_schema, {}),
            },
        }
        llama_tools.append(llama_tool)
    return llama_tools


def extract_tool_calls_from_content(content: str) -> list[dict[str, Any]] | None:
    """Extract tool calls from assistant content using flexible block parsing."""
    pattern = rf"{MCPConstants.TOOL_CALL_START}\s*\n(.*?)(?:\n{MCPConstants.TOOL_CALL_END}|$)"
    blocks = re.findall(pattern, content, re.DOTALL)
    tool_calls = []
    for block in blocks:
        parsed = _parse_tool_block(block)
        if parsed is not None:
            tool_calls.append(parsed)

    return tool_calls if tool_calls else None


def _parse_tool_block(block: str) -> dict[str, Any] | None:
    """Parse a single tool block into tool call dict."""
    lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
    if not lines:
        return None

    tool_name = _extract_tool_name(lines)
    args_dict = _extract_tool_args(lines)

    if tool_name:
        return {
            MCPKeys.tool_name: tool_name,
            MCPKeys.args: args_dict,
            MCPKeys.tool_use_id: f"{tool_name}_{hash(block) % 1000}",
        }
    return None


def _extract_tool_name(lines: list[str]) -> str | None:
    """Extract tool name from lines."""
    tool_name_from_keyword = None
    tool_name_from_standalone = None

    for line in lines:
        if line.startswith(f"{MCPKeys.tool_name}:"):
            tool_name_from_keyword = line.split(":", 1)[1].strip()
        elif ":" in line and not line.startswith(("args:", f"{MCPKeys.args}:")):
            key_part = line.split(":", 1)[0].strip()
            value_part = line.split(":", 1)[1].strip()
            if value_part.startswith(("{", "[")):
                return key_part
        elif (
            not line.startswith(("args:", f"{MCPKeys.args}:", "tool_name:"))
            and not line.startswith(("{", "["))
            and tool_name_from_standalone is None
        ):
            tool_name_from_standalone = line

    return tool_name_from_keyword or tool_name_from_standalone


def _extract_tool_args(lines: list[str]) -> dict[str, Any]:
    """Extract tool arguments from lines."""
    for line in lines:
        args_str = None

        if line.startswith(("args:", f"{MCPKeys.args}:")):
            args_str = line.split(":", 1)[1].strip()
        elif ":" in line and not line.startswith(f"{MCPKeys.tool_name}:"):
            value_part = line.split(":", 1)[1].strip()
            if value_part.startswith(("{", "[")):
                args_str = value_part

        if args_str is not None:
            if args_str in ["{}", "", "null", "None"]:
                return {}

            try:
                return json.loads(args_str)
            except json.JSONDecodeError as e:
                sinapsis_logger.warning(f"Failed to parse args: {args_str}, error: {e}")
                if not args_str.strip() or args_str.strip() in ["{}", "null", "None"]:
                    return {}
    return {}


def format_json_content(raw_text: str) -> str:
    """Format JSON content if valid, otherwise return as-is."""
    text = raw_text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return raw_text

    try:
        parsed_data = json.loads(raw_text)
        return json.dumps(parsed_data, indent=2)
    except json.JSONDecodeError:
        return raw_text


def build_tool_description(tool: dict[str, Any]) -> str:
    """Build a formatted tool description."""
    if not (tool.get(MCPKeys.tool_type) == MCPKeys.function and MCPKeys.function in tool):
        return ""

    func = tool[MCPKeys.function]
    description = f"\n## {func[MCPKeys.name]}\n"
    description += f"Description: {func.get(MCPKeys.description, '')}\n"

    params = func.get(MCPKeys.parameters, {})
    if params.get(MCPKeys.properties):
        description += "Parameters:\n"
        required_params = params.get(MCPKeys.required, [])

        for param_name, param_info in params[MCPKeys.properties].items():
            param_type = param_info.get(MCPKeys.tool_type, "string")
            param_desc = param_info.get(MCPKeys.description, "")
            req_text = " (required)" if param_name in required_params else " (optional)"
            description += f"  - {param_name} ({param_type}){req_text}: {param_desc}\n"

    return description + "\n"


def extract_text_content(content_list: list[Any]) -> str:
    """Extract text content from MCP content list."""
    if not content_list or not hasattr(content_list[0], MCPKeys.text):
        return ""
    return content_list[0].text
