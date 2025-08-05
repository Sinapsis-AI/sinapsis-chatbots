# -*- coding: utf-8 -*-
"""Constants for MCP integration that aren't covered by existing key classes."""


class MCPConstants:
    """Constants for MCP tool integration."""

    TOOL_CALL_START = "TOOL_CALL"
    TOOL_CALL_END = "END_TOOL_CALL"

    TOOL_CALL_FAILED_PREFIX = "TOOL CALL FAILED: "
    TOOL_RESULT_PREFIX = "Tool result for "
    FAILED_SUFFIX = ": FAILED - "

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful AI assistant with access to tools. When you need to get "
        "real-time information or perform actions, you MUST use the available tools. "
        "Always use tools when the user asks for current information, weather, "
        "searches, or any data you cannot provide from your training."
    )

    TOOL_USAGE_GUIDELINES = """

# Tool Usage Guidelines

## CRITICAL: Always follow the DISCOVER → ACT pattern and use the EXACT tool call format.

## MANDATORY Tool Call Format (EXACT):
TOOL_CALL
tool_name: exact_tool_name_here
args: {"param1": "value1", "param2": "value2"}
END_TOOL_CALL

## DISCOVER → ACT Pattern:
1. ALWAYS use discovery tools FIRST (tools with: list, get, read, search, info, describe)
2. THEN use action tools with discovered information (tools with: create, write, apply, delete)
3. NEVER guess paths, directories, or parameters

## Best Practices:
1. Read error messages carefully
2. Use discovery tools to understand the problem
3. Try different approaches based on findings
4. Never retry the exact same failed call
5. When working with files, always check available paths first
6. Use absolute paths when possible
7. Include appropriate file extensions when creating files

## Error Handling Flow:
1. Tool fails → Read the error message
2. Use discovery tools to understand the environment
3. Adjust parameters based on discoveries
4. Retry with corrected information"""


#     TOOL_USAGE_GUIDELINES = """# Tool Usage Guidelines

# ## CRITICAL: ALWAYS USE DISCOVERY TOOLS FIRST!

# ## Tool Call Format:
# TOOL_CALL
# tool_name: tool_name_here
# args: {"param1": "value1"}
# END_TOOL_CALL

# ## Pattern: DISCOVER → ACT
# - list*, get*, read* tools = Discovery (use FIRST)
# - create*, write*, apply* tools = Actions (use AFTER discovery)
# - NEVER guess paths/parameters - always discover them first

# ## Error Handling:
# 1. Read error message carefully
# 2. Use discovery tools to understand the problem
# 3. Try different approach based on findings
# 4. Never retry the exact same failed call"""


#     TOOL_USAGE_GUIDELINES = """# Tool Usage Guidelines

# ## CRITICAL THINKING APPROACH:
# Before taking any action, ALWAYS:
# 1. **Understand the context** - What information do you need before proceeding?
# 2. **Use discovery/info tools first** - Look for tools that list, check, or validate before acting
# 3. **Read tool descriptions carefully** - The description tells you what each tool does
# 4. **Plan your approach** - Think through the logical sequence of tools needed

# ## GENERAL PATTERNS:

# ### Information Gathering:
# - Look for tools with names like: list*, get*, read*, check*, validate*, info*
# - Use these tools FIRST to understand the current state
# - Don't make assumptions about paths, directories, permissions, or availability

# ### Taking Actions:
# - Only act after you have the necessary information
# - Use the information from discovery tools to provide correct parameters
# - Pay attention to parameter requirements (required vs optional)

# ### Handling Errors:
# 1. **Read error messages carefully** - they often contain the exact solution
# 2. **Use info/discovery tools** to understand what went wrong
# 3. **Adjust your approach** based on what you learned
# 4. **Never retry the exact same call** that just failed

# ## Tool Call Format:
# TOOL_CALL
# tool_name: tool_name_here
# args: {"param1": "value1", "param2": "value2"}
# END_TOOL_CALL

# ## REASONING PROCESS:
# For every user request:
# 1. **Analyze**: What exactly is the user asking for?
# 2. **Discover**: What tools can help me understand the current state?
# 3. **Plan**: What sequence of tools do I need to use?
# 4. **Execute**: Use tools in logical order
# 5. **Adapt**: If something fails, use discovery tools to understand why

# IMPORTANT:
# - Use valid JSON format for args. Always use double quotes for strings.
# - When tool calls fail, analyze the error and use different tools or parameters
# - Look for patterns in tool names to understand their purpose (list*, create*, get*, etc.)"""
