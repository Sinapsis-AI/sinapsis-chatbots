# -*- coding: utf-8 -*-
from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class LLMChatKeys:
    """
    A class to hold constants for the keys used in chat interactions with an LLM (Large Language Model).

    These keys represent the standard fields in a chat interaction, such as the role of the participant
    and the content of the message. They are typically used when constructing input messages or
    processing the output from an LLM.
    """

    role: Literal["role"] = "role"
    content: Literal["content"] = "content"
    choices: Literal["choices"] = "choices"
    message: Literal["message"] = "message"
    llm_responses: Literal["llm_responses"] = "llm_responses"
    system_value: Literal["system"] = "system"
    user_value: Literal["user"] = "user"
    assistant_value: Literal["assistant"] = "assistant"


@dataclass
class MCPKeys:
    tools: Literal["tools"] = "tools"
    tool_use: Literal["tool_use"] = "tool_use"
    tool_results: Literal["tool_results"] = "tool_results"
    tool_name: Literal["tool_name"] = "tool_name"
    args: Literal["args"] = "args"
    tool_use_id: Literal["tool_use_id"] = "tool_use_id"
    partial_response: Literal["partial_response"] = "partial_response"
    tool_calls: Literal["tool_calls"] = "tool_calls"
    last_container: Literal["_last_container"] = "_last_container"
    input_schema: Literal["input_schema"] = "input_schema"
    properties: Literal["properties"] = "properties"
    required: Literal["required"] = "required"
    partial_query: Literal["partial_query"] = "partial_query"
    tool_type: Literal["type"] = "type"
    function: Literal["function"] = "function"
    name: Literal["name"] = "name"
    description: Literal["description"] = "description"
    parameters: Literal["parameters"] = "parameters"
    is_error: Literal["is_error"] = "is_error"
    text: Literal["text"] = "text"
    finish_reason: Literal["finish_reason"] = "finish_reason"
    tool_id: Literal["id"] = "id"
    arguments: Literal["arguments"] = "arguments"
    tool_call_id: Literal["tool_call_id"] = "tool_call_id"
    tool: Literal["tool"] = "tool"
