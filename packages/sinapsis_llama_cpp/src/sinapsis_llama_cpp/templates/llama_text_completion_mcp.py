# -*- coding: utf-8 -*-

import json
from typing import Any, cast

from llama_cpp.llama_types import CreateChatCompletionResponse
from sinapsis_chatbots_base.helpers.llm_keys import LLMChatKeys, MCPKeys
from sinapsis_chatbots_base.helpers.postprocess_text import postprocess_text
from sinapsis_chatbots_base.helpers.tags import Tags
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base.base_models import TemplateAttributeType

from sinapsis_llama_cpp.helpers.llama_keys import LLaMAModelKeys
from sinapsis_llama_cpp.helpers.mcp_constants import MCPConstants
from sinapsis_llama_cpp.helpers.mcp_helpers import (
    build_tool_description,
    extract_tool_calls_from_content,
    format_json_content,
    make_tools_llama_compatible,
)
from sinapsis_llama_cpp.templates.llama_text_completion import LLaMATextCompletion

LLaMAMultiModalUIProperties = LLaMATextCompletion.UIProperties
LLaMAMultiModalUIProperties.tags.extend([Tags.MCP])


class LLaMATextCompletionWithMCP(LLaMATextCompletion):
    """Template for LLaMA text completion with MCP tool integration."""

    UIProperties = LLaMAMultiModalUIProperties
    system_prompt: str | None

    class AttributesBaseModel(LLaMATextCompletion.AttributesBaseModel):
        generic_key: str = ""
        max_tool_retries: int = 3
        add_tool_to_prompt: bool = True

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.available_tools: list = []
        self.tool_results: list = []
        self.tool_calls: list = []
        self.partial_response: str = ""
        self.partial_query: list = []
        self._last_container = None

    def _add_tools_to_system_prompt(self) -> None:
        """Add tools description to system prompt."""
        current_prompt = self.system_prompt or MCPConstants.DEFAULT_SYSTEM_PROMPT

        tools_section = "\n\n# Available Tools\n"
        for tool in self.available_tools:
            tools_section += build_tool_description(tool)
        tools_section += MCPConstants.TOOL_USAGE_GUIDELINES

        self.system_prompt = current_prompt + tools_section

    def get_response(self, input_message: str | list) -> str | None:
        """Generate response from model with tool support."""
        self.logger.debug(f"Query is {input_message}")
        chat_completion = self._create_chat_completion(input_message)
        response_text = self._process_chat_completion(chat_completion)

        if response_text:
            response_text = postprocess_text(str(response_text), self.attributes.pattern, self.attributes.keep_before)

        if self.partial_response and self.tool_calls:
            self.partial_response = f"{self.partial_response}\n{response_text}"
            return self.partial_response
        elif self.partial_response and not self.tool_calls:
            final_response = f"{self.partial_response}\n{response_text}"
            self.partial_response = ""
            return final_response
        else:
            return response_text

    def _create_chat_completion(self, input_message: str | list):
        """Create chat completion with appropriate format."""
        try:
            if self.attributes.chat_format == LLaMAModelKeys.chatml_function_calling:
                return self.llm.create_chat_completion(
                    messages=input_message, tools=self.available_tools, tool_choice="auto"
                )
            return self.llm.create_chat_completion(messages=input_message)
        except IndexError:
            self.reset_llm_state()
            return self._create_chat_completion(input_message)

    def _process_chat_completion(self, chat_completion) -> str:
        """Process chat completion response."""
        chat_completion = cast(CreateChatCompletionResponse, chat_completion)
        llm_response_choice = chat_completion[LLMChatKeys.choices][0]
        finish_reason = llm_response_choice[MCPKeys.finish_reason]
        message = llm_response_choice[LLMChatKeys.message]

        # self.logger.debug(llm_response_choice)
        self.tool_calls = []

        if finish_reason == MCPKeys.tool_calls:
            return self._handle_function_calling_response(message)
        return self._handle_regular_response(message)

    def _handle_function_calling_response(self, message: dict) -> str:
        """Handle chatml-function-calling response format."""
        response_parts = []

        if message[LLMChatKeys.content]:
            response_parts.append(message[LLMChatKeys.content])
            self.partial_query.append(self.generate_dict_msg(LLMChatKeys.assistant_value, message[LLMChatKeys.content]))

        if message.get(MCPKeys.tool_calls):
            for tool_call in message[MCPKeys.tool_calls]:
                tool_name = tool_call[MCPKeys.function][MCPKeys.name]
                tool_args = json.loads(tool_call[MCPKeys.function][MCPKeys.arguments])

                self.tool_calls.append(
                    {
                        MCPKeys.tool_name: tool_name,
                        MCPKeys.args: tool_args,
                        MCPKeys.tool_use_id: tool_call[MCPKeys.tool_id],
                    }
                )
                call_info = f"[Calling tool {tool_name} with args {tool_args}]"
                response_parts.append(call_info)
                self.partial_query.append(self.generate_dict_msg(LLMChatKeys.assistant_value, call_info))

        return "\n".join(response_parts)

    def _handle_regular_response(self, message: dict) -> str:
        """Handle regular chatml response format."""
        response_text = message[LLMChatKeys.content]
        self.tool_calls = extract_tool_calls_from_content(response_text)

        if self.tool_calls:
            self.logger.debug(f"Extracted {len(self.tool_calls)} tool calls from content")
            response_parts = [response_text]

            response_parts.extend(
                [
                    f"[Calling tool {tool_call[MCPKeys.tool_name]} with args {tool_call[MCPKeys.args]}]"
                    for tool_call in self.tool_calls
                ]
            )
            response_text = "\n".join(response_parts)

            self.partial_query.append(self.generate_dict_msg(LLMChatKeys.assistant_value, message[LLMChatKeys.content]))

        return response_text

    def get_extra_context(self, packet: TextPacket) -> str | None:
        """Override to get state from packet and format tool results if present."""
        self.tool_results = packet.generic_data.get(MCPKeys.tool_results, [])
        self.partial_response = packet.generic_data.get(MCPKeys.partial_response, "")
        self.partial_query = packet.generic_data.get(MCPKeys.partial_query, [])

        if not self.available_tools and not self.tool_results:
            self.logger.error("No tools found on text packet's generic data")
        return super().get_extra_context(packet)

    def num_elements(self) -> int:
        """Control WhileLoop continuation based on pending tool calls."""
        if not self._has_pending_tool_calls():
            return -1

        failure_count = sum(1 for result in self.tool_results if result.get(MCPKeys.is_error, False))
        if failure_count >= self.attributes.max_tool_retries:
            self.logger.warning(f"Stopping loop after {failure_count} consecutive tool failures")
            return -1
        return 1

    def _has_pending_tool_calls(self) -> bool:
        """Check if there are pending tool calls."""
        if not hasattr(self, MCPKeys.last_container) or not self._last_container:
            return False
        return any(MCPKeys.tool_calls in packet.generic_data for packet in self._last_container.texts)

    def _format_tool_results_for_conversation(self, tool_results: list[dict[str, Any]]) -> list[dict]:
        """Format tool execution results based on chat format."""
        if self.attributes.chat_format == LLaMAModelKeys.chatml_function_calling:
            return self._format_as_tool_messages(tool_results)
        return self._format_as_user_messages(tool_results)

    def _format_as_tool_messages(self, tool_results: list[dict[str, Any]]) -> list[dict]:
        """Format tool results as tool role messages."""
        tool_messages = []
        for tool in tool_results:
            tool_call_id = tool.get(MCPKeys.tool_use_id, "unknown")
            content = tool.get(LLMChatKeys.content, [])
            is_error = tool.get(MCPKeys.is_error, False)

            raw_text = content[0].text
            text_content = format_json_content(raw_text)

            tool_messages.append(
                {
                    LLMChatKeys.role: MCPKeys.tool,
                    MCPKeys.tool_call_id: tool_call_id,
                    LLMChatKeys.content: f"{MCPConstants.TOOL_CALL_FAILED_PREFIX}{text_content}"
                    if is_error
                    else text_content,
                }
            )
        return tool_messages

    def _format_as_user_messages(self, tool_results: list[dict[str, Any]]) -> list[dict]:
        """Format tool results as user messages."""
        tool_messages = []
        for tool in tool_results:
            tool_call_id = tool.get(MCPKeys.tool_use_id, "unknown")
            content = tool.get(LLMChatKeys.content, [])
            is_error = tool.get(MCPKeys.is_error, False)

            raw_text = content[0].text
            text_content = format_json_content(raw_text)

            if len(text_content) > 1500:
                text_content = text_content[:1500]

            if is_error:
                content_text = (
                    f"{MCPConstants.TOOL_RESULT_PREFIX}{tool_call_id}{MCPConstants.FAILED_SUFFIX}{text_content}"
                )
            else:
                content_text = f"{MCPConstants.TOOL_RESULT_PREFIX}{tool_call_id}: {text_content}"

            tool_messages.append(
                {
                    LLMChatKeys.role: LLMChatKeys.user_value,
                    LLMChatKeys.content: content_text,
                }
            )
        return tool_messages

    def generate_response(self, container: DataContainer) -> DataContainer:
        """Process text packets and generate responses."""
        self._last_container = container
        self.logger.debug("Chatbot in progress")
        raw_tools = self._get_generic_data(container, self.attributes.generic_key)
        self.available_tools = make_tools_llama_compatible(raw_tools)
        if self.attributes.add_tool_to_prompt:
            self._add_tools_to_system_prompt()

        responses = []
        for packet in container.texts:
            user_id, session_id, prompt = self.prepare_conversation_context(packet)

            if self.partial_query:
                self.partial_query.extend(self._format_tool_results_for_conversation(self.tool_results))
            else:
                if self.system_prompt:
                    system_msg = self.generate_dict_msg(LLMChatKeys.system_value, self.system_prompt)
                    self.partial_query.append(system_msg)

                if self.attributes.chat_history_key:
                    self.partial_query.extend(packet.generic_data.get(self.attributes.chat_history_key, []))

                user_msg = self.generate_dict_msg(LLMChatKeys.user_value, prompt)
                self.partial_query.append(user_msg)

            response = self.infer(self.partial_query)
            self.logger.debug(f"Response is {response}")

            self.logger.debug("End of interaction.")

            if self.tool_calls:
                self._store_conversation_state(packet, response)
            else:
                responses.append(TextPacket(source=session_id, content=response, id=user_id))
                self._cleanup_conversation_state(packet)

        container.texts.extend(responses)
        return container

    def _store_conversation_state(self, packet: TextPacket, response: str) -> None:
        """Store conversation state for tool execution."""
        packet.generic_data[MCPKeys.tool_calls] = self.tool_calls
        packet.generic_data[MCPKeys.partial_query] = self.partial_query
        packet.generic_data[MCPKeys.partial_response] = response
        packet.generic_data.pop(MCPKeys.tool_results, None)

    def _cleanup_conversation_state(self, packet: TextPacket) -> None:
        """Clean up conversation state when done."""
        for key in [MCPKeys.tool_calls, MCPKeys.partial_query, MCPKeys.partial_response, MCPKeys.tool_results]:
            packet.generic_data.pop(key, None)

        self.partial_query = []
        self.partial_response = ""
        self.tool_calls = []
        self.tool_results = []
