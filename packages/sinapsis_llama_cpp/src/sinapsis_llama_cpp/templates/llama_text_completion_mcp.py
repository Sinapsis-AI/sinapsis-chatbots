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
    """Template for LLaMA text completion with MCP tool integration.

    Extends LLaMATextCompletion to handle multi-turn conversations involving
    tool use (function calling) based on the Model Context Protocol (MCP).
    It parses tool definitions, potentially modifies the system prompt, detects
    tool calls from the model (either via native support or text parsing),
    manages state across turns, and formats tool results for follow-up calls.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: LLaMATextCompletion
      class_name: LLaMATextCompletion
      template_input: InputTemplate
      attributes:
        init_args:
          llm_model_name: 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'
          llm_model_file: 'mistral-7b-instruct-v0.2.Q2_K.gguf'
          n_gpu_layers: -1
          use_mmap: true
          use_mlock: false
          seed: 42
          n_ctx: 8192
          n_batch: 512
          n_ubatch: 512
          n_threads: null
          n_threads_batch: null
          flash_attn: true
          chat_format: null
          verbose: true
        completion_args:
          temperature: 0.2
          top_p: 0.95
          top_k: 40
          max_tokens: 4096
          min_p: 0.05
          stop: null
          seed: 42
          repeat_penalty: 1.0
          presence_penalty: 0.0
          frequency_penalty: 0.0
          logit_bias: null
        chat_history_key: null
        rag_context_key: null
        system_prompt: You are an expert in AI.
        pattern: null
        keep_before: true
        tools_key: "Tools"
        max_tool_retries: 3
        add_tool_to_prompt: true
    """

    UIProperties = LLaMAMultiModalUIProperties
    system_prompt: str | None

    class AttributesBaseModel(LLaMATextCompletion.AttributesBaseModel):
        """Attributes for LLaMA-CPP MCP template.

        Attributes:
            init_args (LLaMAInitArgs): LLaMA model arguments, including the 'llm_model_name'.
            completion_args (LLaMACompletionArgs): LLaMA generation arguments, including
                'max_tokens', 'temperature', 'top_p', and 'top_k', among others.
            chat_history_key (str | None): Key in the packet's generic_data to find
                the conversation history.
            rag_context_key (str | None): Key in the packet's generic_data to find
                RAG context to inject.
            system_prompt (str | Path | None): The system prompt (or path to one)
                to instruct the model.
            pattern (str | None): A regex pattern used to post-process the model's response.
            keep_before (bool): If True, keeps text before the 'pattern' match; otherwise,
                keeps text after.
            tools_key (str): Key used to extract the raw tools from the data container. Defaults
                to `""`.
            max_tool_retries (int): Maximum consecutive tool execution failures before stopping. Defaults to `3`.
            add_tool_to_prompt (bool): Whether to automatically append tool descriptions to the system prompt.
                Defaults to `True`.
        """

        tools_key: str = ""
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
        """Generates a response from the model, handling potential tool calls and partial responses.

        Args:
            input_message (str | list): The input messages list for the LLM.

        Returns:
            str | None: The final response text, the accumulated partial response text
                        if tool calls are pending, or None/error string if generation failed.
        """
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

    def _create_chat_completion(self, input_message: str | list) -> CreateChatCompletionResponse:
        """Calls the LLaMA model to create a chat completion, handling potential errors and retries.

        Args:
            input_message (str | list): The list of messages to send to the model.

        Returns:
            CreateChatCompletionResponse | None: The response object from llama-cpp,
                                                 or None if generation failed after retries.
        """
        completion_args = self.attributes.completion_args.model_dump(exclude_none=True)
        try:
            if self.attributes.init_args.chat_format == LLaMAModelKeys.chatml_function_calling:
                return self.llm.create_chat_completion(
                    messages=input_message, tools=self.available_tools, tool_choice="auto", **completion_args
                )
            return self.llm.create_chat_completion(messages=input_message, **completion_args)
        except (IndexError, AttributeError):
            self.reset_llm_state()
            return self._create_chat_completion(input_message)

    def _process_chat_completion(self, chat_completion: CreateChatCompletionResponse) -> str:
        """Processes the raw chat completion response to extract text and tool calls.

        Args:
            chat_completion (CreateChatCompletionResponse): The response object from llama-cpp.

        Returns:
            str: The extracted response text, potentially including placeholder text
                 for detected tool calls. Returns an empty string if processing fails.
        """
        chat_completion = cast(CreateChatCompletionResponse, chat_completion)
        llm_response_choice = chat_completion[LLMChatKeys.choices][0]
        finish_reason = llm_response_choice[MCPKeys.finish_reason]
        message = llm_response_choice[LLMChatKeys.message]

        self.tool_calls = []

        if finish_reason == MCPKeys.tool_calls:
            return self._handle_function_calling_response(message)
        return self._handle_regular_response(message)

    def _handle_function_calling_response(self, message: dict) -> str:
        """Handles responses using the native chatml-function-calling format.

        Args:
            message (dict[str, Any]): The message dictionary from the LLM response.

        Returns:
            str: The response text, including placeholders for tool calls made.
        """
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
        """Handles responses by extracting potential tool calls from the text content.

        Args:
            message (dict[str, Any]): The message dictionary from the LLM response.

        Returns:
            str: The original response text if no tools are extracted, or the text
                 including placeholders if tool calls were found. Returns empty string on error.
        """
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
        """Loads MCP state from the packet and calls the parent method for RAG context.

        Args:
            packet (TextPacket): The incoming text packet containing potential state data.

        Returns:
            str | None: RAG context string from the parent, or None.
        """
        self.tool_results = packet.generic_data.get(MCPKeys.tool_results, [])
        self.partial_response = packet.generic_data.get(MCPKeys.partial_response, "")
        self.partial_query = packet.generic_data.get(MCPKeys.partial_query, [])

        if not self.available_tools and not self.tool_results:
            self.logger.error("No tools found on text packet's generic data")
        return super().get_extra_context(packet)

    def num_elements(self) -> int:
        """Controls WhileLoop continuation based on pending tool calls and retries.

        Returns:
            int: 1 to continue loop (pending calls within retry limit), -1 to stop.
        """
        if not self._has_pending_tool_calls():
            return -1

        failure_count = sum(1 for result in self.tool_results if result.get(MCPKeys.is_error, False))
        if failure_count >= self.attributes.max_tool_retries:
            self.logger.warning(f"Stopping loop after {failure_count} consecutive tool failures")
            return -1
        return 1

    def _has_pending_tool_calls(self) -> bool:
        """Checks if the last processed container indicates pending tool calls.

        Returns:
            bool: True if `tool_calls` key exists in the first text packet's
                  generic_data of the last container, False otherwise.
        """
        if not hasattr(self, MCPKeys.last_container) or not self._last_container:
            return False
        return any(MCPKeys.tool_calls in packet.generic_data for packet in self._last_container.texts)

    def _format_tool_results_for_conversation(self, tool_results: list[dict[str, Any]]) -> list[dict]:
        """Formats tool execution results into messages suitable for the LLM conversation history.

        Args:
            tool_results (list[dict[str, Any]]): A list of tool result dictionaries, expected
                                                 to contain 'tool_use_id', 'content', and optionally 'is_error'.

        Returns:
            list[dict]: A list of messages formatted for the conversation history
                        (either as 'tool' role or 'user' role).
        """
        if self.attributes.init_args.chat_format == LLaMAModelKeys.chatml_function_calling:
            return self._format_as_tool_messages(tool_results)
        return self._format_as_user_messages(tool_results)

    @staticmethod
    def _format_as_tool_messages(tool_results: list[dict[str, Any]]) -> list[dict]:
        """Formats tool results as 'tool' role messages for chatml-function-calling.

        Args:
            tool_results (list[dict[str, Any]]): A list of tool result dictionaries.

        Returns:
            list[dict]: A list of messages formatted with role 'tool', including 'tool_call_id'.
        """
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

    @staticmethod
    def _format_as_user_messages(tool_results: list[dict[str, Any]]) -> list[dict]:
        """Formats tool results as 'user' role messages for regular chat formats.

        Args:
            tool_results (list[dict[str, Any]]): A list of tool result dictionaries.

        Returns:
            list[dict]: A list of messages formatted with role 'user', containing formatted result text.
        """
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
        """Processes text packets, handles MCP state, generates responses including tool calls.

        Args:
            container (DataContainer): Input container, possibly with tool results in packet generic_data.

        Returns:
            DataContainer: Updated container. If tool calls are pending, packet generic_data
                           contains state. If final, container.texts holds the response.
        """
        self._last_container = container
        self.logger.debug("Chatbot in progress")
        raw_tools = self._get_generic_data(container, self.attributes.tools_key)
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
        """Stores intermediate MCP state in the provided packet's generic_data.

        Args:
            packet (TextPacket): The text packet associated with the current turn to store state within.
            response (str | None): The partial text response generated in this turn, or None.
        """
        packet.generic_data[MCPKeys.tool_calls] = self.tool_calls
        packet.generic_data[MCPKeys.partial_query] = self.partial_query
        packet.generic_data[MCPKeys.partial_response] = response
        packet.generic_data.pop(MCPKeys.tool_results, None)

    def _cleanup_conversation_state(self, packet: TextPacket) -> None:
        """Clears MCP state variables on self and removes them from the provided packet's generic_data.

        Args:
            packet (TextPacket): The text packet from which to remove state keys.
        """
        for key in [MCPKeys.tool_calls, MCPKeys.partial_query, MCPKeys.partial_response, MCPKeys.tool_results]:
            packet.generic_data.pop(key, None)

        self.partial_query = []
        self.partial_response = ""
        self.tool_calls = []
        self.tool_results = []
