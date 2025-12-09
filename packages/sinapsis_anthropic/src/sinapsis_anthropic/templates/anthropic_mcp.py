# -*- coding: utf-8 -*-
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock
from sinapsis_chatbots_base.helpers.llm_keys import LLMChatKeys, MCPKeys
from sinapsis_chatbots_base.helpers.tags import Tags
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base.base_models import TemplateAttributeType

from sinapsis_anthropic.helpers.anthropic_keys import AnthropicKeys
from sinapsis_anthropic.helpers.mcp_tool_helper import make_tools_anthropic_compatible
from sinapsis_anthropic.templates.anthropic_text_generation import AnthropicAttributes, AnthropicTextGeneration

AnthropicMultiModalUIProperties = AnthropicTextGeneration.UIProperties
AnthropicMultiModalUIProperties.tags.extend([Tags.MCP])


class AnthropicWithMCP(AnthropicTextGeneration):
    """Template for chat processing using Anthropic's Claude models with MCP tool support.

    This template extends AnthropicTextGeneration to handle multi-turn conversations
    involving tool use (function calling) following the Model Context Protocol (MCP).
    It processes tool definitions, sends them to the model, parses tool use requests,
    stores intermediate state, and handles incoming tool results for follow-up calls.

    Usage example:

    agent:
      name: my_claude_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
    - template_name: AnthropicWithMCP
      class_name: AnthropicWithMCP
      template_input: InputTemplate
      attributes:
        init_args:
          llm_model_name: claude-3-opus-20240229
        completion_args:
          max_tokens: 1024
          temperature: 1
          top_p: 0.95
          top_k: 40
          service_tier: standard_only
          stop_sequences: null
        chat_history_key: null
        rag_context_key: null
        system_prompt: null
        pattern: null
        keep_before: true
        extended_thinking:
          type: disabled
        web_search: false
        tools_key: "MCP"
    """

    UIProperties = AnthropicMultiModalUIProperties

    class AttributesBaseModel(AnthropicAttributes):
        """Attributes for Anthropic MCP template.

        Attributes:
            init_args (LLMInitArgs): Model arguments, including the 'llm_model_name'.
            completion_args (AnthropicThinkingArgs): Anthropic-specific sampling args
                (e.g., max_tokens, temperature, service_tier, stop_sequences).
            chat_history_key (str | None): Key in the packet's generic_data to find
                the conversation history.
            rag_context_key (str | None): Key in the packet's generic_data to find
                RAG context to inject.
            system_prompt (str | Path | None): The system prompt (or path to one)
                to instruct the model.
            pattern (str | None): A regex pattern used to post-process the model's response.
            keep_before (bool): If True, keeps text before the 'pattern' match; otherwise,
                keeps text after.
            extended_thinking (AnthropicThinkingArgs): Configuration for enabling or
                disabling the extended "thinking" feature.
            tools_key (str): Key used to extract the raw tools from the data container. Defaults
                to `""`.
        """

        tools_key: str = ""

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.available_tools: list = []
        self.tool_results: list = []
        self.tool_calls: list = []
        self.partial_response: str = ""
        self.partial_query: list = []
        self.assistant_content: list = []

    def get_extra_context(self, packet: TextPacket) -> str | None:
        """Retrieves tool context data from packet metadata using the configured tools_context_key.

        Searches the packet's generic_data dictionary for the specified key. If found and contains
        data, joins multiple context items into a single newline-delimited string. Returns None
        if no key is configured, the key is missing, or the context data is empty.

        Args:
            packet (TextPacket): The incoming TextPacket containing potential context data in its generic_data.

        Returns:
            str | None: Newline-joined context strings when RAG data exists, or None when
                rag_context_key is unset, missing, or points to empty data
        """
        self.tool_results = packet.generic_data.get(MCPKeys.tool_results, [])
        self.partial_response = packet.generic_data.get(MCPKeys.partial_response, "")
        self.partial_query = packet.generic_data.get(MCPKeys.partial_query, [])
        if not self.available_tools and not self.tool_results:
            self.logger.error("No tools found on text packet's generic data")

        return super().get_extra_context(packet)

    def build_create_args(self, input_message: str | list) -> dict:
        """Builds the arguments required for making a request to the LLM model.

        This method constructs the dictionary of parameters needed for the model's
        `create()` method based on the provided input message and the object's attributes.

        Args:
            input_message (str | list): The input text or prompt to send to the model.

        Returns:
            dict: The dictionary containing the parameters for the model's request.
        """
        create_args = {
            AnthropicKeys.model: self.attributes.init_args.llm_model_name,
            AnthropicKeys.messages: input_message,
            AnthropicKeys.tools: self.available_tools,
        }
        if self.system_prompt:
            create_args[AnthropicKeys.system] = self.system_prompt

        if self.attributes.extended_thinking.type == AnthropicKeys.enabled:
            create_args[AnthropicKeys.thinking] = self.attributes.extended_thinking.model_dump()

        create_args.update(self.attributes.completion_args.model_dump(exclude_none=True))
        return create_args

    def extract_response_text(self, message_response: Message) -> str:
        """Extracts the response text from the message response.

        This method iterates over the content blocks of the message response and
        concatenates the text from content blocks of type "text".

        Args:
            message_response (Message): The response object containing the content blocks.

        Returns:
            str: The concatenated text response from the content blocks.
        """
        self.tool_calls = []
        self.assistant_content = []

        response_parts: list = []
        has_tool_calls = False
        for content_block in message_response.content:
            if content_block.type == AnthropicKeys.thinking:
                self._handle_thinking_block(content_block, response_parts)
            elif content_block.type == AnthropicKeys.text:
                self._handle_text_block(content_block, response_parts)
            elif content_block.type == MCPKeys.tool_use:
                has_tool_calls = True
                self._handle_tool_use_block(content_block, response_parts)

        current_response = "\n".join(response_parts)

        if self.partial_response and has_tool_calls:
            self.partial_response = f"{self.partial_response}\n{current_response}"
            return self.partial_response
        elif self.partial_response and not has_tool_calls:
            final_response = f"{self.partial_response}\n{current_response}"
            self.partial_response = ""
            return final_response
        else:
            return current_response

    @staticmethod
    def _handle_thinking_block(content_block: ThinkingBlock, response_parts: list) -> None:
        """Processes and formats a 'thinking' content block.

        Args:
            content_block (ThinkingBlock): The thinking block object from Anthropic response.
            response_parts (list): The list accumulating response text parts.
        """
        response_parts.append("\n🧠 THINKING BLOCK:\n")
        thinking_text = (
            content_block.thinking[:500] + "..." if len(content_block.thinking) > 500 else content_block.thinking
        )
        response_parts.append(thinking_text)

    def _handle_text_block(self, content_block: TextBlock, response_parts: list) -> None:
        """Processes a 'text' content block and adds it to assistant content state.

        Args:
            content_block (TextBlock): The text block object from Anthropic response.
            response_parts (list): The list accumulating response text parts.
        """
        if self.attributes.extended_thinking.type == AnthropicKeys.enabled:
            response_parts.append("\n✓ FINAL ANSWER:\n")
        response_parts.append(content_block.text)
        self.assistant_content.append(content_block)

    def _handle_tool_use_block(self, content_block: ToolUseBlock, response_parts: list) -> None:
        """Processes a 'tool_use' block, formats a placeholder text, and stores tool call info.

        Args:
            content_block (ToolUseBlock): The tool use block object from Anthropic response.
            response_parts (list): The list accumulating response text parts.
        """
        tool_name, tool_args = content_block.name, content_block.input

        response_parts.append(f"[Calling tool {tool_name} with args {tool_args}]")
        self.assistant_content.append(content_block)
        self.tool_calls.append(
            {
                MCPKeys.tool_name: tool_name,
                MCPKeys.args: tool_args,
                MCPKeys.tool_use_id: content_block.id,
            }
        )

    def num_elements(self) -> int:
        """Determines if the MCP loop should continue based on pending tool calls.

        Checks if the last processed container's text packet contains tool calls
        that need to be executed.

        Returns:
            int: Returns 1 if tool calls are pending (continue loop), -1 otherwise (stop loop).
        """
        if hasattr(self, MCPKeys.last_container):
            for packet in self._last_container.texts:
                if MCPKeys.tool_calls in packet.generic_data:
                    return 1
        return -1

    def generate_response(self, container: DataContainer) -> DataContainer:
        """Handles the multi-turn MCP conversation flow.

        Retrieves tools, builds the appropriate prompt (initial or with tool results),
        calls the model via `infer`, and then either stores state for the next turn
        (if tool calls were made) or cleans up state and returns the final response.

        Args:
            container (DataContainer): The input container, potentially containing tool results
                                       or the initial user query.

        Returns:
            DataContainer: The updated container. If tool calls are pending, it includes
                           `tool_calls` in generic_data. If the conversation is complete,
                           it includes the final text response.
        """
        self._last_container = container
        self.logger.debug("Chatbot in progress")
        responses = []
        raw_tools = self._get_generic_data(container, self.attributes.tools_key)
        self.available_tools = make_tools_anthropic_compatible(raw_tools)

        for packet in container.texts:
            user_id, session_id, prompt = self.prepare_conversation_context(packet)
            if self.partial_query:
                self.partial_query.append(self.generate_dict_msg(LLMChatKeys.user_value, self.tool_results))
            else:
                if self.attributes.chat_history_key:
                    self.partial_query.extend(packet.generic_data.get(self.attributes.chat_history_key, []))

                message = self.generate_dict_msg(LLMChatKeys.user_value, prompt)
                self.partial_query.append(message)

            response = self.infer(self.partial_query)
            self.logger.debug("End of interaction.")

            if self.tool_calls:
                self.partial_query.append(self.generate_dict_msg(LLMChatKeys.assistant_value, self.assistant_content))
                self._store_conversation_state(packet, response)
            else:
                responses.append(TextPacket(source=session_id, content=response, id=user_id))
                self._cleanup_conversation_state(packet)

        container.texts.extend(responses)
        return container

    def _store_conversation_state(self, packet: TextPacket, response: str) -> None:
        """Stores intermediate MCP state (tool calls, query history, partial response) in the packet's generic_data.

        Args:
            packet (TextPacket): The text packet to store state within.
            response (str): The partial text response generated so far.
        """
        packet.generic_data[MCPKeys.tool_calls] = self.tool_calls
        packet.generic_data[MCPKeys.partial_query] = self.partial_query
        packet.generic_data[MCPKeys.partial_response] = response
        packet.generic_data.pop(MCPKeys.tool_results, None)

    def _cleanup_conversation_state(self, packet: TextPacket) -> None:
        """Clears MCP state variables on self and removes them from the packet's generic_data.

        Args:
            packet (TextPacket): The text packet from which to remove state data.
        """
        self.partial_query = []
        self.partial_response = ""
        self.tool_calls = []
        self.assistant_content = []
        self.tool_results = []
        packet.generic_data.pop(MCPKeys.tool_calls, None)
        packet.generic_data.pop(MCPKeys.partial_query, None)
        packet.generic_data.pop(MCPKeys.partial_response, None)
        packet.generic_data.pop(MCPKeys.tool_results, None)
