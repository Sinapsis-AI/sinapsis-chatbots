# -*- coding: utf-8 -*-
from typing import Literal

from anthropic import Anthropic
from anthropic.types import Message
from pydantic import BaseModel, Field
from sinapsis_chatbots_base.helpers.tags import Tags
from sinapsis_chatbots_base.templates.llm_text_completion_base import (
    LLMCompletionArgs,
    LLMTextCompletionAttributes,
    LLMTextCompletionBase,
)

from sinapsis_anthropic.helpers.anthropic_keys import AnthropicKeys
from sinapsis_anthropic.helpers.env_var_keys import AnthropicEnvVars

AnthropicTextGenerationUIProperties = LLMTextCompletionBase.UIProperties
AnthropicTextGenerationUIProperties.tags.extend([Tags.ANTHROPIC])


class AnthropicCompletionArgs(LLMCompletionArgs):
    """Anthropic-specific generation arguments, inheriting from the base LLMCompletionArgs.

    Attributes:
        temperature (float): Controls randomness. 0.0 = deterministic, >0.0 = random. Defaults to `0.2`.
        top_p (float): Nucleus sampling. Considers tokens with cumulative probability >= top_p. Defaults to `0.95`.
        top_k (int): Top-k sampling. Considers the top 'k' most probable tokens. Defaults to `40`.
        max_tokens (int): The maximum number of new tokens to generate.
        service_tier (Literal["auto", "standard_only"]): Specifies the service tier for the request. Defaults to
            `'standard_only'`.
        stop_sequences (list[str] | None): Custom text sequences that will cause the model to stop generating. Defaults
            to `None`.
    """

    max_tokens: int
    service_tier: Literal["auto", "standard_only"] = "standard_only"
    stop_sequences: list[str] | None = None


class AnthropicThinkingArgs(BaseModel):
    """Controls Anthropic's "extended thinking" feature.

    This allows the model to generate intermediate 'thinking' blocks
    before providing a final answer, which can improve quality on complex tasks.

    Attributes:
        budget_tokens (int): The max tokens to use for internal reasoning. Must be ≥1024 and less than `max_tokens`.
            Defaults to `2048`.
        type (Literal["enabled", "disabled"]): To disable or enable extended thinking. Defaults to `'disabled'`.
    """

    budget_tokens: int = 2048
    type: Literal["enabled", "disabled"] = "disabled"


class AnthropicAttributes(LLMTextCompletionAttributes):
    """Base attributes for Anthropic templates.

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
    """

    completion_args: AnthropicCompletionArgs
    extended_thinking: AnthropicThinkingArgs = Field(default_factory=AnthropicThinkingArgs)


class AnthropicTextGeneration(LLMTextCompletionBase):
    """A class to interact with the Anthropic API for text and code generation.

    This class provides methods to initialize the Anthropic client, reset its state,
    and generate responses based on input messages. It leverages the Anthropic model
    for text and code generation and allows for dynamic interaction with the API.

    Usage example:

    agent:
      name: my_claude_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
    - template_name: AnthropicTextGeneration
      class_name: AnthropicTextGeneration
      template_input: InputTemplate
      attributes:
        init_args:
          llm_model_name: claude-3-7-sonnet-latest
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
    """

    class AttributesBaseModel(AnthropicAttributes):
        """Attributes for Anthropic text and code generation template.

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
            web_search (bool): If True, enables the web search tool for the model.
        """

        web_search: bool = False

    UIProperties = AnthropicTextGenerationUIProperties

    def init_llm_model(self) -> Anthropic:
        """Initializes the Anthropic client using ANTHROPIC_API_KEY.

        Returns:
            Anthropic: An initialized instance of the Anthropic client.
        """
        try:
            return Anthropic(api_key=AnthropicEnvVars.ANTHROPIC_API_KEY.value)
        except TypeError:
            self.logger.error("Invalid API key")

    def reset_llm_state(self) -> None:
        """Resets the internal state, ensuring that no memory, context, or cached information persists."""
        self.llm = self.init_llm_model()

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
        }
        if self.system_prompt:
            create_args[AnthropicKeys.system] = self.system_prompt

        if self.attributes.extended_thinking.type == AnthropicKeys.enabled:
            create_args[AnthropicKeys.thinking] = self.attributes.extended_thinking.model_dump()

        if self.attributes.web_search:
            create_args[AnthropicKeys.tools] = [
                {
                    AnthropicKeys.name: AnthropicKeys.web_search,
                    AnthropicKeys.type: AnthropicKeys.web_search_20250305,
                }
            ]
        create_args.update(self.attributes.completion_args.model_dump(exclude_none=True))
        return create_args

    def extract_response_text(self, message_response: Message) -> str:
        """Extracts the response text from the message response.

        This method iterates over the content blocks of the message response and
        concatenates the text from content blocks of type "text".

        Args:
            message_response: The response object containing the content blocks.

        Returns:
            str: The concatenated text response from the content blocks.
        """
        response_parts = []

        for content_block in message_response.content:
            if content_block.type == AnthropicKeys.thinking:
                response_parts.append("\n🧠 THINKING BLOCK:\n")
                response_parts.append(
                    content_block.thinking[:500] + "..."
                    if len(content_block.thinking) > 500
                    else content_block.thinking
                )
            elif content_block.type == AnthropicKeys.text:
                if self.attributes.extended_thinking.type == AnthropicKeys.enabled:
                    response_parts.append("\n✓ FINAL ANSWER:\n")
                response_parts.append(content_block.text)

        return "".join(response_parts)

    def get_response(self, input_message: str | list) -> str | None:
        """Generates a response from the model based on the provided text input.

        This method sends the input text to the model and receives a response.

        Args:
            input_message (list): The input text or prompt to which the model
            will respond.

        Returns:
            str|None: The model's response as a string, or None if no response
                is generated.
        """
        self.logger.debug(f"Query is {input_message}")
        create_args = self.build_create_args(input_message)
        try:
            message_response = self.llm.messages.create(**create_args)
        except IndexError:
            self.reset_llm_state()
            message_response = self.llm.messages.create(**create_args)

        response = self.extract_response_text(message_response)

        self.logger.debug(f"Anthropic model response: {response}")

        return response
