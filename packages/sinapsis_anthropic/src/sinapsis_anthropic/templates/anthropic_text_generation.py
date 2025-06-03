# -*- coding: utf-8 -*-
from anthropic import Anthropic
from sinapsis_chatbots_base.templates.llm_text_completion_base import (
    LLMTextCompletionAttributes,
    LLMTextCompletionBase,
)
from sinapsis_core.template_base.template import TemplateAttributeType

from sinapsis_anthropic.helpers.anthropic_keys import AnthropicKeys
from sinapsis_anthropic.helpers.env_var_keys import ANTHROPIC_API_KEY


class AnthropicAttributes(LLMTextCompletionAttributes):
    """
    Attributes for Anthropic text and code generation template

    llm_model_name (str): The name of the LLM model to use.
    role (Literal["system", "user", "assistant"]): The role in the conversation, such as
        "system", "user", or "assistant". Defaults to "assistant".
    prompt (str): A set of instructions provided to the LLM to guide how to respond.
        The default value is an empty string.
    system_prompt (str | None): The prompt that indicates the LLM how to behave
        (e.g. you are an expert on...)
    context_max_len (int): The maximum length for the conversation context.
        The default value is 6.
    budget_tokens (int): The maximum number of tokens to allocate for intermediate
        thinking steps when `extended_thinking` is enabled. Defaults to 2000.
    extended_thinking (bool): A flag indicating whether to display "thinking" content
        blocks in the response. If `True`, the model will include intermediate "thinking"
        steps in its response. Defaults to `False`.
    max_tokens (int): Maximum number of tokens to generate.
    temperature (float): Sampling temperature for the model.
    web_search (bool): A boolean flag indicating whether web search should be enabled.
        If `True`, the LLM will have access to web search tools to retrieve external
        information. Defaults to `False`.
    """

    budget_tokens: int = 2000
    extended_thinking: bool = False
    max_tokens: int = 4000
    temperature: float = 1.0
    web_search: bool = False


class AnthropicTextGeneration(LLMTextCompletionBase):
    """
    A class to interact with the Anthropic API for text and code generation.

    This class provides methods to initialize the Anthropic client, reset its state,
    and generate responses based on input messages. It leverages the Anthropic model
    for text and code generation and allows for dynamic interaction with the API.

    Example configuration for agent setup:

    agent:
      name: my_claude_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
    - template_name: AnthropicTextGeneration
      class_name: AnthropicTextGeneration
      template_input: InputTemplate
      attributes:
        llm_model_name: claude-3-7-sonnet-latest
        max_tokens: 1024
        temperature: 1
        web_search: False
        extended_thinking: False
    """

    AttributesBaseModel = AnthropicAttributes

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.system_prompt = None
        self.system = self._set_system_prompt()

    def init_llm_model(self) -> Anthropic:
        """
        Initializes the Anthropic client using ANTHROPIC_API_KEY

        Returns:
            Anthropic: An initialized instance of the Anthropic client.
        """
        try:
            return Anthropic(api_key=ANTHROPIC_API_KEY)
        except TypeError:
            self.logger.error("Invalid API key")

    def reset_llm_state(self) -> None:
        """
        Resets the internal state of the language model, ensuring that no memory,
        context, or cached information from previous interactions persists in the
        current session.
        """
        self.llm = self.init_llm_model()

    def build_create_args(self, input_message: str | list) -> dict:
        """
        Builds the arguments required for making a request to the LLM model.

        This method constructs the dictionary of parameters needed for the model's
        `create()` method based on the provided input message and the object's attributes.

        Args:
            input_message (str | list): The input text or prompt to send to the model.

        Returns:
            dict: The dictionary containing the parameters for the model's request.
        """
        return {
            AnthropicKeys.model: self.attributes.llm_model_name,
            AnthropicKeys.max_tokens: self.attributes.max_tokens,
            AnthropicKeys.messages: input_message,
            AnthropicKeys.system: self.system if self.system else "",
            AnthropicKeys.temperature: self.attributes.temperature,
            **(
                {
                    AnthropicKeys.tools: [
                        {
                            AnthropicKeys.name: AnthropicKeys.web_search,
                            AnthropicKeys.type: AnthropicKeys.web_search_20250305,
                        }
                    ]
                }
                if self.attributes.web_search
                else {}
            ),
            **(
                {
                    AnthropicKeys.thinking: {
                        AnthropicKeys.type: AnthropicKeys.enabled,
                        AnthropicKeys.budget_tokens: self.attributes.budget_tokens,
                    }
                }
                if self.attributes.extended_thinking
                else {}
            ),
        }

    def extract_response_text(self, message_response) -> str:
        """
        Extracts the response text from the message response.

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
                if self.attributes.extended_thinking:
                    response_parts.append("\n✓ FINAL ANSWER:\n")
                response_parts.append(content_block.text)

        return "".join(response_parts)

    def get_response(self, input_message: str | list) -> str | None:
        """
        Generates a response from the model based on the provided text input.

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

        self.logger.debug(response)

        return response
