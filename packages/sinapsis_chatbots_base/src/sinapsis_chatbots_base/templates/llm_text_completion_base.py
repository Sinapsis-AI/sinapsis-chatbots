# -*- coding: utf-8 -*-

import uuid
from abc import abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Literal

from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)

from sinapsis_chatbots_base.helpers.llm_keys import LLMChatKeys


class LLMTextCompletionAttributes(TemplateAttributes):
    """
    Attributes for BaseLLMTextCompletion.

    This class defines the attributes required for the LLM-based text completion
    template.
    It includes configuration settings for model context size, role, prompt, and
    chat format.

    Attributes:
        llm_model_name (str): The name of the LLM model to use.
        n_ctx (int): Maximum context size for the model.
        role (Literal["system", "user", "assistant"]): The role in the conversation,
            such as "system", "user", or
            "assistant". Defaults to "assistant".
        prompt (str): A set of instructions provided to the LLM to guide how to respond.
            The default
            value is an empty string.
        system_prompt (str | None): The prompt that indicates the LLM how to behave
            (e.g. you are an expert on...)
        chat_format (str | None): The format for the chat messages
            (e.g., llama-2, chatml, etc.).
        context_max_len (int): The maximum length for the conversation context.
            The default value is 6.
        pattern (str | None): A regex pattern to match delimiters. The default value is
            `<|...|>` and `</...>`.
        keep_before (bool): If True, returns the portion before the first match;
            if False, returns the portion after the first match.
    """

    llm_model_name: str
    n_ctx: int = 9000
    role: Literal["system", "user", "assistant"] = "assistant"
    generic_key: str = "SourceHistoryAggregator"
    prompt: str = ""
    system_prompt: str | Path | None = None
    chat_format: str = "chatml"
    context_max_len: int = 6
    pattern: str | None = None
    keep_before: bool = True


class LLMTextCompletionBase(Template):
    """
    Base template to get a response message from any LLM.

    This is a base template class for LLM-based text completion. It is designed to work
    with different LLM models (e.g., Llama, GPT). The base functionality includes
    model initialization, response generation, state resetting, and context management.
    Specific model interactions must be implemented in subclasses.

    """

    AttributesBaseModel = LLMTextCompletionAttributes
    UIProperties = UIPropertiesMetadata(category="Chatbots", output_type=OutputTypes.TEXT)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """
        Initializes the base template with the provided attributes and initializes
        the LLM model.

        Args:
            attributes (TemplateAttributeType): Attributes specific to the LLM model.
        """
        super().__init__(attributes)

        self.llm = self.init_llm_model()
        self._clear_context()
        self.system_prompt = self._set_system_prompt()

    def _set_system_prompt(self):
        system_prompt = self.attributes.system_prompt
        if system_prompt and (("/" in system_prompt) or ("\\" in system_prompt) or (system_prompt.endswith(".txt"))):
            return Path(system_prompt).read_text()
        else:
            return system_prompt

    def _set_context(self, conversation_id: str) -> None:
        """
        Sets the context for the specified conversation ID, ensuring that a deque
        for the conversation is available to store conversation history.

        Args:
            conversation_id (str): The unique identifier for the conversation.
        """
        if conversation_id not in self.context:
            self.context[conversation_id] = deque(maxlen=self.attributes.context_max_len)

    def _clear_context(self) -> None:
        """
        Clears the context, resetting the stored conversation history for
        all conversations.
        """
        self.context: dict = {}

    @abstractmethod
    def init_llm_model(self) -> Any:
        """
        Initializes the LLM model. This method must be implemented by subclasses
        to set up the specific model.

        Returns:
            Llama | Any: The initialized model instance.
        """
        raise NotImplementedError("Must be implemented by the subclass.")

    @abstractmethod
    def get_response(self, input_message: str | list | dict) -> str | None:
        """
        Generates a response from the model based on the provided text input.

        Args:
            input_message (str | list | dict): The input text or prompt to which the model
            will respond.

        Returns:
            str | None: The model's response as a string, or None if no response is
            generated.

        This method should be implemented by subclasses to handle the specifics of
        response generation for different models.
        """
        raise NotImplementedError("Must be implemented by the subclass.")

    def reset_llm_state(self) -> None:
        """
        Resets the internal state of the language model, ensuring that no memory,
        context, or cached information from previous interactions persists in the
        current session.

        This method calls `reset()` on the model to clear its internal state and
        `reset_llm_context()` to reset any additional context management mechanisms.


        Subclasses may override this method to implement model-specific reset behaviors
        if needed.
        """
        self.llm.reset()

    def infer(self, text: str | list) -> str | None:
        """
        Gets a response from the model, handling any errors or issues by resetting
        the model state if necessary.

        Args:
            text (str): The input text for which the model will generate a response.

        Returns:
            str | None: The model's response as a string or None if the model fails
            to respond.
        """
        try:
            return self.get_response(text)
        except ValueError:
            self.reset_llm_state()
            return self.get_response(text)

    def append_to_context(self, conv_id: str, message: dict) -> None:
        """
        Appends a new message to the conversation context for the given `conv_id`.

        Args:
            conv_id (str): The conversation ID.
            message (dict): The message dictionary with role and content keys. Role can be
            'system', 'assistant' or 'user'.
        """
        if message:
            self.context[conv_id].append(message)

    @staticmethod
    def generate_dict_msg(role: str, msg_content: str | list | None) -> dict:
        """For the provided content, generate a dictionary to be appended as the context
        for the response.
        Args:
            role (str): Role of the message, Can be system, user or assistant
            msg_content (str | list | None): Content of the message to be passed to the llm.
        Returns:
            The dictionary with the key pair values for role and content.

        """
        return {LLMChatKeys.role: role, LLMChatKeys.content: msg_content}

    def get_conv_id_and_content(self, packet: TextPacket) -> tuple[str, str]:
        """For an input packet, return a conversation id and the content of the message.
        If the incoming packet does not have id, generate a new one.

        Args:
            packet(TextPacket): the incoming packet
        Returns:
            tuple[str, str]: The pair of conversation_id and message
        """
        if packet:
            conv_id = packet.source
            prompt = packet.content
        else:
            conv_id = str(uuid.uuid4())
            prompt = self.attributes.prompt

        return conv_id, prompt

    def generate_response(self, container: DataContainer) -> DataContainer:
        """
        Processes a list of `TextPacket` objects, generating a response for each
        text packet.

        If the packet is empty, it generates a new response based on the prompt.
        Otherwise, it uses the conversation context and appends the response to the
        history.

        Args:
            container (DataContainer): Container where the incoming message is located and
            where the generated response will be appended.

        Returns:
            DataContainer: Updated DataContainer with the response from the llm.
        """

        self.logger.debug("Chatbot in progress")
        responses = []
        full_context = []
        contexts = self._get_generic_data(container, self.attributes.generic_key)
        for context in contexts:
            context_message = self.generate_dict_msg(LLMChatKeys.user_value, context)
            full_context.append(context_message)
        for packet in container.texts:
            conv_id, prompt = self.get_conv_id_and_content(packet)
            if self.system_prompt:
                system_prompt_msg = self.generate_dict_msg(LLMChatKeys.system_value, self.system_prompt)
                full_context.append(system_prompt_msg)

            message = self.generate_dict_msg(LLMChatKeys.user_value, prompt)

            full_context.append(message)
            response = self.infer(full_context)
            _ = self.generate_dict_msg(LLMChatKeys.assistant_value, response)
            self.logger.debug("End of interaction.")

            responses.append(TextPacket(source=conv_id, content=response, id=conv_id))

        container.texts.extend(responses)
        return container

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Executes the LLMChatTemplate by processing the input `DataContainer`
        and generating a response.

        This method is responsible for handling the conversation flow, processing the input,
        and returning a response. It also ensures that the model has a prompt or previous conversation
        to work with.

        Args:
            container (DataContainer): Input data container containing texts.

        Returns:
            DataContainer: The output data container with the model's response added to the `texts` attribute.
        """
        if not container.texts and not self.attributes.prompt:
            self.logger.debug("No need to process response.")
            return container

        container = self.generate_response(container)
        return container
