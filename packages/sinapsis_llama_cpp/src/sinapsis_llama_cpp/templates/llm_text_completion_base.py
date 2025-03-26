# -*- coding: utf-8 -*-

import uuid
from abc import abstractmethod
from collections import deque
from typing import Literal

from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base import (
    Template,
    TemplateAttributes,
    TemplateAttributeType,
)

from sinapsis_llama_cpp.helpers.llama_keys import (
    LLM_MODEL_TYPE,
    LLMChatKeys,
)


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
    n_ctx: int
    role: Literal["system", "user", "assistant"] = "assistant"
    prompt: str = ""
    system_prompt: str | None = None
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
    def init_llm_model(self) -> LLM_MODEL_TYPE:
        """
        Initializes the LLM model. This method must be implemented by subclasses
        to set up the specific model.

        Returns:
            Llama | Any: The initialized model instance.
        """
        raise NotImplementedError("Must be implemented by the subclass.")

    @abstractmethod
    def get_response(self, input_message: str | list) -> str | None:
        """
        Generates a response from the model based on the provided text input.

        Args:
            input_message (str | list): The input text or prompt to which the model
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
        Obtains a response from the model, handling any errors or issues by resetting
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

    def append_to_context(self, conv_id: str, role: str, content: str | None) -> None:
        """
        Appends a new message to the conversation context for the given `conv_id`.

        Args:
            conv_id (str): The conversation ID.
            role (str): The role of the message sender ('user' or 'assistant').
            content (str): The content of the message.
        """
        if content:
            self.context[conv_id].append({LLMChatKeys.role: role, LLMChatKeys.content: content})

    def return_text_packet(self, packet: list[TextPacket]) -> list[TextPacket]:
        """
        Processes a list of `TextPacket` objects, generating a response for each
        text packet.

        If the packet is empty, it generates a new response based on the prompt.
        Otherwise, it uses the conversation context and appends the response to the
        history.

        Args:
            packet (list[TextPacket]): List of text packets containing conversation
            history or prompts.

        Returns:
            list[TextPacket]: A list of updated text packets with the model's response
            added as content.
        """

        self.logger.debug("Chatbot in progress")
        if packet:
            conv_id = packet[0].id
            prompt = packet[0].content
        else:
            packet = []
            conv_id = str(uuid.uuid4())
            prompt = self.attributes.prompt

        self._set_context(conv_id)
        if self.attributes.system_prompt:
            self.append_to_context(conv_id, LLMChatKeys.system_value, self.attributes.system_prompt)
        self.append_to_context(conv_id, LLMChatKeys.user_value, prompt)
        response = self.infer(list(self.context[conv_id]))
        self.append_to_context(conv_id, LLMChatKeys.assistant_value, response)
        self.logger.debug("End of interaction.")

        packet.append(TextPacket(source=self.instance_name, content=response, id=conv_id))
        return packet

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
        container.texts = self.return_text_packet(container.texts)
        return container
