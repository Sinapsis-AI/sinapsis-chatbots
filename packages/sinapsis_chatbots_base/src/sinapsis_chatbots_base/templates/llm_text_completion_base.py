# -*- coding: utf-8 -*-

from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal

import torch
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)

from sinapsis_chatbots_base.helpers.llm_keys import LLMChatKeys
from sinapsis_chatbots_base.helpers.tags import Tags


class LLMTextCompletionAttributes(TemplateAttributes):
    """Configuration attributes for LLM-based text completion templates.

    This class defines all configurable parameters required for LLM text completion,
    including model settings, conversation context management, and prompt handling.

    Attributes:
        llm_model_name (str): The name of the LLM model to use.
        n_ctx (int): Maximum context size for the model.
        role (Literal["system", "user", "assistant"]): The role in the conversation,
            such as "system", "user", or "assistant". Defaults to "assistant".
        chat_history_key (str): Key to identify chat history in each text packet's generic data.
        prompt (str | None): A set of instructions provided to the LLM to guide how to respond.
            The default
            value is an empty string.
        system_prompt (str | None): The prompt that indicates the LLM how to behave
            (e.g. you are an expert on...)
        chat_format (str | None): The format for the chat messages
            (e.g., llama-2, chatml, etc.).
        pattern (str | None): A regex pattern to match delimiters. The default value is
            `<|...|>` and `</...>`.
        keep_before (bool): If True, returns the portion before the first match;
            if False, returns the portion after the first match.
    """

    llm_model_name: str
    n_ctx: int = 9000
    role: Literal["system", "user", "assistant"] = "assistant"
    chat_history_key: str | None = None
    rag_context_key: str | None = None
    prompt: str | None = None
    system_prompt: str | Path | None = None
    chat_format: str = "chatml"
    pattern: str | None = None
    keep_before: bool = True


class LLMTextCompletionBase(Template):
    """Base template to get a response message from any LLM.

    This is a base template class for LLM-based text completion. It is designed to work
    with different LLM models (e.g., Llama, GPT). The base functionality includes
    model initialization, response generation, state resetting, and context management.
    Specific model interactions must be implemented in subclasses.
    """

    AttributesBaseModel = LLMTextCompletionAttributes
    UIProperties = UIPropertiesMetadata(
        category="Chatbots",
        output_type=OutputTypes.TEXT,
        tags=[Tags.CHATBOTS, Tags.CONTEXT, Tags.LLM, Tags.TEXT_COMPLETION, Tags.TEXT],
    )

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)

        self.llm = self.init_llm_model()
        self.system_prompt = self._set_system_prompt()

    def _set_system_prompt(self) -> str:
        """Loads and returns the system prompt from a file if specified, otherwise returns the direct prompt.

        If the system_prompt attribute is a file path (determined by containing slashes or ending with .txt),
        reads the content from the file. Otherwise, returns the system_prompt as-is.

        Returns:
            str: The system prompt content, either loaded from file or taken directly from attributes.
        """
        system_prompt = self.attributes.system_prompt
        if system_prompt and (("/" in system_prompt) or ("\\" in system_prompt) or (system_prompt.endswith(".txt"))):
            return Path(system_prompt).read_text()
        else:
            return system_prompt

    @abstractmethod
    def init_llm_model(self) -> Any:
        """Initializes the LLM model. This method must be implemented by subclasses to set up the specific model.

        Returns:
            Any: The initialized model instance.
        """
        raise NotImplementedError("Must be implemented by the subclass.")

    def cancel_execution(self) -> None:
        """Safely cancels the current execution and releases GPU memory.

        Deletes the current instance of the loaded language model and resets the corresponding
        attribute to `None`."""
        del self.llm
        self.llm = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @abstractmethod
    def get_response(self, input_message: str | list | dict) -> str | None:
        """Generates a response from the model based on the provided text input.

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
        """Resets the internal state of the language model.

        This method calls `reset()` on the model to clear its internal state and
        `reset_llm_context()` to reset any additional context management mechanisms.

        Subclasses may override this method to implement model-specific reset behaviors
        if needed.
        """
        self.llm.reset()

    def infer(self, text: str | list) -> str | None:
        """Gets a response from the model, handling any errors or issues by resetting the model state if necessary.

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

    @staticmethod
    def generate_dict_msg(role: str, msg_content: str | list | None) -> dict:
        """For the provided content, generate a dictionary to be appended as the context for the response.

        Args:
            role (str): Role of the message, Can be system, user or assistant
            msg_content (str | list | None): Content of the message to be passed to the llm.

        Returns:
            The dictionary with the key pair values for role and content.
        """
        return {LLMChatKeys.role: role, LLMChatKeys.content: msg_content}

    def get_extra_context(self, packet: TextPacket) -> str | None:
        """Retrieves RAG context data from packet metadata using the configured rag_context_key.

        Searches the packet's generic_data dictionary for the specified key. If found and contains
        data, joins multiple context items into a single newline-delimited string. Returns None
        if no key is configured, the key is missing, or the context data is empty.

        Args:
            packet (TextPacket): The incoming TextPacket containing potential context data in its generic_data.

        Returns:
            str | None: Newline-joined context strings when RAG data exists, or None when
                rag_context_key is unset, missing, or points to empty data
        """
        if self.attributes.rag_context_key is None:
            return None

        context_data = packet.generic_data.get(self.attributes.rag_context_key)
        if not context_data:
            return None

        return "\n".join(context_data)

    def prepare_conversation_context(self, packet: TextPacket) -> tuple[str, str | None, str]:
        """Constructs complete conversation context including identifiers and augmented prompt.

        Extracts user_id and session_id from the packet, generating UUIDs if missing. Retrieves
        and injects RAG context when configured. Falls back to the template's default prompt
        when no packet content exists. The returned prompt combines context and query when
        RAG data is available.

        Args:
            packet(TextPacket): the incoming packet
        Returns:
            tuple[str, str, str]: The `user_id`, session_id` and prompt to use.
        """
        extra_context = self.get_extra_context(packet)
        prompt = packet.content
        if extra_context:
            prompt = f"Context:\n{extra_context}\n\nQuery:\n{packet.content}"
        return packet.id, packet.source, prompt

    def generate_response(self, container: DataContainer) -> DataContainer:
        """Processes a list of `TextPacket` objects, generating a response for each text packet.

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
        for packet in container.texts:
            full_context = []
            user_id, session_id, prompt = self.prepare_conversation_context(packet)
            if self.system_prompt:
                system_prompt_msg = self.generate_dict_msg(LLMChatKeys.system_value, self.system_prompt)
                full_context.append(system_prompt_msg)

            if self.attributes.chat_history_key:
                full_context.extend(packet.generic_data.get(self.attributes.chat_history_key, []))

            message = self.generate_dict_msg(LLMChatKeys.user_value, prompt)
            full_context.append(message)
            response = self.infer(full_context)
            self.logger.debug("End of interaction.")

            responses.append(TextPacket(source=session_id, content=response, id=user_id))

        container.texts.extend(responses)
        return container

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the LLMChatTemplate by processing the input `DataContainer` and generating a response.

        This method is responsible for handling the conversation flow, processing the input,
        and returning a response. It also ensures that the model has a prompt or previous conversation
        to work with.

        Args:
            container (DataContainer): Input data container containing texts.

        Returns:
            DataContainer: The output data container with the model's response added to the `texts` attribute.
        """
        if not container.texts:
            if self.attributes.prompt:
                container.texts.append(TextPacket(content=self.attributes.prompt))
            else:
                self.logger.debug("No need to process response.")
                return container

        return self.generate_response(container)

    def reset_state(self, template_name: str | None = None) -> None:
        self.cancel_execution()
        super().reset_state(template_name)
