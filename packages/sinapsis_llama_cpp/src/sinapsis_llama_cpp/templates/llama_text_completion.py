# -*- coding: utf-8 -*-

from typing import cast

from llama_cpp import Llama
from llama_cpp.llama_types import CreateChatCompletionResponse
from sinapsis_chatbots_base.helpers.llm_keys import LLMChatKeys
from sinapsis_chatbots_base.helpers.postprocess_text import postprocess_text
from sinapsis_chatbots_base.templates.llm_text_completion_base import (
    LLMTextCompletionAttributes,
    LLMTextCompletionBase,
)

from sinapsis_llama_cpp.helpers.llama_init_model import init_llama_model
from sinapsis_llama_cpp.helpers.llama_keys import (
    LLaMAModelKeys,
)


class LLaMATextCompletionAttributes(LLMTextCompletionAttributes):
    """
    Attributes for LLaMA-CPP text completion template


    llm_model_name (str): The name of the LLM model to use.
    n_ctx (int): Maximum context size for the model.
    role (Literal["system", "user", "assistant"]): The role in the conversation, such as
        "system", "user", or "assistant". Defaults to "assistant".
    prompt (str): A set of instructions provided to the LLM to guide how to respond.
        The default value is an empty string.
    system_prompt (str | None): The prompt that indicates the LLM how to behave
        (e.g. you are an expert on...)
    chat_format (str | None): The format for the chat messages
        (e.g., llama-2, chatml, etc.).
    context_max_len (int): The maximum length for the conversation context.
        The default value is 6.
    pattern (str | None): A regex pattern to match delimiters. The default value is
        `<|...|>` and `</...>`.
    keep_before (bool): If True, returns the portion before the first match; if False,
        returns the portion after the first match.
    llm_model_file (str): File path to the large language model.
    max_tokens (int): Maximum number of tokens to generate.
    temperature (float): Sampling temperature for the model.
    n_threads (int): Number of CPU threads to use for processing.
    n_gpu_layers (int): Number of layers to offload to GPU. If -1,
        all layers are offloaded.
    """

    llm_model_file: str
    max_tokens: int = 256
    temperature: float = 0.5
    n_threads: int = 4
    n_gpu_layers: int = 0


class LLaMATextCompletion(LLMTextCompletionBase):
    """Template for configuring and initializing a LLaMA-based text completion model.

    This template is responsible for setting up and initializing a LLaMA-CPP model based
    on the provided configuration. It handles the model setup by downloading
    the model from the Hugging Face Hub and configuring the necessary parameters.
    The template takes a text input from the DataContainer, and generates a response
    using the llm model.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
    - template_name: LLaMATextCompletion
      class_name: LLaMATextCompletion
      template_input: InputTemplate
      attributes:
        llm_model_name: 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'
        llm_model_file: 'mistral-7b-instruct-v0.2.Q2_K.gguf'
        max_tokens: 256
        temperature: 0.5
        n_threads: 4
        n_gpu_layers: 0
        n_ctx: '3000'
        role: assistant
        chat_format: chatml
        context_max_len: 6
        keep_before: true

    """

    AttributesBaseModel = LLaMATextCompletionAttributes

    def init_llm_model(self) -> Llama:
        """
        Initializes the LLaMA model using the downloaded model path and the
        configuration attributes.

        This method downloads the model from the Hugging Face Hub using the
        model name and file attributes, then configures the model with
        parameters such as context size, temperature, and other relevant
        settings. The initialized Llama model is returned.

        Returns:
            Llama: An initialized instance of the Llama model.
        """
        return init_llama_model(
            self.attributes.model_dump(exclude_none=True),
            model_type=LLaMAModelKeys.model_type,
        )

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

        try:
            chat_completion = self.llm.create_chat_completion(messages=input_message)
        except IndexError:
            self.reset_llm_state()
            chat_completion = self.llm.create_chat_completion(messages=input_message)

        chat_completion = cast(CreateChatCompletionResponse, chat_completion)
        llm_response_choice = chat_completion[LLMChatKeys.choices]
        response = llm_response_choice[0][LLMChatKeys.message][LLMChatKeys.content]
        self.logger.debug(response)

        if response:
            return postprocess_text(str(response), self.attributes.pattern, self.attributes.keep_before)
        return None
