# -*- coding: utf-8 -*-
from typing import cast

from llama_cpp import LLAMA_DEFAULT_SEED, Llama
from llama_cpp.llama_types import CreateChatCompletionResponse
from sinapsis_chatbots_base.helpers.llm_keys import LLMChatKeys
from sinapsis_chatbots_base.helpers.postprocess_text import postprocess_text
from sinapsis_chatbots_base.templates.llm_text_completion_base import (
    LLMCompletionArgs,
    LLMInitArgs,
    LLMTextCompletionAttributes,
    LLMTextCompletionBase,
)

from sinapsis_llama_cpp.helpers.llama_init_model import init_llama_model
from sinapsis_llama_cpp.helpers.llama_keys import (
    LLaMAModelKeys,
)


class LLaMAInitArgs(LLMInitArgs):
    """LLaMA-specific arguments for initializing the `llama_cpp.Llama` model.

    Attributes:
        llm_model_name (str): The name or path of the LLM model to use
                            (e.g. 'TheBloke/Llama-2-7B-GGUF').
        llm_model_file (str): The specific GGUF model file (e.g., 'llama-2-7b.Q2_K.gguf').
        n_gpu_layers (int): Number of layers to offload to the GPU (-1 for all). Defaults to `0`.
        use_mmap (bool): Use 'memory-mapping' to load the model. Defaults to `True`.
        use_mlock (bool): Force the model to be kept in RAM. Defaults to `False`.
        seed (int): RNG seed for model initialization. Defaults to `LLAMA_DEFAULT_SEED`.
        n_ctx (int): The context window size. Defaults to `512`.
        n_batch (int): The batch size for prompt processing. Defaults to `512`.
        n_ubatch (int): The batch size for token generation. Defaults to `512`.
        n_threads (int | None): CPU threads for generation. Defaults to `None`.
        n_threads_batch (int | None): CPU threads for batch processing. Defaults to `None`.
        flash_attn (bool): Enable Flash Attention if supported by the GPU. Defaults to `False`.
        chat_format (str | None): Chat template format (e.g., 'chatml'). Defaults to `None`.
        verbose (bool): Enable verbose logging from llama.cpp. Defaults to `True`.
    """

    llm_model_file: str
    n_gpu_layers: int = 0
    use_mmap: bool = True
    use_mlock: bool = False
    seed: int = LLAMA_DEFAULT_SEED
    n_ctx: int = 512
    n_batch: int = 512
    n_ubatch: int = 512
    n_threads: int | None = None
    n_threads_batch: int | None = None
    flash_attn: bool = False
    chat_format: str | None = None
    verbose: bool = True


class LLaMACompletionArgs(LLMCompletionArgs):
    """LLaMA-specific arguments for `create_chat_completion`, inheriting base parameters.

    Attributes:
        temperature (float): Controls randomness. 0.0 = deterministic, >0.0 = random. Defaults to `0.2`.
        top_p (float): Nucleus sampling. Considers tokens with cumulative probability >= top_p. Defaults to `0.95`.
        top_k (int): Top-k sampling. Considers the top 'k' most probable tokens. Defaults to `40`.
        max_tokens (int): The maximum number of new tokens to generate.
        min_p (float): Min-p sampling, filters tokens below this probability. Defaults to `0.05`.
        stop (str | list[str] | None): Stop sequences to halt generation. Defaults to `None`.
        seed (int | None): Overrides the model's seed just for this call. Defaults to `None`.
        repeat_penalty (float): Penalty for repeating tokens (1.0 = no penalty). Defaults to `1.0`.
        presence_penalty (float): Penalty for new tokens (0.0 = no penalty). Defaults to `0.0`.
        frequency_penalty (float): Penalty for frequent tokens (0.0 = no penalty). Defaults to `0.0`.
        logit_bias (dict[int, float] | None): Applies a bias to specific tokens. Defaults to `None`.
    """

    max_tokens: int
    min_p: float = 0.05
    stop: str | list[str] | None = None
    seed: int | None = None
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] | None = None


class LLaMATextCompletionAttributes(LLMTextCompletionAttributes):
    """Attributes for LLaMA-CPP text completion template.

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
    """

    init_args: LLaMAInitArgs
    completion_args: LLaMACompletionArgs


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
    """

    AttributesBaseModel = LLaMATextCompletionAttributes

    def init_llm_model(self) -> Llama:
        """Initializes the LLaMA model using the downloaded model path and the configuration attributes.

        This method downloads the model from the Hugging Face Hub using the
        model name and file attributes, then configures the model with
        parameters such as context size, temperature, and other relevant
        settings. The initialized Llama model is returned.

        Returns:
            Llama: An initialized instance of the Llama model.
        """
        return init_llama_model(
            self.attributes.init_args.llm_model_name,
            self.attributes.init_args.llm_model_file,
            self.attributes.init_args.model_dump(exclude_none=True, exclude={"llm_model_file", "llm_model_name"}),
            model_type=LLaMAModelKeys.model_type,
        )

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
        completion_args = self.attributes.completion_args.model_dump(exclude_none=True)
        chat_completion = None
        try:
            chat_completion = self.llm.create_chat_completion(messages=input_message, **completion_args)
        except (IndexError, AttributeError):
            self.reset_llm_state()
            if self.llm:
                chat_completion = self.llm.create_chat_completion(messages=input_message, **completion_args)

        if chat_completion:
            chat_completion = cast(CreateChatCompletionResponse, chat_completion)
            self.logger.info(chat_completion)
            llm_response_choice = chat_completion[LLMChatKeys.choices]
            response = llm_response_choice[0][LLMChatKeys.message][LLMChatKeys.content]
            self.logger.debug(response)

            if response:
                return postprocess_text(str(response), self.attributes.pattern, self.attributes.keep_before)
            return None
        return None
