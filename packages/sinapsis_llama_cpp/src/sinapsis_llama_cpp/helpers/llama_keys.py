# -*- coding: utf-8 -*-
from typing import Any, Literal

from llama_cpp import Llama
from llama_index.llms.llama_cpp import LlamaCPP

# from llama_index.vector_stores.postgres import PGVectorStore
from pydantic.dataclasses import dataclass

LLM_MODEL_TYPE = Llama | LlamaCPP | Any


@dataclass
class LLMChatKeys:
    """
    A class to hold constants for the keys used in chat interactions with an LLM (Large Language Model).

    These keys represent the standard fields in a chat interaction, such as the role of the participant
    and the content of the message. They are typically used when constructing input messages or
    processing the output from an LLM.
    """

    role: Literal["role"] = "role"
    content: Literal["content"] = "content"
    choices: Literal["choices"] = "choices"
    message: Literal["message"] = "message"
    llm_responses: Literal["llm_responses"] = "llm_responses"
    system_value: Literal["system"] = "system"
    user_value: Literal["user"] = "user"
    assistant_value: Literal["assistant"] = "assistant"


@dataclass
class LLaMAModelKeys:
    """
    A class to hold constants for the keys used in LLama init method.
    """

    llm_model_name: Literal["llm_model_name"] = "llm_model_name"
    llm_model_file: Literal["llm_model_file"] = "llm_model_file"
    max_tokens: Literal["max_tokens"] = "max_tokens"
    max_new_tokens: Literal["max_new_tokens"] = "max_new_tokens"
    temperature: Literal["temperature"] = "temperature"
    n_threads: Literal["n_threads"] = "n_threads"
    n_gpu_layers: Literal["n_gpu_layers"] = "n_gpu_layers"
    n_ctx: Literal["n_ctx"] = "n_ctx"
    context_window: Literal["context_window"] = "context_window"
    chat_format: Literal["chat_format"] = "chat_format"
    verbose: Literal["verbose"] = "verbose"
    model_path: Literal["model_path"] = "model_path"
    model_type: Literal["Llama"] = "Llama"
    model_kwargs: Literal["model_kwargs"] = "model_kwargs"
