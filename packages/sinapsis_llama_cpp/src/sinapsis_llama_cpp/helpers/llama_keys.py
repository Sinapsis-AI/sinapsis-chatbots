# -*- coding: utf-8 -*-
from typing import Literal

from pydantic.dataclasses import dataclass


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
