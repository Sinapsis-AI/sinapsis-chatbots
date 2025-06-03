# -*- coding: utf-8 -*-
import os.path
from typing import Literal

from huggingface_hub import errors, hf_hub_download
from llama_cpp import Llama
from llama_index.llms.llama_cpp import LlamaCPP

from sinapsis_llama_cpp.helpers.llama_keys import LLaMAModelKeys

LLM_MODEL_TYPE = Llama | LlamaCPP


def init_llama_model(attributes: dict, model_type: Literal["Llama", "LlamaCPP"] = "LlamaCPP") -> Llama | LlamaCPP:
    """
    Initializes the LLaMA model based on the specified model type ('Llama' or 'LlamaCPP').

    Downloads the model from the Hugging Face Hub using the provided model name and file attributes,
    and starts the model with relevant settings such as the number of tokens, temperature, GPU/CPU
    settings, and context size. The method returns an instance of either `Llama` or `LlamaCPP`
    based on the model_type argument.

    Args:
        attributes (dict): A dictionary containing the configuration attributes for the model,
                           such as `llm_model_name`, `llm_model_file`, `temperature`, `n_gpu_layers`,
                           `n_threads`, `max_tokens`, `n_ctx`, and `chat_format`.
        model_type (Literal["Llama", "LlamaCPP"], optional): The type of model to initialize.
                          Defaults to "Llama". Use "LlamaCPP" for the more efficient LlamaCPP model.

    Returns:
        Llama | LlamaCPP: The initialized model (either `Llama` or `LlamaCPP`).
    """
    model_class = Llama if model_type == LLaMAModelKeys.model_type else LlamaCPP
    try:
        model_path = hf_hub_download(
            attributes[LLaMAModelKeys.llm_model_name], filename=attributes[LLaMAModelKeys.llm_model_file]
        )
    except errors.HFValidationError:
        model_path = os.path.join(attributes[LLaMAModelKeys.llm_model_name], attributes[LLaMAModelKeys.llm_model_file])

    model_args = {
        LLaMAModelKeys.model_path: model_path,
        LLaMAModelKeys.temperature: attributes[LLaMAModelKeys.temperature],
        LLaMAModelKeys.verbose: True,
    }
    model_kwargs = {
        LLaMAModelKeys.n_gpu_layers: attributes[LLaMAModelKeys.n_gpu_layers],
        LLaMAModelKeys.n_threads: attributes[LLaMAModelKeys.n_threads],
    }
    if model_type == LLaMAModelKeys.model_type:
        model_args[LLaMAModelKeys.max_tokens] = attributes[LLaMAModelKeys.max_tokens]
        model_args[LLaMAModelKeys.n_ctx] = attributes[LLaMAModelKeys.n_ctx]
        model_args[LLaMAModelKeys.chat_format] = attributes[LLaMAModelKeys.chat_format]
        model_args.update(model_kwargs)
    else:
        model_args[LLaMAModelKeys.max_new_tokens] = attributes[LLaMAModelKeys.max_tokens]
        model_args[LLaMAModelKeys.context_window] = attributes[LLaMAModelKeys.n_ctx]
        model_args[LLaMAModelKeys.model_kwargs] = model_kwargs

    return model_class(**model_args)
