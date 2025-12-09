# -*- coding: utf-8 -*-
import os.path
from typing import Literal

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_index.llms.llama_cpp import LlamaCPP
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR

from sinapsis_llama_cpp.helpers.llama_keys import LLaMAModelKeys


def init_llama_model(
    llm_model_name: str, llm_model_file: str, init_args: dict, model_type: Literal["Llama", "LlamaCPP"] = "LlamaCPP"
) -> Llama | LlamaCPP:
    """Initializes the LLaMA model based on the specified model type.

    Downloads the model from the Hugging Face Hub if not found locally,
    then instantiates the model class with the provided init_args.

    Args:
        llm_model_name (str): The Hugging Face repo ID or local folder path.
        llm_model_file (str): The GGUF model file name.
        init_args (Dict[str, Any]): A dictionary of arguments to pass to the
            model constructor (e.g., n_ctx, n_gpu_layers).
        model_type (Literal["Llama", "LlamaCPP"], optional): The type of model to
            initialize. Defaults to "LlamaCPP".

    Returns:
        Llama | LlamaCPP: The initialized model.
    """
    model_class = Llama if model_type == LLaMAModelKeys.model_type else LlamaCPP
    model_path = os.path.join(llm_model_name, llm_model_file)

    if not os.path.exists(model_path):
        model_path = hf_hub_download(llm_model_name, filename=llm_model_file, cache_dir=SINAPSIS_CACHE_DIR)

    model_args = {LLaMAModelKeys.model_path: model_path, **init_args}

    return model_class(**model_args)
