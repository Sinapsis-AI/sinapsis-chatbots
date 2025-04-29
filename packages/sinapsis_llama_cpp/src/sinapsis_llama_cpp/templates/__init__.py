# -*- coding: utf-8 -*-
import importlib
from typing import Callable

_root_lib_path = "sinapsis_llama_cpp.templates"

_template_lookup = {
    "LLama4MultiModal": f"{_root_lib_path}.llama4_multimodal",
    "LLaMATextCompletionWithContext": f"{_root_lib_path}.llama_text_completion_with_context",
    "LLaMATextCompletion": f"{_root_lib_path}.llama_text_completion",
    "LLama4TextToText": f"{_root_lib_path}.llama_4_text_to_text",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
