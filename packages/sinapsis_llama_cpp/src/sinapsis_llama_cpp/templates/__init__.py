# -*- coding: utf-8 -*-
import importlib
from typing import Callable

_root_lib_path = "sinapsis_llama_cpp.templates"

_template_lookup = {
    "LLaMATextCompletion": f"{_root_lib_path}.llama_text_completion",
    "QueryContextualizeFromText": f"{_root_lib_path}.query_contextualize_from_text",
    "QueryContextualizeFromFile": f"{_root_lib_path}.query_contextualize_from_file",
    # "LLaMARAGTextCompletion": f"{_root_lib_path}.llama_rag_text_completion",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
