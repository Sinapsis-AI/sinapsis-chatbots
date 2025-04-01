# -*- coding: utf-8 -*-
import importlib
from typing import Callable

_root_lib_path = "sinapsis_llama_index.templates"

_template_lookup = {
    "EmbeddingNodeGenerator": f"{_root_lib_path}.embedding_node_generator",
    "CodeEmbeddingNodeGenerator": f"{_root_lib_path}.code_embedding_node_generator",
    "LLaMAIndexInsertNodes": f"{_root_lib_path}.llama_index_insert_nodes",
    "LLaMAIndexNodeRetriever": f"{_root_lib_path}.llama_index_node_retriever",
    "LLaMAIndexRAGTextCompletion": f"{_root_lib_path}.llama_index_rag_text_completion",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
