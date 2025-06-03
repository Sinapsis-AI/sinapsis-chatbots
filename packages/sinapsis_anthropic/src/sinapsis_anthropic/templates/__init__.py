# -*- coding: utf-8 -*-
import importlib
from typing import Callable

_root_lib_path = "sinapsis_anthropic.templates"

_template_lookup = {
    "AnthropicMultiModal": f"{_root_lib_path}.anthropic_multimodal",
    "AnthropicTextGeneration": f"{_root_lib_path}.anthropic_text_generation",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
