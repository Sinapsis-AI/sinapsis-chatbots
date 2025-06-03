# -*- coding: utf-8 -*-
import importlib
from typing import Callable

_root_lib_path = "sinapsis_mem0.templates"

_template_lookup = {
    "Mem0Add": f"{_root_lib_path}.mem0_add",
    "Mem0Delete": f"{_root_lib_path}.mem0_delete",
    "Mem0Get": f"{_root_lib_path}.mem0_get",
    "Mem0Reset": f"{_root_lib_path}.mem0_reset",
    "Mem0Search": f"{_root_lib_path}.mem0_search",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
