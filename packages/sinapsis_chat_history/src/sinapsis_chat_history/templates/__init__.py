# -*- coding: utf-8 -*-
import importlib
from typing import Callable

_root_lib_path = "sinapsis_chat_history.templates"

_template_lookup = {
    "ChatHistoryFetcher": f"{_root_lib_path}.chat_history_fetcher",
    "ChatHistoryRemover": f"{_root_lib_path}.chat_history_remover",
    "ChatHistoryReset": f"{_root_lib_path}.chat_history_reset",
    "ChatHistorySaver": f"{_root_lib_path}.chat_history_saver",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
