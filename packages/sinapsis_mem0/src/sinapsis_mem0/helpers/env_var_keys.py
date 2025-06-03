# -*- coding: utf-8 -*-
from typing import Any

from pydantic import BaseModel
from sinapsis_core.utils.env_var_keys import EnvVarEntry, doc_str, return_docs_for_vars


class _Mem0Keys(BaseModel):
    """Env vars for Mem0."""

    MEM0_API_KEY: EnvVarEntry = EnvVarEntry(
        var_name="MEM0_API_KEY",
        default_value=None,
        allowed_values=None,
        description="Set API key for Mem0",
    )


Mem0EnvVars = _Mem0Keys()

doc_str = return_docs_for_vars(Mem0EnvVars, docs=doc_str, string_for_doc="""Mem0 env vars available: \n""")
__doc__ = doc_str


def __getattr__(name: str) -> Any:
    """to use as an import, when updating the value is not important"""
    if name in Mem0EnvVars.model_fields:
        return Mem0EnvVars.model_fields[name].default.value

    raise AttributeError(f"Agent does not have `{name}` env var")


_all__ = (*list(Mem0EnvVars.model_fields.keys()), "Mem0EnvVars")
