# -*- coding: utf-8 -*-
from typing import Any

from pydantic import BaseModel
from sinapsis_core.utils.env_var_keys import EnvVarEntry, doc_str, return_docs_for_vars


class _AnthropicKeys(BaseModel):
    """
    Env vars for Anthropic
    """

    ANTHROPIC_API_KEY: EnvVarEntry = EnvVarEntry(
        var_name="ANTHROPIC_API_KEY",
        default_value=None,
        allowed_values=None,
        description="set api key for Anthropic",
    )


AnthropicEnvVars = _AnthropicKeys()

doc_str = return_docs_for_vars(AnthropicEnvVars, docs=doc_str, string_for_doc="""Anthropic env vars available: \n""")
__doc__ = doc_str


def __getattr__(name: str) -> Any:
    """to use as an import, when updating the value is not important"""
    if name in AnthropicEnvVars.model_fields:
        return AnthropicEnvVars.model_fields[name].default.value

    raise AttributeError(f"Agent does not have `{name}` env var")


_all__ = (*list(AnthropicEnvVars.model_fields.keys()), "AnthropicEnvVars")
