# -*- coding: utf-8 -*-
from typing import Any

from pydantic import BaseModel
from sinapsis_core.utils.env_var_keys import EnvVarEntry, doc_str, return_docs_for_vars


class _HuggingfaceEnvVars(BaseModel):
    """
    Env vars for HuggingFace
    """

    HF_TOKEN: EnvVarEntry = EnvVarEntry(
        var_name="OPENAI_API_KHF_TOKEN",
        default_value=" ",
        allowed_values=None,
        description="set api key for HuggingFace API",
    )


HuggingFaceEnvVars = _HuggingfaceEnvVars()

doc_str = return_docs_for_vars(HuggingFaceEnvVars, docs=doc_str, string_for_doc="""HF env vars available: \n""")
__doc__ = doc_str


def __getattr__(name: str) -> Any:
    """to use as an import, when updating the value is not important"""
    if name in HuggingFaceEnvVars.model_fields:
        return HuggingFaceEnvVars.model_fields[name].default.value

    raise AttributeError(f"Agent does not have `{name}` env var")


_all__ = (*list(HuggingFaceEnvVars.model_fields.keys()), "HuggingFaceEnvVars")
