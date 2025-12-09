# -*- coding: utf-8 -*-
from typing import Any

from pydantic import BaseModel
from sinapsis_core.utils.env_var_keys import EnvVarEntry, doc_str, return_docs_for_vars


class _HuggingfaceEnvVars(BaseModel):
    """Env vars for HuggingFace."""

    HF_TOKEN: EnvVarEntry = EnvVarEntry(
        var_name="HF_TOKEN",
        default_value=" ",
        allowed_values=None,
        description="set api key for HuggingFace API",
    )


HuggingFaceEnvVars = _HuggingfaceEnvVars()

doc_str = return_docs_for_vars(HuggingFaceEnvVars, docs=doc_str, string_for_doc="""HF env vars available: \n""")
__doc__ = doc_str


def __getattr__(name: str) -> Any:
    """Allows accessing environment variable default values directly as module attributes.

    Args:
        name (str): The name of the environment variable.

    Raises:
        AttributeError: If the requested attribute `name` is not a defined
                        environment variable.

    Returns:
        Any: The default value of the requested environment variable.
    """
    if name in HuggingFaceEnvVars.model_fields:
        return HuggingFaceEnvVars.model_fields[name].default.value

    raise AttributeError(f"Agent does not have `{name}` env var")


_all__ = (*list(HuggingFaceEnvVars.model_fields.keys()), "HuggingFaceEnvVars")
