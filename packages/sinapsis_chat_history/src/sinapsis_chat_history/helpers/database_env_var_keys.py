# -*- coding: utf-8 -*-
from typing import Any

from pydantic import BaseModel
from sinapsis_core.utils.env_var_keys import EnvVarEntry, doc_str, return_docs_for_vars


class _DatabaseEnvVars(BaseModel):
    """
    Env vars for OpenAI
    """

    DB_USER: EnvVarEntry = EnvVarEntry(
        var_name="DB_USER",
        default_value=None,
        allowed_values=None,
        description="Input user for the DB connection",
    )
    DB_PASSWORD: EnvVarEntry = EnvVarEntry(
        var_name="DB_PASSWORD",
        default_value=None,
        allowed_values=None,
        description="Input password for the DB connection",
    )
    DB_HOST: EnvVarEntry = EnvVarEntry(
        var_name="DB_HOST",
        default_value=None,
        allowed_values=None,
        description="Input host for the DB connection",
    )

    DB_PORT: EnvVarEntry = EnvVarEntry(
        var_name="DB_PORT",
        default_value=None,
        allowed_values=None,
        description="Input host for the DB connection",
    )


DatabaseEnvVars = _DatabaseEnvVars()

doc_str = return_docs_for_vars(DatabaseEnvVars, docs=doc_str, string_for_doc="""Database env vars available: \n""")
__doc__ = doc_str


def __getattr__(name: str) -> Any:
    """to use as an import, when updating the value is not important"""
    if name in DatabaseEnvVars.model_fields:
        return DatabaseEnvVars.model_fields[name].default.value

    raise AttributeError(f"Agent does not have `{name}` env var")


_all__ = (*list(DatabaseEnvVars.model_fields.keys()), "DatabaseEnvVars")
