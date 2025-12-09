# -*- coding: utf-8 -*-
from typing import Any

from pydantic import BaseModel
from sinapsis_core.utils.env_var_keys import EnvVarEntry, doc_str, return_docs_for_vars


class _RAGTextCompletionEnvVars(BaseModel):
    """Env vars for RAG Chat Completion webapp."""

    FEED_DB_DEFAULT_PATH: EnvVarEntry = EnvVarEntry(
        var_name="FEED_DB_DEFAULT_PATH",
        default_value=None,
        allowed_values=None,
        description="set config path for feeding the database with default documents",
    )

    FEED_DB_FROM_PDF_PATH: EnvVarEntry = EnvVarEntry(
        var_name="FEED_DB_FROM_PDF_PATH",
        default_value=None,
        allowed_values=None,
        description="set config path for feeding the database with pdf document",
    )

    PGVECTOR_USER: EnvVarEntry = EnvVarEntry(
        var_name="PGVECTOR_USER",
        default_value=None,
        allowed_values=None,
        description="Username for connecting to the PGVector database.",
    )

    PGVECTOR_PASSWORD: EnvVarEntry = EnvVarEntry(
        var_name="PGVECTOR_PASSWORD",
        default_value=None,
        allowed_values=None,
        description="Password for connecting to the PGVector database.",
    )
    PGVECTOR_HOST: EnvVarEntry = EnvVarEntry(
        var_name="PGVECTOR_HOST",
        default_value=None,
        allowed_values=None,
        description="Hostname or IP address for the PGVector database.",
    )

    PGVECTOR_PORT: EnvVarEntry = EnvVarEntry(
        var_name="PGVECTOR_PORT",
        default_value=None,
        allowed_values=None,
        description="Port number for the PGVector database.",
    )


RAGTextCompletionEnvVars = _RAGTextCompletionEnvVars()

doc_str = return_docs_for_vars(
    RAGTextCompletionEnvVars, docs=doc_str, string_for_doc="""RAG Text Completion env vars available: \n"""
)
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
    if name in RAGTextCompletionEnvVars.model_fields:
        return RAGTextCompletionEnvVars.model_fields[name].default.value

    raise AttributeError(f"Agent does not have `{name}` env var")


_all__ = (*list(RAGTextCompletionEnvVars.model_fields.keys()), "RAGTextCompletionEnvVars")
