# -*- coding: utf-8 -*-
from typing import Literal

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_core.template_base.template import Template

from sinapsis_chat_history.helpers.database_env_var_keys import DatabaseEnvVars
from sinapsis_chat_history.helpers.factory import StorageProviderFactory
from sinapsis_chat_history.helpers.postgres_provider import PostgresDatabaseConfig


@dataclass
class ChatHistoryColumns:
    """Holds the default column names for the chat history database table."""

    user_id: str = "user_id"
    role: str = "role"
    session_id: str = "session_id"
    timestamp: str = "timestamp"
    content: str = "content"
    metadata: str = "metadata"


class ChatHistoryBaseAttributes(TemplateAttributes):
    """Attribute configuration for chat history templates.

    Attributes:
        provider (Literal["postgres"]): The storage backend to use (currently only "postgres" is supported).
        db_config (dict[str, Any]): Configuration dictionary for initializing the selected storage provider.
    """

    provider: Literal["postgres"] = "postgres"
    db_config: PostgresDatabaseConfig = Field(default_factory=PostgresDatabaseConfig)

    @model_validator(mode="after")
    def load_from_env_and_validate(self) -> "ChatHistoryBaseAttributes":
        """Ensures database configuration is valid, prioritizing environment variables.

        Returns:
            The validated and configured model instance.

        Raises:
            ValueError: If `user` or `password` is not configured either through
                        an attribute or an environment variable.
        """
        env_map = {
            "user": DatabaseEnvVars.DB_USER,
            "password": DatabaseEnvVars.DB_PASSWORD,
            "host": DatabaseEnvVars.DB_HOST,
            "port": DatabaseEnvVars.DB_PORT,
        }

        for attr_name, env_var in env_map.items():
            if env_var.value:
                value = int(env_var.value) if attr_name == "port" else env_var.value
                setattr(self.db_config, attr_name, value)

        if not self.db_config.user:
            raise ValueError(
                "Database user must be provided via 'DB_USER' environment variable or as a direct attribute."
            )
        if not self.db_config.password:
            raise ValueError(
                "Database password must be provided via 'DB_PASSWORD' environment variable or as a direct attribute."
            )

        return self


class ChatHistoryBase(Template):
    """Base class for all chat history-related templates.

    Handles shared initialization logic and provides a database connection instance (`self.db`)
    based on the provider and configuration supplied via attributes.
    """

    AttributesBaseModel = ChatHistoryBaseAttributes
    UIProperties = UIPropertiesMetadata(category="databases", output_type=OutputTypes.TEXT)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.db = StorageProviderFactory.create(provider=self.attributes.provider, config=self.attributes.db_config)
