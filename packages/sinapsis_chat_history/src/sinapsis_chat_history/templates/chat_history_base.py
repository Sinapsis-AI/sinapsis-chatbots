# -*- coding: utf-8 -*-
from typing import Any, Literal

from pydantic import Field
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
    db_config: dict[str, Any] = Field(default_factory=dict)


class ChatHistoryBase(Template):
    """Base class for all chat history-related templates.

    Handles shared initialization logic and provides a database connection instance (`self.db`)
    based on the provider and configuration supplied via attributes.
    """

    AttributesBaseModel = ChatHistoryBaseAttributes
    UIProperties = UIPropertiesMetadata(category="databases", output_type=OutputTypes.TEXT)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        merged_config = {
            key: value
            for key, value in {
                "user": DatabaseEnvVars.DB_USER.value,
                "password": DatabaseEnvVars.DB_PASSWORD.value,
                "host": DatabaseEnvVars.DB_HOST.value,
                "port": int(DatabaseEnvVars.DB_PORT.value),
            }.items()
            if value
        }
        merged_config.update(self.attributes.db_config)
        self.db = StorageProviderFactory.create(
            provider=self.attributes.provider, config=PostgresDatabaseConfig(**merged_config)
        )
