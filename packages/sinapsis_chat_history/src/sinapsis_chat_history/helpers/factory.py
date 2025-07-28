# -*- coding: utf-8 -*-
from typing import ClassVar

from sinapsis_chat_history.helpers.base_provider import BaseStorageProvider, DatabaseConfig
from sinapsis_chat_history.helpers.postgres_provider import PostgresStorageProvider


class StorageProviderFactory:
    """Factory class for creating instances of storage providers.

    This factory dispatches the creation of storage provider instances based on
    a string identifier and configuration dictionary. It supports multiple
    providers registered in the internal `_registry`.
    """

    _registry: ClassVar[dict[str, type[BaseStorageProvider]]] = {
        "postgres": PostgresStorageProvider,
    }

    @classmethod
    def create(cls, provider: str, config: DatabaseConfig) -> BaseStorageProvider:
        """Create an instance of a storage provider based on the provider type.

        Args:
            provider (str): The identifier string of the storage provider to instantiate.
                For example, "postgres".
            config (dict[str, Any]): Configuration parameters required to initialize
                the storage provider instance.

        Raises:
            NotImplementedError: If the requested provider is not registered or supported.

        Returns:
            BaseStorageProvider: An instance of the requested storage provider,
                initialized with the given configuration.
        """
        if provider not in cls._registry:
            raise NotImplementedError(f"Unsupported provider: {provider}")
        return cls._registry[provider](config)
