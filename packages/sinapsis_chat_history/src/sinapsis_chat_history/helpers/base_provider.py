# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database connection attributes.

    Attributes:
        - user (str): The username for connecting to the PostgreSQL database.
        - password (str): The password for the database user.
        - port (int): The port to connect to the PostgreSQL database.
        - host (str): The host where the PostgreSQL database is running.
        - db_name (str): The name of the database to connect to or create.
    """

    user: str
    password: str
    port: int = 5432
    host: str = "localhost"
    db_name: str = "sinapsis_db"
    table: str = "sinapsis_table"
    columns: dict[str, str] = Field(default_factory=dict)


class BaseStorageProvider(ABC):
    """Abstract base class for chat message history storage providers.

    Defines the interface for storing and retrieving chatbot conversation messages
    across different storage backends. All implementations must provide the abstract methods.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config
        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the chat message storage structure.

        Creates required database structures (tables/collections) specifically
        designed for storing conversation messages.
        """

    @abstractmethod
    def insert(self, records: list[dict[str, Any]]) -> None:
        """Store conversation messages in the database.

        Args:
            records: List of message dictionaries to store, each representing
                    a single chat message in a conversation.
        """

    @abstractmethod
    def query(self, columns_to_retrieve: str | list[str], condition: str) -> list[Any]:
        """Retrieve chat messages from the Provider database with optional filters.

        Args:
            columns_to_retrieve (str | list[str]): Columns to retrieve from the table

            condition (str): Conditions the query meets to return results

        Returns:
            list[Any]: A list of messages formatted according to the specified return_type.
        """

    @abstractmethod
    def remove(self, condition: str) -> None:
        """Delete chat messages matching the criteria."""

    @abstractmethod
    def remove_last_n(self, last_n: int, condition: dict[str, str]) -> None:
        """Delete the last N chat messages matching the provided filters."""

    @abstractmethod
    def reset(self) -> None:
        """Completely reset all chat message storage.

        Wipes and recreates all structures used for storing conversation history.
        """
