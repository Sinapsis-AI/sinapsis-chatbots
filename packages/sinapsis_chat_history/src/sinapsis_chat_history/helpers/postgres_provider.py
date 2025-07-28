# -*- coding: utf-8 -*-
import json
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.errors import Error
from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_chat_history.helpers.base_provider import BaseStorageProvider, DatabaseConfig


class PostgresDatabaseConfig(DatabaseConfig):
    """Configuration of the postgres table."""

    columns: dict[str, str] = {  # noqa: RUF012.
        "id": "SERIAL PRIMARY KEY",
        "user_id": "TEXT NOT NULL",
        "role": "TEXT NOT NULL",
        "content": "TEXT NOT NULL",
        "timestamp": "TIMESTAMPTZ DEFAULT now()",
        "session_id": "TEXT",
        "metadata": "JSONB",
    }


class PostgresStorageProvider(BaseStorageProvider):
    """PostgreSQL implementation for storing and managing chatbot conversation histories.

    Provides persistent storage for chat messages using PostgreSQL, with support for:
    - Storing complete conversation threads
    - Retrieving message history by user and/or conversation
    - Managing message metadata
    """

    def _get_connection_string(self, database: str | None = None) -> str:
        """Generate PostgreSQL connection string.

        Args:
            database (str | None, optional): Specific database to connect. Defaults to None.

        Returns:
            str: PostgreSQL connection string
        """
        _ = database
        return (
            f"host={self.config.host} port={self.config.port} user={self.config.user} password={self.config.password}"
        )

    def _get_connection_to_db_str(self):
        return f"{self._get_connection_string()} dbname={self.config.db_name}"

    def initialize(self) -> None:
        """Initialize PostgreSQL storage for chat messages.

        Creates the database (if needed) and sets up the message table structure.
        """
        default_conn_str = self._get_connection_string()
        with psycopg.connect(default_conn_str, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.config.db_name,))
            if not cur.fetchone():
                cur.execute(f'CREATE DATABASE "{self.config.db_name}"')

        self._create_table()

    def _create_table(self) -> None:
        """Create the table structure for storing chat messages."""
        columns_sql = ",\n    ".join(f"{name} {col_type}" for name, col_type in self.config.columns.items())
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.config.table} (
                {columns_sql}
            );
            """

        with psycopg.connect(self._get_connection_to_db_str()) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
            conn.commit()

    @staticmethod
    def build_insert_values(values_to_insert: dict, insert_columns: list[str]) -> tuple:
        """Builds the tuple of values to be inserted into the table.

        Args:
            values_to_insert (dict): dictionary with the key:value pairs of the data to be inserted
            insert_columns (list[str]): list of columns to insert the values from the values_to_insert dictionary
        """
        values = []
        for col in insert_columns:
            if col == "timestamp":
                # Provide a default timestamp if missing
                values.append(values_to_insert.get(col) or datetime.now(timezone.utc))
            elif col == "metadata":
                values.append(json.dumps(values_to_insert.get(col)) if values_to_insert.get(col) is not None else None)
            else:
                values.append(values_to_insert.get(col))
        return tuple(values)

    def insert(self, records: list[dict[str, Any]]) -> None:
        """Store chat messages in PostgreSQL.

        Args:
            records(list[dict[str, Any]]): List of message dictionaries containing the key:value pairs to be inserted
            into the table. The keys must match the table schema
        """
        self.config.columns.pop("id", None)
        column_names = list(self.config.columns.keys())
        placeholders = ", ".join(["%s"] * len(column_names))
        column_list = ", ".join(column_names)
        insert_sql = f"""
            INSERT INTO {self.config.table} ({column_list})
            VALUES ({placeholders})
            """

        with psycopg.connect(self._get_connection_to_db_str()) as conn:
            with conn.cursor() as cur:
                for r in records:
                    cur.execute(
                        insert_sql,
                        self.build_insert_values(r, column_names),
                    )
            conn.commit()

    def query(self, columns_to_retrieve: str | list[str], condition: str) -> list[Any]:
        """Retrieve chat messages from the PostgreSQL database with optional filters.

        Args:
            columns_to_retrieve (str | list[str]): columns from the table to retrieve information from
            condition (str): Condition the query must meet for results to be retrieved

        Returns:
            list[Any]: A list of messages formatted according to the specified return_type.
        """
        query = f""" SELECT {columns_to_retrieve} FROM {self.config.table} WHERE {condition} """

        with psycopg.connect(self._get_connection_to_db_str()) as conn, conn.cursor() as cur:
            cur.execute(query)  # , params)
            rows = cur.fetchall()

        rows.reverse()
        return rows

    def remove(self, condition: str) -> None:
        """Delete chat messages from PostgreSQL."""
        query = f"DELETE FROM {self.config.table} WHERE TRUE AND {condition}"

        with psycopg.connect(self._get_connection_to_db_str()) as conn:
            with conn.cursor() as cur:
                cur.execute(query)  # , params)
            conn.commit()

    def remove_last_n(self, last_n: int, condition: dict[str, str]) -> None:
        """Delete the last N chat messages matching the provided filters."""
        query = f"""
               DELETE FROM {self.config.table}
               WHERE id IN (
                   SELECT id FROM {self.config.table}
                   WHERE
           """
        params = []

        for key, value in condition.items():
            query += f"{key} = %s AND "
            params.append(value)

        # Remove the last AND if there are conditions
        if len(condition) > 0:
            query = query[:-4]  # Remove the last " AND "

        query += " ORDER BY timestamp DESC LIMIT %s )"
        params.append(str(last_n))

        try:
            with psycopg.connect(self._get_connection_to_db_str()) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
        except Error as e:
            sinapsis_logger.error(f"Error deleting messages: {e}")

    def reset(self) -> None:
        """Reset chat message storage by dropping and recreating the table."""
        with psycopg.connect(self._get_connection_to_db_str()) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.config.table};")
            conn.commit()
        self._create_table()
