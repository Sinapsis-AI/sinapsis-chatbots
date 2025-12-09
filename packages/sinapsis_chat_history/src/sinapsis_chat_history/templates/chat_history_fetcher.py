# -*- coding: utf-8 -*-
from typing import Any, Literal

from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket

from sinapsis_chat_history.templates.chat_history_base import ChatHistoryBase, ChatHistoryBaseAttributes

RETURN_TYPE = Literal["openai", "plain", "rich"]


class ChatHistoryFetcher(ChatHistoryBase):
    """Template for retrieving chat histories from a storage backend.

    Inherits from ChatHistoryBase and allows fetching messages in different formats,
    either based on the `TextPacket`'s dynamic context or overridden via static search parameters.
    """

    class AttributesBaseModel(ChatHistoryBaseAttributes):
        """Attributes for configuring how chat history is fetched.

        return_type (Literal["openai", "plain", "rich"]): Output format of the retrieved messages.
        search_kwargs (dict[str, Any] | None): Optional fixed filters (e.g., user_id, session_id, limit).
                                                If provided, these override corresponding values from the text packet.
        """

        return_type: RETURN_TYPE = "openai"
        search_kwargs: dict[str, Any] | None = None
        # columns_to_retrieve : str | list[str] = ["user_id", "session_id", "content", "timestamp"]

    def _fetch_histories(self, text_packet: TextPacket) -> None:
        """Fetch chat messages for a single `TextPacket`, optionally applying static search overrides.

        Args:
            text_packet (TextPacket): The input packet containing user and conversation identifiers.
        """
        query_kwargs = {"user_id": text_packet.id}

        if text_packet.source:
            query_kwargs["session_id"] = text_packet.source

        if self.attributes.search_kwargs:
            self.logger.debug(f"Applying search_kwargs override: {self.attributes.search_kwargs}")
            query_kwargs.update(self.attributes.search_kwargs)

        self.logger.debug(f"Final query_kwargs: {query_kwargs}")
        query_cond = ""

        order = ""
        limit = ""
        offset = ""

        for k, v in query_kwargs.items():
            k_lower = k.lower()

            if k_lower == "order_by":
                order += f" ORDER BY {v}"
            elif k_lower == "limit":
                limit += f" LIMIT {v}"
            elif k_lower == "offset":
                offset += f" OFFSET {v}"
            else:
                query_cond += f" \"{k}\" = '{v}' AND"
        query_cond = query_cond.rstrip("AND").strip()
        modifiers = order + limit + offset
        query_cond += modifiers

        columns = "role, content"
        query_results = self.db.query(columns_to_retrieve=columns, condition=query_cond)
        messages = self.process_query(columns, query_results, self.attributes.return_type)
        text_packet.generic_data[self.instance_name] = messages

    @staticmethod
    def process_query(columns: str, rows: list, return_type: RETURN_TYPE) -> list[dict]:
        """Formats the query to add the container.

        Args:
            columns (str): columns to be added as keys in the dictionary
            rows (list): list of results from the query
            return_type (RETURN_TYPE): format of choice for the conversation
        """
        if return_type == "openai":
            return [{"role": role, "content": content} for role, content in rows]
        elif return_type == "plain":
            return [content for _, content in rows]
        else:
            return [dict(zip(columns, rows))]

    def execute(self, container: DataContainer) -> DataContainer:
        """Execute the chat history retrieval on all text packets in the input container.

        Args:
            container (DataContainer): A container holding multiple text packets.

        Returns:
            DataContainer: The modified container, with fetched histories stored in each packet's `generic_data`.
        """
        for text_packet in container.texts:
            self._fetch_histories(text_packet)
        return container
