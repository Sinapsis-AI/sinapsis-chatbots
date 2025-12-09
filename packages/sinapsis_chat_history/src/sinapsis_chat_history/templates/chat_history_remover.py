# -*- coding: utf-8 -*-
from typing import Any

from sinapsis_core.data_containers.data_packet import DataContainer

from sinapsis_chat_history.templates.chat_history_base import ChatHistoryBase, ChatHistoryBaseAttributes


class ChatHistoryRemover(ChatHistoryBase):
    """Template for deleting chat history records based on filters.

    This component deletes messages from the storage provider using
    a dictionary of filters provided in the `attributes.filters`.
    """

    class AttributesBaseModel(ChatHistoryBaseAttributes):
        """Attributes model for the remover template.

        Attributes:
            last_n (int): the number of entries to delete
            filters (dict[str, Any]): Dictionary of filters to apply for deletion. Must include
                the columns that exist in the table, otherwise the deletion will fail
        """

        last_n: int
        filters: dict[str, Any]

    def execute(self, container: DataContainer) -> DataContainer:
        """Execute the deletion of chat messages based on the provided filters.

        Args:
            container (DataContainer): Incoming data container (not used for deletion logic).

        Returns:
            DataContainer: The same container passed in, unmodified.
        """
        self.db.remove_last_n(self.attributes.last_n, condition=self.attributes.filters)
        return container
