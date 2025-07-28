# -*- coding: utf-8 -*-
from sinapsis_core.data_containers.data_packet import DataContainer

from sinapsis_chat_history.templates.chat_history_base import ChatHistoryBase


class ChatHistoryReset(ChatHistoryBase):
    """Performs complete reset of chat history by dropping and recreating the table.

    Warning: This is a destructive operation that will permanently erase all chat history.
    The table structure will be recreated according to the current schema definition.
    """

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the table reset operation.

        Args:
            container (DataContainer): Incoming data container (not used for reset logic).

        Returns:
            DataContainer: The same container passed in, unmodified.
        """
        self.db.reset()
        return container
