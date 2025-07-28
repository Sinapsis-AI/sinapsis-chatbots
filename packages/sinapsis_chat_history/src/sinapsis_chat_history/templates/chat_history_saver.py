# -*- coding: utf-8 -*-
from typing import Any

from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket

from sinapsis_chat_history.templates.chat_history_base import (
    ChatHistoryBase,
    ChatHistoryBaseAttributes,
    ChatHistoryColumns,
)


class ChatHistorySaver(ChatHistoryBase):
    """Template for saving chat messages into the database.

    It expects a list of `TextPacket` instances, ideally arranged as user/assistant pairs.
    If the number of packets is even, they are split into user and assistant roles.
    If the number is odd, all packets are assumed to be from the user.
    """

    class AttributesBaseModel(ChatHistoryBaseAttributes):
        """Attributes for configuring ChatHistorySaver.

        metadata_key (str): Key to extract custom metadata from the packet's generic_data.
        """

        metadata_key: str = "metadata"

    def extract_metadata(self, packet: TextPacket) -> dict[str, Any]:
        """Extract metadata from a text packet using the configured key.

        Args:
            packet (TextPacket): The text packet to extract metadata from.

        Returns:
            dict[str, Any]: Extracted metadata dictionary, or an empty one if missing or invalid.
        """
        metadata = packet.generic_data.get(self.attributes.metadata_key, {})
        if not isinstance(metadata, dict):
            self.logger.warning(f"Metadata under key '{self.attributes.metadata_key}' is not a dict. Ignoring.")
            return {}
        return metadata

    def _save_messages(self, text_packets: list[TextPacket]) -> None:
        """Save a batch of chat messages into the database.

        Args:
            text_packets (list[TextPacket]): A list of packets representing chat messages.
        """
        n = len(text_packets)
        if n % 2 != 0:
            self.logger.warning("Odd number of packets received; assigning all to user role.")
            messages = [
                {
                    ChatHistoryColumns.user_id: packet.id,
                    ChatHistoryColumns.role: "user",
                    ChatHistoryColumns.content: packet.content,
                    ChatHistoryColumns.session_id: packet.source,
                    ChatHistoryColumns.metadata: self.extract_metadata(packet),
                }
                for packet in text_packets
            ]
        else:
            half = n // 2
            user_packets = text_packets[:half]
            assistant_packets = text_packets[half:]
            messages = [
                {
                    ChatHistoryColumns.user_id: packet.id,
                    ChatHistoryColumns.role: role,
                    ChatHistoryColumns.content: packet.content,
                    ChatHistoryColumns.session_id: packet.source,
                    ChatHistoryColumns.metadata: self.extract_metadata(packet),
                }
                for role, packets in [("user", user_packets), ("assistant", assistant_packets)]
                for packet in packets
            ]

        self.db.insert(messages)

    def execute(self, container: DataContainer) -> DataContainer:
        """Execute the chat history saving on all text packets in the input container.

        Args:
            container (DataContainer): Container holding text packets to be stored.

        Returns:
            DataContainer: Unmodified input container, post-storage operation.
        """
        self._save_messages(container.texts)
        return container
