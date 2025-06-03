# -*- coding: utf-8 -*-
from sinapsis_core.data_containers.data_packet import DataContainer

from sinapsis_mem0.templates.mem0_base import Mem0Base


class Mem0Reset(Mem0Base):
    """Template for resetting Mem0's memory storage.

    Behavior differs by configuration:
    - Local Mode (use_managed=False): Deletes the vector store collection, resets the database and
    recreates the vector store with a new client.

    - Managed API Mode (use_managed=True): deletes all users, agents, sessions, and memories associated
    with the client.

    Usage example:
    agent:
    name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: Mem0Get
    class_name: Mem0Get
    template_input: InputTemplate
    attributes:
        use_managed: true
        memory_config:
            host: https://api.mem0.ai
            org_id: null
            project_id: null
    """

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes memory retrieval based on configured attributes.

        If `get_all` is True, it retrieves all memories in scope (e.g., for a user).
        Otherwise, it retrieves specific memory using the provided `memory_id` in `get_kwargs`.

        Args:
            container (DataContainer): Input data container

        Returns:
            DataContainer: The unmodified input container
        """
        self._generate_response("reset")
        return container
