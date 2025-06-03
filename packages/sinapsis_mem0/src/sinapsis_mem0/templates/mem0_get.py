# -*- coding: utf-8 -*-
from typing import Any

from sinapsis_core.data_containers.data_packet import DataContainer

from sinapsis_mem0.templates.mem0_base import Mem0Base, Mem0BaseAttributes


class Mem0GetAttributes(Mem0BaseAttributes):
    """Configuration attributes for memory retrieval operations in Mem0.

    Attributes:
        get_all (bool): If True, retrieves all memories for the given context (e.g., user, agent, or run).
            If False, retrieves a specific memory using parameters like `memory_id`.
        get_kwargs (dict[str, Any]): Additional parameters to pass to the memory retrieval method.
            Can include fields such as `user_id`, `agent_id`, or `memory_id`, depending on the context.
    """

    get_all: bool = False
    get_kwargs: dict[str, Any]


class Mem0Get(Mem0Base):
    """Template for retrieving memories from Mem0's storage.

    Supports retrieving all memories for a given context, or retrieving specific ones
    using identifiers such as `memory_id`, `user_id`, or `agent_id`.

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
        get_all: true
        get_kwargs:
            user_id: my_user
    """

    AttributesBaseModel = Mem0GetAttributes

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes memory retrieval based on configured attributes.

        If `get_all` is True, it retrieves all memories in scope (e.g., for a user).
        Otherwise, it retrieves specific memory using the provided `memory_id` in `get_kwargs`.

        Args:
            container (DataContainer): Input data container

        Returns:
            DataContainer: The unmodified input container
        """
        method_to_use = "get_all" if self.attributes.get_all else "get"
        memories = self._generate_response(method_to_use, **self.attributes.get_kwargs)
        if isinstance(memories, dict):
            memories = memories["results"]
        self._set_generic_data(container, memories)
        return container
