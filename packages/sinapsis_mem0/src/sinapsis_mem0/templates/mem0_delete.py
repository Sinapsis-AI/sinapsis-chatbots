# -*- coding: utf-8 -*-
from typing import Any

from sinapsis_core.data_containers.data_packet import DataContainer

from sinapsis_mem0.templates.mem0_base import Mem0Base, Mem0BaseAttributes


class Mem0DeleteAttributes(Mem0BaseAttributes):
    """Configuration attributes for memory deletion operations in Mem0.

    Attributes:
        delete_all: If True, performs a complete memory wipe for the specified scope (agent, run or user).
                    If False, performs targeted deletion based on `memory_id`.
        delete_kwargs: Parameters for the deletion operation. Depending on the type of deletion, this may
            include `user_id`, `agent_id`, or `memory_id`.
    """

    delete_all: bool = False
    delete_kwargs: dict[str, Any]


class Mem0Delete(Mem0Base):
    """Template for deleting memories from Mem0's storage.

    Supports two deletion modes: all memories or specific memory by ID.

    Usage example:
    agent:
    name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: Mem0Delete
    class_name: Mem0Delete
    template_input: InputTemplate
    attributes:
        use_managed: true
        memory_config:
            host: https://api.mem0.ai
            org_id: null
            project_id: null
        delete_all: true
        delete_kwargs:
            user_id: my_user
    """

    AttributesBaseModel = Mem0DeleteAttributes

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes memory deletion operation based on configured attributes.

        Args:
            container (DataContainer): Input data container

        Returns:
            DataContainer: The unmodified input container
        """
        method_to_use = "delete_all" if self.attributes.delete_all else "delete"
        self._generate_response(method_to_use, **self.attributes.delete_kwargs)
        return container
