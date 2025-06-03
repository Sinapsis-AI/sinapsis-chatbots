# -*- coding: utf-8 -*-
from abc import ABC
from typing import Any

from mem0 import Memory, MemoryClient
from mem0.client.main import APIError
from pydantic import Field
from sinapsis_core.template_base import Template, TemplateAttributes, TemplateAttributeType
from sinapsis_core.template_base.base_models import OutputTypes, UIPropertiesMetadata

from sinapsis_mem0.helpers.env_var_keys import MEM0_API_KEY


class Mem0BaseAttributes(TemplateAttributes):
    """Configuration base attributes for Mem0 templates.

    Attributes:
        use_managed (bool): If True, use the managed Mem0 API (MemoryClient).
            If False, use the local memory (Memory).
        memory_config (dict[str, Any]): Parameters to configure either MemoryClient or Memory,
            depending on the value of `use_managed`.
    """

    use_managed: bool = False
    memory_config: dict[str, Any] = Field(default_factory=dict)


class Mem0Base(Template, ABC):
    """Base template for Mem0 templates providing core API and self hosted functionality.

    This class initializes the appropriate memory backend — either the managed
    `MemoryClient` (via API key) or the self-hosted `Memory` class — depending
    on the `use_managed` flag in the template attributes.
    """

    AttributesBaseModel = Mem0BaseAttributes
    UIProperties = UIPropertiesMetadata(category="Chatbots", output_type=OutputTypes.TEXT)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.client = (
            MemoryClient(api_key=MEM0_API_KEY, **self.attributes.memory_config)
            if self.attributes.use_managed
            else Memory.from_config(self.attributes.memory_config)
        )

    def _generate_response(self, method_name: str, **kwargs) -> list[dict[str, Any]] | dict[str, Any] | None:
        """Dynamically call a method on the underlying memory backend (local or managed).

        This method routes calls to either a `MemoryClient` or `Memory` instance,
        depending on the `use_managed` flag. It performs method existence checks,
        logs execution details, and captures common errors such as invalid input
        or failed API calls.

        Args:
            method_name (str): The name of the method to call on the memory instance.
        Returns:
            list[dict[str, Any]] | dict[str, Any] | None: The result of the method call, or None if failed.
        """
        if not hasattr(self.client, method_name):
            self.logger.error(f"{type(self.client).__name__} has no method '{method_name}'")
            return None

        method = getattr(self.client, method_name)
        self.logger.debug(f"Calling {type(self.client).__name__}.{method_name} with kwargs: {kwargs}")

        try:
            response = method(**kwargs)
            self.logger.debug(f"{type(self.client).__name__}.{method_name} response: {response}")
            return response
        except (ValueError, APIError) as e:
            self.logger.error(f"{type(self.client).__name__} {method_name} failed: {e}")
            return None
