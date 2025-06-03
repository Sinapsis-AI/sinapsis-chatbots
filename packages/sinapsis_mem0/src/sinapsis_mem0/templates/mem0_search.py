# -*- coding: utf-8 -*-
from typing import Any, Literal

from pydantic.dataclasses import dataclass
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base import TemplateAttributeType

from sinapsis_mem0.templates.mem0_base import Mem0Base, Mem0BaseAttributes


@dataclass(frozen=True)
class EnclosureFormats:
    """
    String templates for formatting memory/context injection into prompts for LLMs.

    Attributes:
        plain (str):   Memories and query are inserted with a simple newline separator.
        bracket (str): Memories and query are each introduced by bracketed section headers.
        dashed (str):  Uses dashed section headers for both memories and the query, for clear separation.
        xml (str):     Uses XML-like tags to wrap the memories and the query for explicit LLM delimiting.
    """

    plain: str = "{memories}\n\n{query}"
    bracket: str = "[Relevant memories]\n{memories}\n\n[Query]\n{query}"
    dashed: str = "---- Relevant Memories ----\n{memories}\n\n---- Query ----\n{query}"
    xml: str = "<Memories>\n{memories}\n</Memories>\n<Query>\n{query}\n</Query>"

    @classmethod
    def get(cls, name: str) -> str:
        """Retrieve the template string for the given enclosure style.

        Args:
            name (str): The name of the enclosure style.
                        Should be one of: 'plain', 'bracket', 'dashed', or 'xml'.

        Returns:
            str: The corresponding template string.
        """
        return getattr(cls, name, cls.plain)


class Mem0SearchAttributes(Mem0BaseAttributes):
    """Configuration attributes for Mem0 memory search operations.

    Attributes:
        search_kwargs (dict[str, Any]):
            Additional parameters for the search API call.
            Common options include:
              - 'user_id': The user ID associated with the memory
              - 'top_k': The number of top results to return
              - 'threshold': The minimum similarity score for returned results

        enclosure (Literal["plain", "bracket", "dashed", "xml"], default "plain"):
            Determines how relevant memories are injected into the prompt before the user query.
            Options:
              - "plain": No special section or title, just memories and query.
              - "bracket": Section headers "[Relevant memories]" and "[Query]" wrap each block.
              - "dashed": Uses dashed section headers for both memories and query.
              - "xml": Wraps memories and query in XML-style tags ("<Memories>", "<Query>").
    """

    search_kwargs: dict[str, Any]
    enclosure: Literal["plain", "bracket", "dashed", "xml"] = "plain"


class Mem0Search(Mem0Base):
    """Template for retrieving and injecting relevant memories from Mem0.

    Searches Mem0's memory store based on input queries and enhances prompts
    with relevant contextual memories.

    Usage example:
    agent:
    name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: Mem0Search
    class_name: Mem0Search
    template_input: InputTemplate
    attributes:
        use_managed: true
        memory_config:
            host: https://api.mem0.ai
            org_id: null
            project_id: null
        search_kwargs:
            version: v1
            user_id: 'my_user'
            top_k: 3
        enclosure: bracket
    """

    AttributesBaseModel = Mem0SearchAttributes
    ENCLOSURE = EnclosureFormats()

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        if not self.attributes.use_managed:
            self.threshold = self.attributes.search_kwargs.pop("threshold", 0.5)

    def _get_memories(self, query: str) -> list[str]:
        """Retrieves relevant memories from Mem0 for a given query.

        Args:
            query (str): The search string to find related memories.

        Returns:
            list[str]: List of memory strings ordered by relevance.
        """
        response = self._generate_response("search", query=query, **self.attributes.search_kwargs)
        if response is None:
            return []
        if isinstance(response, dict):
            response = response["results"]
            response = [entry for entry in response if entry.get("score", 0) > self.threshold]
        memories = [entry["memory"] for entry in response]

        return memories

    def _update_prompt(self, text_packet: TextPacket, memories: list[str]) -> None:
        """Enhances a text prompt with relevant memories. No modification occurs if memories list is empty.

        Args:
            text_packet (TextPacket): The original text packet to modify.
            memories (list[str]): List of memory strings to inject.
        """
        if not memories:
            return
        memories_str = "\n".join(f"* {m}" for m in memories)
        template = self.ENCLOSURE.get(self.attributes.enclosure)
        text_packet.content = template.format(memories=memories_str, query=text_packet.content)

    def execute(self, container: DataContainer) -> DataContainer:
        """Processes all texts in container, enhancing them with memories (if available).

        Args:
            container (DataContainer): Input data with texts to enhance.

        Returns:
            DataContainer: Modified container with enhanced prompts and the original prompts in
                the generic data field.
        """
        if not container.texts:
            self.logger.debug("No text to enhance in the container.")
            return container

        original_prompts = []

        for text_packet in container.texts:
            original_prompts.append(text_packet.content)
            memories = self._get_memories(text_packet.content)
            self._update_prompt(text_packet, memories)

        self._set_generic_data(container, original_prompts)

        return container
