# -*- coding: utf-8 -*-
from typing import Any

from sinapsis_core.data_containers.data_packet import DataContainer

from sinapsis_mem0.templates.mem0_base import Mem0Base, Mem0BaseAttributes


class Mem0AddAttributes(Mem0BaseAttributes):
    """Configuration attributes for memory addition operations.

    Attributes:
        add_kwargs (dict[str, str]): Dictionary of parameters to pass to Mem0's add API.
                    Common keys include 'user_id', 'agent_id', and 'metadata'.
        generic_key (str): Key used to retrieve original prompts from the container's
                    generic data storage.
    """

    add_kwargs: dict[str, Any]
    generic_key: str


class Mem0Add(Mem0Base):
    """Handles formatting and storage of conversation history in Mem0.

    Transforms user prompts and LLM responses into Mem0's message format
    and persists them via the Mem0 API or the local infrastructure.

    Usage example:
    agent:
    name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: Mem0Add
    class_name: Mem0Add
    template_input: InputTemplate
    attributes:
        use_managed: true
        memory_config:
            host: https://api.mem0.ai
            org_id: null
            project_id: null
        add_kwargs:
            user_id: 'my_user'
        generic_key: 'Mem0Search'
    """

    AttributesBaseModel = Mem0AddAttributes

    def _form_messages(self, user_prompts: list[str], llm_responses: list[str]) -> list[dict[str, str]]:
        """Formats conversation pairs into Mem0's expected message structure.

        Args:
            user_prompts (list[str]): List of original user input strings.
            llm_responses (list[str]): List of corresponding LLM response strings.

        Returns:
            list[dict[str, str]]: List of message dictionaries in the expected format.
        """
        messages = []
        for prompt, response in zip(user_prompts, llm_responses):
            if "user_id" in self.attributes.add_kwargs:
                messages.append({"role": "user", "content": prompt})
            if "agent_id" in self.attributes.add_kwargs:
                messages.append({"role": "assistant", "content": response})
        return messages

    def execute(self, container: DataContainer) -> DataContainer:
        """Processes conversation history and stores it in Mem0.

        Args:
            container (DataContainer): Data container holding the original queries in the generic data field
                and the llm responses. It's expected that the number of llm outputs and original queries match.

        Returns:
            DataContainer: The unmodified input container
        """
        user_prompts = self._get_generic_data(container)
        if not user_prompts:
            self.logger.debug("No original prompts found in the generic data.")
            return container

        n_prompts = len(user_prompts)

        for i in range(n_prompts):
            container.texts[i].content = user_prompts[i]

        llm_responses = [p.content for p in container.texts[-n_prompts:]] if n_prompts else []
        messages = self._form_messages(user_prompts, llm_responses)
        self._generate_response("add", messages=messages, **self.attributes.add_kwargs)

        return container
