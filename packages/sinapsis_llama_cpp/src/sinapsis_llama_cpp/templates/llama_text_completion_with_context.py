# -*- coding: utf-8 -*-

from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base.base_models import (
    TemplateAttributeType,
)

from sinapsis_llama_cpp.templates.llama_text_completion import LLaMATextCompletion, LLaMATextCompletionAttributes


class LLaMATextCompletionWithContext(LLaMATextCompletion):
    """
    Template to initialize a LLaMA-based text completion model.
    This template sets up and initializes a LlaMA model with the corresponding
    configuration that is selected through the template attributes. It initializes the
    model, prepares the query and makes a call to the create_chat_completion method.
    Finally, if necessary, it post-processes the answer and returns it as a TextPacket
    in the DataContainer

    NEED TO ADD THE USAGE EXAMPLE
    """

    class AttributesBaseModel(LLaMATextCompletionAttributes):
        generic_key: str

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.extra_context: str | None = None

    def get_conv_id_and_content(self, packet: TextPacket) -> tuple[str, str]:
        """For an input packet, return a conversation id and the content of the message.
        If the incoming packet does not have id, generate a new one.
        If there is extra context for the query, add it to the prompt before returning it.

        Args:
            packet(TextPacket): the incoming packet
        Returns:
            tuple[str, str]: The pair of conversation_id and message
        """
        conv_id, prompt = super().get_conv_id_and_content(packet)
        if self.extra_context:
            prompt = f"{prompt} here is the context {self.extra_context}"
        return conv_id, prompt

    def execute(self, container: DataContainer) -> DataContainer:
        nodes_context = self._get_generic_data(container, self.attributes.generic_key)
        self.extra_context = " ".join(nodes_context)

        container = super().execute(container)
        return container
