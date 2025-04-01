# -*- coding: utf-8 -*-
import abc
from typing import Any

from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template, TemplateAttributes


class QueryContextualize(Template, abc.ABC):
    """A base class for contextualizing queries based on certain keywords."""

    class AttributesBaseModel(TemplateAttributes):
        """Attributes for contextualization.

        Attributes:
            keywords (list[str]): A list of keywords to be used for retrieving context.
        """

        keywords: list[str]

    @abc.abstractmethod
    def retrieve_context(self, keyword: str, context_data: dict | Any) -> str:
        """
        Abstract method to retrieve context related to a given keyword.

        This method must be implemented by subclasses. The implementation should
        return context information associated with the provided `keyword`.

        Args:
            keyword (str): The keyword for which context is to be retrieved.
            context_data (dict | Any): The data source (could be a dictionary or any other type)
                                   from which the context information will be retrieved.

        Returns:
            str: The context related to the provided keyword.
                  The return type should be a string representing the relevant context.
                  If no context is found, the method should return an empty string.
        """

    def load_context(self, container: DataContainer) -> None:
        """
        Loads context into the text packets within the provided `DataContainer`
        based on keywords.

        This method iterates over the text packets in the container and appends the
        retrieved context for each keyword found in the packet's content. It retrieves
        the context using the `retrieve_context` method and appends it to the text
        packet's content.

        Args:
            container (DataContainer): A container holding the text packets that
            need contextualization.

        The method checks each text packet in the container, looks for keywords in the
        packet's content, and appends the corresponding context
        (retrieved via `retrieve_context`) to the content if found.
        """

        for text_packet in container.texts:
            context: str = ""
            for keyword in self.attributes.keywords:
                if keyword in text_packet.content:
                    context += self.add_context_to_content(keyword, container)

            if context:
                text_packet.content += f" here is the context: {context}"

            if context:
                text_packet.content += f" here is the context: {context}"

    @abc.abstractmethod
    def add_context_to_content(self, kwd: str, container: DataContainer) -> str:
        """
        Depending on the keyword, append the additional context to the current one
        Args:
             kwd (str): kwd to be added in the context
             container (DataContainer): Container where additional context comes from
        Returns:
            str : The updated context
        """

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Executes the contextualization process on the provided DataContainer.

        Args:
            container (DataContainer): A container holding the texts that need
            contextualization.

        Returns:
            DataContainer: The updated DataContainer with the added context to its
            texts.

        This method checks if there are texts to process and then loads context into the
        container
        using the `load_context` method.
        """
        if not container.texts:
            self.logger.info("No query to process.")
            return container
        self.load_context(container)

        return container
