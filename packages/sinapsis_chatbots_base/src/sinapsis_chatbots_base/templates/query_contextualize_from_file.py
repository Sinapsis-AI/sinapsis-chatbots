# -*- coding: utf-8 -*-
from typing import Any

from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base.base_models import OutputTypes, UIPropertiesMetadata

from sinapsis_chatbots_base.templates.query_contextualize import (
    QueryContextualize,
)


class QueryContextualizeFromFile(QueryContextualize):
    """A subclass of QueryContextualize that retrieves context from files loaded into
    the `generic_data`
    of a DataContainer.

    This class overrides the `retrieve_context` method to provide context retrieval
    from data files that have been previously read and stored in the `generic_data`
    attribute of a DataContainer. It checks if the provided keyword matches a key
    in the `generic_keys` attribute and then retrieves the associated context
    (in the form of `page_content`) from the `generic_data`.

    This subclass is intended for use cases where the context is dynamically loaded
     from external files (using a different template or process) and retrieved
     using keywords found in the content.


    """

    class AttributesBaseModel(QueryContextualize.AttributesBaseModel):
        """This class extends the AttributesBaseModel of the parent
        `QueryContextualize`.

        Attributes:
            keywords (list[str]): A list of keywords to be used for retrieving context.
            generic_keys (list[str]): A list of keywords that can be used to retrieve
                specific context data from the `generic_data` of the container.
        """

        generic_keys: list[str]

    UIProperties = UIPropertiesMetadata(category="Chatbots", output_type=OutputTypes.TEXT)

    def retrieve_context(self, keyword: str, context_data: dict | Any) -> str:
        """Retrieve the context associated with the given keyword from the provided
        context data.

        This method first checks if the `keyword` exists in the `generic_keys` attribute
         of the current instance.
        If a match is found, it then looks for the corresponding context data in the
        `context_data` passed to the method. If the `keyword` is found in both
        `generic_keys` and `context_data`, it returns the `page_content`
         associated with that keyword.

        If no match is found in either `generic_keys` or `context_data`, the method
         returns an empty string.

        Args:
            keyword (str): The key for which the context value is to be retrieved.
                This should correspond to an entry in the `generic_keys` attribute
                of the current instance or a key in the `context_data`.
            context_data (dict[str, Any]): A dictionary or any data source containing
                additional context data. The data should include relevant content
                associated with the `keyword`, typically in the form of `page_content`.

        Returns:
            str: The context value (i.e., `page_content`) associated with the `keyword`
            if found in both `generic_keys` and `context_data`; otherwise, an empty
            string if no match is found.
        """
        matching_key = next(
            (key for key in self.attributes.generic_keys if keyword.lower() in key.lower()),
            None,
        )
        if matching_key and matching_key in context_data:
            return context_data[matching_key][0].page_content
        return ""

    def add_context_to_content(self, kwd: str, container: DataContainer) -> str:
        """
        Depending on the keyword, append the additional context to the current one
        Args:
             kwd (str): kwd to be added in the context
             container (DataContainer): Container where additional context comes from
        Returns:
            str : The updated context
        """

        context = self.retrieve_context(kwd, container.generic_data) + "\n"

        return context
