# -*- coding: utf-8 -*-
from typing import Literal

from llama_index.core.node_parser import CodeSplitter
from sinapsis_core.template_base.base_models import OutputTypes, TemplateAttributeType, UIPropertiesMetadata
from sinapsis_llama_index.templates.embedding_node_generator import DocumentType, EmbeddingNodeGenerator

language_mapping: dict = {
    "python": ".py",
    "sql": ".sql",
    "javascript": ".js",
    "java": "java",
    "json": ".json",
    "bash": ".sh",
    "markdown": ".md",
    "ruby": ".rb",
    "cpp": ".cpp",
    "css": ".css",
    "yaml": ".yaml",
    "html": ".html",
}


class CodeEmbeddingNodeGenerator(EmbeddingNodeGenerator):
    """Template to generate nodes for a code base.
    It performs a chunking strategy based on the file with the code
    and returns meaningful Nodes that are transported in the generic_data field
    of the DataContainer
    """

    class AttributesBaseModel(EmbeddingNodeGenerator.AttributesBaseModel):
        """
        Attributes:
            chunk_size (int): The maximum size of each chunk of text after splitting
            separator (str): The separator used to split the document into chunks
            model_name (str): Name of the model to generate embeddings
            generic_keys (list[str] | str | None): Key or list of keys for retrieving
                        the document data from the container
            programming_language (Literal): list of allowed programming language to use with
                        the CodeSplitter
        """

        programming_language: Literal[tuple(language_mapping)] | None = "python"  # type:ignore[valid-type]

    UIProperties = UIPropertiesMetadata(category="Embeddings", output_type=OutputTypes.MULTIMODAL)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.code_splitter = (
            CodeSplitter(language=self.attributes.programming_language)
            if self.attributes.programming_language
            else None
        )

    def language_in_metadata(self, document: DocumentType) -> bool:
        """Method to check if the programming language from the attributes is in the
        metadata of the Document
        Args:
            document (DocumentType): Document to check metadata in
        Returns:
            bool: Whether the programming language is in the metadata values
        """
        return language_mapping.get(self.attributes.programming_language, False) in document.metadata.values()

    @staticmethod
    def is_not_init_file(document: DocumentType) -> bool:
        """
        Method to determine if the document is an init file from python
        Args:
            document (DocumentType): Document to check the extension

        Returns:
            bool: Whether the file is a python init file
        """
        return not document.metadata.get("file_path", False).endswith("__init__.py")

    def process_chunk(self, doc: str | DocumentType) -> list | None:
        """
        Method to process the chunk of code using the CodeSplitter if the
        incoming document if a code from a selected programming language. If the
        document is a string or belongs to a code other than the programming
        language, the chunking strategy is done using a SentenceSplitter

        Args:
            doc (str | DocumentType): Incoming document for the chunking. Can be a string
                        or a Document object from LlamaIndex

        Returns:
            list : The list of chunks that will be used to get the embeddings
        """
        if not isinstance(doc, str):
            merged_chunks = []
            code_node = ""
            if self.code_splitter and self.language_in_metadata(doc) and self.is_not_init_file(doc):
                cur_text_chunks = self.code_splitter.split_text(self._get_text(doc))
                for chunk in cur_text_chunks:
                    code_node += chunk + "\n"
                    if len(code_node) >= self.attributes.chunk_size:
                        merged_chunks.append(code_node.strip())
                        code_node = ""
                if code_node:
                    merged_chunks.append(code_node.strip())
                return merged_chunks

        else:
            cur_text_chunks = super().process_chunk(doc)
            return cur_text_chunks
        return None
