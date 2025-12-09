# -*- coding: utf-8 -*-
from typing import Literal

from llama_index.core.node_parser.text.code import DEFAULT_CHUNK_LINES, DEFAULT_LINES_OVERLAP, DEFAULT_MAX_CHARS
from llama_index.core.schema import Document
from pydantic import BaseModel, Field
from sinapsis_chatbots_base.helpers.tags import Tags

from sinapsis_llama_index.helpers.schemas import SplitterArgs
from sinapsis_llama_index.templates.embedding_node_generator import EmbeddingNodeGenerator

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

CodeEmbeddingNodeGeneratorUIProperties = EmbeddingNodeGenerator.UIProperties
CodeEmbeddingNodeGeneratorUIProperties.tags.extend([Tags.CODE])


class FileExclusionConfig(BaseModel):
    """Configuration for filtering documents based on metadata keys.

    Defines rules for excluding documents based on their file path or type,
    which are typically found in the document's metadata.

    Attributes:
        startswith_exclude (list[str]): A list of string prefixes. Documents whose
            file path (from `file_path_key`) starts with any of these strings
            will be excluded. Defaults to an empty list.
        endswith_exclude (list[str]): A list of string suffixes. Documents whose
            file path (from `file_path_key`) ends with any of these strings
            will be excluded (e.g., "__init__.py", ".tmp"). Defaults to an empty list.
        file_path_key (str): The key in the document's metadata dictionary that
            contains the full file path string to check against. Defaults to `"file_path"`.
        file_type_key (str): The key in the document's metadata dictionary that
            contains the file type string. Defaults to `"file_type"`.
    """

    startswith_exclude: list[str] = Field(default_factory=list)
    endswith_exclude: list[str] = Field(default_factory=list)
    file_path_key: str = "file_path"
    file_type_key: str = "file_type"


class CodeSplitterArgs(SplitterArgs):
    """Configuration arguments for LlamaIndex's `CodeSplitter`.

    Inherits base settings from `SplitterArgs` and adds parameters specific to splitting code into chunks based on
    line count.

    Attributes:
        chunk_lines (int): The number of lines to include in each chunk.
            Defaults to `DEFAULT_CHUNK_LINES`.
        chunk_lines_overlap (int): The number of lines to overlap between
            consecutive chunks. Defaults to `DEFAULT_LINES_OVERLAP`.
        max_chars (int): The maximum number of characters allowed in a chunk.
            Defaults to `DEFAULT_MAX_CHARS`.
    """

    language: Literal[tuple(language_mapping)] = "python"  # type:ignore[valid-type]
    chunk_lines: int = DEFAULT_CHUNK_LINES
    chunk_lines_overlap: int = DEFAULT_LINES_OVERLAP
    max_chars: int = DEFAULT_MAX_CHARS


class CodeEmbeddingNodeGenerator(EmbeddingNodeGenerator):
    r"""Template to generate nodes for a code base.

    It performs a chunking strategy based on the file with the code
    and returns meaningful Nodes that are transported in the generic_data field
    of the DataContainer.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: EmbeddingNodeGenerator
      class_name: EmbeddingNodeGenerator
      template_input: InputTemplate
      attributes:
        splitter_args:
          include_metadata: true
          include_prev_next_rel: true
          language: python
          chunk_lines: 40
          chunk_lines_overlap: 15
          max_chars: 1500
        embedding_config:
          model_name: 'Snowflake/snowflake-arctic-embed-xs'
          normalize: true
          embed_batch_size: 10
          trust_remote_code: true
          device: auto
          parallel_process: false
        exclusion_config:
          endswith_exclude: ['__init__.py']
          file_path_key: file_path
          file_type_key: file_type
        generic_keys: ['GitLoader']
    """

    SPLITTER_CLASS = "CodeSplitter"

    class AttributesBaseModel(EmbeddingNodeGenerator.AttributesBaseModel):
        """Attributes for the CodeEmbeddingNodeGenerator, extending EmbeddingNodeGenerator.

        Attributes:
            splitter_args (SentenceSplitterArgs): Configuration for the text splitter (e.g., chunk_lines size,
                max_chars).
            embedding_config (HFEmbeddingConfig): Configuration for the HuggingFace embedding model (e.g., model
                name, device).
            generic_keys (list[str] | str | None): The key or list of keys for retrieving the document data from
                the container. If `None`, it will try to use `container.texts`. Defaults to `None`.
            programming_language (Literal): list of allowed programming language to use with
                        the CodeSplitter
        """

        splitter_args: CodeSplitterArgs = Field(default_factory=CodeSplitterArgs)
        exclusion_config: FileExclusionConfig = Field(default_factory=FileExclusionConfig)

    def _language_in_metadata(self, document: Document) -> bool:
        """Method to check if the programming language from the attributes is in the metadata of the Document.

        Args:
            document (Document): Document to check metadata in
        Returns:
            bool: Whether the programming language is in the metadata values
        """
        file_type_key = self.attributes.exclusion_config.file_type_key
        file_type = document.metadata.get(file_type_key, None)
        if not file_type:
            self.logger.debug(f"Document metadata has no '{file_type_key}', skipping.")
            return False
        return language_mapping.get(self.attributes.splitter_args.language, False) == file_type

    def _should_process_document(self, document: Document) -> bool:
        """Determines if a document should be processed based on exclusion rules.

        Checks the document's 'file_path' metadata (using the key from
        `exclusion_config.file_path_key`) against the `startswith_exclude`
        and `endswith_exclude` lists in the `exclusion_config` attribute.

        Args:
            document (Document): The LlamaIndex Document object to check.

        Returns:
            bool: `True` if the document should be processed (i.e., it does not
                  match any exclusion rules, or has no file_path), `False` if
                  it matches an exclusion rule and should be skipped.
        """
        file_path_key = self.attributes.exclusion_config.file_path_key
        file_path = document.metadata.get(file_path_key, None)
        if not file_path:
            self.logger.debug(f"Document metadata has no '{file_path_key}', cannot apply file path exclusion rule.")
            return True

        startswith_tuple = tuple(self.attributes.exclusion_config.startswith_exclude)
        endswith_tuple = tuple(self.attributes.exclusion_config.endswith_exclude)

        if startswith_tuple and file_path.startswith(startswith_tuple):
            self.logger.debug(f"Excluding document due to startswith match: {file_path}")
            return False
        if endswith_tuple and file_path.endswith(endswith_tuple):
            self.logger.debug(f"Excluding document due to endswith match: {file_path}")
            return False

        return True

    def filter_documents(self, documents: list[Document]) -> list[Document]:
        """Filters the list of documents before they are chunked.

        This method filters documents based on the language being in the metadata and if it is not marked to
        be excluded.

        Args:
            documents (list[Document]): The list of standardized LlamaIndex
                Document objects to be filtered.

        Returns:
            list[Document]: The list of filtered documents
        """
        filtered_documents = [
            document
            for document in documents
            if self._language_in_metadata(document) and self._should_process_document(document)
        ]
        return filtered_documents
