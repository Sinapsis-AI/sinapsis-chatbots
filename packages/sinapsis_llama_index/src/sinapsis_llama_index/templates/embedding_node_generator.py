# -*- coding: utf-8 -*-
from typing import Literal

from llama_index.core import node_parser
from llama_index.core.bridge.langchain import Document as LCDocument
from llama_index.core.constants import DEFAULT_CHUNK_SIZE
from llama_index.core.node_parser import TextSplitter
from llama_index.core.node_parser.text.sentence import CHUNKING_REGEX, DEFAULT_PARAGRAPH_SEP, SENTENCE_CHUNK_OVERLAP
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import Field
from sinapsis_chatbots_base.helpers.tags import Tags
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)

from sinapsis_llama_index.helpers.schemas import HFEmbeddingConfig, SplitterArgs


class SentenceSplitterArgs(SplitterArgs):
    """Configuration arguments for LlamaIndex's `SentenceSplitter`.

    Inherits base settings from `SplitterArgs` and adds parameters specific to sentence-based text splitting.

    Attributes:
        separator (str): The separator string to use when joining text chunks. Defaults to `" "`.
        chunk_size (int): The maximum size (in tokens or characters, depending on
            the splitter) of each chunk. Defaults to `DEFAULT_CHUNK_SIZE`.
        chunk_overlap (int): The number of tokens/characters to overlap between
            consecutive chunks. Defaults to `SENTENCE_CHUNK_OVERLAP`.
        paragraph_separator (str): The string used to identify paragraph breaks
            within the text. Defaults to `DEFAULT_PARAGRAPH_SEP`.
        secondary_chunking_regex (str): A regex pattern used for secondary
            splitting within larger chunks. Defaults to `CHUNKING_REGEX`.
    """

    separator: str = " "
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = SENTENCE_CHUNK_OVERLAP
    paragraph_separator: str = DEFAULT_PARAGRAPH_SEP
    secondary_chunking_regex: str = CHUNKING_REGEX


class EmbeddingNodeGenerator(Template):
    r"""A class for generating text embeddings using a HuggingFace model.

    This class is responsible for splitting documents into chunks, generating
    corresponding `TextNode` objects, and creating text embeddings using a HuggingFace
    model. It uses the LlamaIndex library's utilities for splitting text and generating
    embeddings.

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
          separator: ' '
          chunk_size: 1024
          chunk_overlap: 200
          paragraph_separator: '\n\n\n'
        embedding_config:
          model_name: 'Snowflake/snowflake-arctic-embed-xs'
          normalize: true
          embed_batch_size: 10
          trust_remote_code: true
          device: auto
          parallel_process: false
        generic_keys: ['GitLoader']
    """

    SPLITTER_CLASS: Literal["SentenceSplitter", "CodeSplitter"] = "SentenceSplitter"

    class AttributesBaseModel(TemplateAttributes):
        """A class for holding the attributes required for text chunking and embedding.

        Attributes:
            splitter_args (SentenceSplitterArgs): Configuration for the text splitter (e.g., chunk size, overlap).
            embedding_config (HFEmbeddingConfig): Configuration for the HuggingFace embedding model (e.g., model
                name, device).
            generic_keys (list[str] | None): The list of keys for retrieving the document data from
                the container. If `None`, it will try to use `container.texts`. Defaults to `None`.
        """

        splitter_args: SentenceSplitterArgs = Field(default_factory=SentenceSplitterArgs)
        embedding_config: HFEmbeddingConfig
        generic_keys: list[str] | None = None

    UIProperties = UIPropertiesMetadata(
        category="Embeddings",
        output_type=OutputTypes.MULTIMODAL,
        tags=[Tags.EMBEDDINGS, Tags.HUGGINGFACE, Tags.TEXT, Tags.DOCUMENTS, Tags.QUERY_CONTEXTUALIZATION, Tags.QUERY],
    )

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """Initializes the HuggingFaceEmbeddingNodeGenerator instance.

        This constructor sets up the `SentenceSplitter` and `HuggingFaceEmbedding` model
        using the provided attributes.
        """
        super().__init__(attributes)
        self.splitter = self._init_splitter()
        self.model = self._init_embed_model()

    def _init_splitter(self) -> TextSplitter:
        """Initializes the TextSplitter instance based on SPLITTER_CLASS and attributes.

        Returns:
            TextSplitter: The initialized splitter instance.
        """
        splitter_class = getattr(node_parser, self.SPLITTER_CLASS)
        return splitter_class(**self.attributes.splitter_args.model_dump())

    def _init_embed_model(self) -> HuggingFaceEmbedding:
        """Initialize the embedding model.

        This method initializes the embedding model using the HuggingFace API. It can be overridden by subclasses
        to provide custom embedding model initialization.

        Returns:
            HuggingFaceEmbedding: The initialized embedding model.
        """
        model = HuggingFaceEmbedding(**self.attributes.embedding_config.model_dump(exclude_none=True))
        return model

    @staticmethod
    def _to_document(doc_data: str | dict | LCDocument) -> Document | None:
        """Converts a single item (str, dict, Document, LCDocument) into a LlamaIndex Document.

        Args:
            doc_data (str | dict | LCDocument): The data to convert.

        Returns:
            Document | None: A LlamaIndex Document object, or None if the input type is unsupported.
        """
        if isinstance(doc_data, str):
            return Document(text=doc_data)
        elif isinstance(doc_data, dict):
            return Document(**doc_data)
        elif isinstance(doc_data, LCDocument):
            return Document.from_langchain_format(doc_data)
        else:
            return None

    def load_documents(self, container: DataContainer) -> list[Document]:
        """Loads documents from the container based on the available generic keys.

        Args:
            container (DataContainer): The container with document data.

        Returns:
            list[Document]: A list of Document
        """
        if self.attributes.generic_keys:
            documents = []
            for key in self.attributes.generic_keys:
                docs_data: list | dict | str = self._get_generic_data(container, key)
                if isinstance(docs_data, list):
                    for document in docs_data:
                        llama_document = self._to_document(document)
                        if llama_document:
                            documents.append(llama_document)
                        else:
                            self.logger.debug("Skipping document with unexpected data format.")
                else:
                    llama_document = self._to_document(docs_data)
                    if llama_document:
                        documents.append(llama_document)
                    else:
                        self.logger.debug("Skipping document with unexpected data format.")
            return documents
        else:
            return [Document(text=text.content) for text in container.texts]

    def filter_documents(self, documents: list[Document]) -> list[Document]:
        """Filters the list of documents before they are chunked.

        This base method acts as a passthrough and does not filter out any
        documents. Subclasses can override this method to implement
        specific document filtering logic.

        Args:
            documents (list[Document]): The list of standardized LlamaIndex
                Document objects to be filtered.

        Returns:
            list[Document]: The list of documents to be processed (unfiltered
                in this base class).
        """
        _ = self
        return documents

    def process_chunk(self, document: Document) -> list[Document]:
        """Splits a single Document into multiple Document chunks using the configured splitter.

        Metadata from the original document is copied to each chunk.

        Args:
            document (Document): The LlamaIndex Document object to split.

        Returns:
            list[Document]: A list of new Document objects, each containing a text chunk.
        """
        text_chunks: list[str] = self.splitter.split_text(document.text)
        chunked_documents = [Document(text=text_chunk, metadata=document.metadata) for text_chunk in text_chunks]

        return chunked_documents

    def generate_chunks(self, documents: list[Document]) -> list[Document]:
        """Splits a list of Documents into a flattened list of Document chunks.

        Args:
            documents (list[Document]): The list of LlamaIndex Document objects to split.

        Returns:
            list[Document]: A flattened list containing Document objects representing chunks
                           from all input documents.
        """
        chunked_documents: list[Document] = []

        for document in documents:
            cur_text_chunks = self.process_chunk(document)
            chunked_documents.extend(cur_text_chunks)

        return chunked_documents

    @staticmethod
    def generate_nodes(documents: list[Document]) -> list[TextNode]:
        """Converts a list of Document chunks into a list of TextNode objects.

        Args:
            documents (list[Document]): The list of Document objects (representing chunks)
                                       to convert.

        Returns:
            list[TextNode]: A list of TextNode objects created from the Document chunks.
        """
        nodes: list[TextNode] = [TextNode(text=document.text, metadata=document.metadata) for document in documents]
        return nodes

    def generate_embeddings(self, nodes: list[TextNode]) -> list[TextNode]:
        """Generates embeddings for each text chunk using the HuggingFace model.

        Args:
            nodes (list[TextNode]): The documents used to generate embeddings.

        Returns:
            list[TextNode]: A list of `TextNode` objects with embeddings attached.
        """
        texts_to_embed = [node.get_content(metadata_mode="all") for node in nodes]
        embeddings = self.model.get_text_embedding_batch(texts_to_embed, show_progress=True)

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the embedding generation process using data from the container.

        This method retrieves the document data from the container using the
        specified keys in `generic_keys`, generates embeddings for the documents, and sets
        the resulting nodes back into the container.

        Args:
            container (DataContainer): A container holding document data to be processed.

        Returns:
            DataContainer: The updated container with the generated nodes and embeddings.
        """
        self.logger.debug(f"Starting execution of {self.instance_name}")
        documents = self.load_documents(container)
        if documents is None:
            return container

        filtered_documents = self.filter_documents(documents)
        chunks = self.generate_chunks(filtered_documents)
        nodes = self.generate_nodes(chunks)
        embedding_nodes = self.generate_embeddings(nodes)
        self._set_generic_data(container, embedding_nodes)
        self.logger.debug(f"Saved {self.instance_name} nodes as generic data")
        return container
