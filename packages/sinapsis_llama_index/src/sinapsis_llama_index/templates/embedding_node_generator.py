# -*- coding: utf-8 -*-

from typing import Any

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)

DocumentType = str | list[Document] | list[Any]


class EmbeddingNodeGenerator(Template):
    """
    A class for generating text embeddings using the HuggingFace model.

    This class is responsible for splitting documents into chunks, generating
    corresponding `TextNode` objects, and creating text embeddings using a HuggingFace
    model. It uses the LlamaIndex library's utilities for splitting text and generating
    embeddings.
    """

    class AttributesBaseModel(TemplateAttributes):
        """
        A class for holding the attributes required for text chunking and embedding.

        Attributes:
            chunk_size (int): The maximum size of each chunk of text after splitting.
            separator (str): The separator used to split the document into chunks.
            model_name (str): The name of the HuggingFace model used to generate embeddings.
            generic_keys (list[str] | str | None): The key or list of keys for retrieving the
                document data from the container.
            trust_remote_code (bool): Flaf to determine if remote code should be trusted
            device (bool): Device to load the model to
        """

        chunk_size: int
        separator: str
        model_name: str
        generic_keys: list[str] | str | None = None
        trust_remote_code: bool = False
        device: str = "cpu"

    UIProperties = UIPropertiesMetadata(category="Embeddings", output_type=OutputTypes.MULTIMODAL)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """
        Initializes the HuggingFaceEmbeddingNodeGenerator instance.

        This constructor sets up the `SentenceSplitter` and `HuggingFaceEmbedding` model
        using the provided attributes.
        """
        super().__init__(attributes)
        self.splitter = SentenceSplitter(chunk_size=self.attributes.chunk_size, separator=self.attributes.separator)
        self.model = HuggingFaceEmbedding(
            model_name=self.attributes.model_name,
            trust_remote_code=self.attributes.trust_remote_code,
            device=self.attributes.device,
        )

    @staticmethod
    def _process_documents(documents: DocumentType) -> list:
        """
        Helper method to process and flatten documents (in case it's nested).
        """
        if isinstance(documents, str):
            return [documents]
        elif documents and isinstance(documents[0], list):
            return [doc for _, sublist in enumerate(documents) for doc in sublist]
        else:
            return [doc for _, doc in enumerate(documents)]

    def process_chunk(self, doc: str | DocumentType) -> list | None:
        """Method to process the chunk of code using the SentenceSplitter class
        from LlamaIndex. It processes the chunks assuming a max size and a separator

        Args:
            doc (str | DocumentType): Incoming document for the chunking. Can be a string
        or a Document object from LlamaIndex

        Returns:
            list : The list of chunks that will be used to get the embeddings
        """
        text_chunks = self.splitter.split_text(self._get_text(doc))
        return text_chunks

    def generate_chunks(self, documents: DocumentType) -> tuple[list[str], list[int]]:
        """
        Splits documents into smaller text chunks using the process_chunks method

        This method processes the input documents, splits them into chunks based on
        the specified chunk size, and associates each chunk with its original document
        index.

        Args:
            documents (DocumentType): The document(s) to split into chunks. This can
                be either a single string, a list of `Document` objects, or a list
                of any other type.

        Returns:
            tuple[list[str], list[int]]: A tuple where the first list contains the
                chunks of text and the second list contains the indices of the documents
                from which each chunk was extracted.
        """
        text_chunks: list = []
        metadata: list = []

        processed_docs = self._process_documents(documents)

        for doc_id, doc in enumerate(processed_docs):
            cur_text_chunks = self.process_chunk(doc)
            if cur_text_chunks:
                text_chunks.extend(cur_text_chunks)
                doc_metadata = doc.metadata if not isinstance(doc, str) else None
                if doc_metadata:
                    metadata.extend([doc_metadata] * len(cur_text_chunks))
                else:
                    metadata.extend([None] * len(cur_text_chunks))

        return text_chunks, metadata

    def generate_nodes(self, documents: DocumentType) -> list[TextNode]:
        """
        Generates `TextNode` objects from the text chunks and adds metadata.

        Args:
            documents (DocumentType): The documents used to generate the `TextNode` objects.

        Returns:
            list[TextNode]: A list of `TextNode` objects, each containing a chunk of text
                and associated metadata (if available).
        """

        text_chunks, metadata = self.generate_chunks(documents)
        nodes: list[TextNode] = []

        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(
                text=text_chunk,
            )

            if metadata[idx] is not None:
                node.metadata = metadata[idx]
            else:
                self.logger.debug(f"Skipping metadata for chunk: {text_chunk}")
            nodes.append(node)
        return nodes

    def generate_embeddings(self, documents: DocumentType) -> list[TextNode]:
        """
        Generates embeddings for each text chunk using the HuggingFace model.

        Args:
            documents (DocumentType): The documents used to generate embeddings.

        Returns:
            list[TextNode]: A list of `TextNode` objects with embeddings attached.
        """
        nodes: list[TextNode] = self.generate_nodes(documents)

        for node in nodes:
            node_embedding = self.model.get_text_embedding(node.get_content(metadata_mode="all"))
            node.embedding = node_embedding
        return nodes

    @staticmethod
    def _get_text(doc: Document | str) -> str:
        """Returns the text content of a Document or string.

        Args:
            doc (Document | str): A document with `text` or `page_content`, or a string.

        Raises:
            AttributeError: If the document lacks a text attribute.

        Returns:
            str: Extracted text content.
        """

        if hasattr(doc, "text"):
            return doc.text
        if hasattr(doc, "page_content"):
            return doc.page_content
        return doc

    def load_documents(self, container: DataContainer) -> list[DocumentType] | DocumentType | None:
        """
        Loads documents from the container based on the available generic keys.

        Args:
            container (DataContainer): The container with document data.

        Returns:
            list[DocumentType] | None: A list of document lists or None if no documents exist.
        """
        if self.attributes.generic_keys and isinstance(self.attributes.generic_keys, list):
            documents = []
            for key in self.attributes.generic_keys:
                docs = self._get_generic_data(container, key)
                documents.append(docs)
            return documents
        elif self.attributes.generic_keys and isinstance(self.attributes.generic_keys, str):
            return self._get_generic_data(container, self.attributes.generic_keys)
        else:
            try:
                documents = [text.content for text in container.texts]
                return documents
            except TypeError:
                return None

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Executes the embedding generation process using data from the container.

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

        nodes = self.generate_embeddings(documents)
        self._set_generic_data(container, nodes)
        self.logger.debug(f"Saved {self.instance_name} nodes as generic data")
        return container
