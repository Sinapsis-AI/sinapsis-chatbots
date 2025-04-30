# -*- coding: utf-8 -*-

from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_llama_index.helpers.llama_index_pg_retriever import connect_to_table


class LLaMAIndexInsertNodes(Template):
    """
    A class for inserting embeddings (nodes) into a PostgreSQL vector database using
    the LlamaIndex `PGVectorStore` to store vectorized data.
    """

    class AttributesBaseModel(TemplateAttributes):
        """
        A model for the specific attributes required to insert nodes into the vector database.

        Attributes:
            user (str): The username for connecting to the PostgreSQL database.
            password (str): The password for the database user.
            port (int): The port to connect to the PostgreSQL database.
            host (str): The host where the PostgreSQL database is running.
            db_name (str): The name of the database to connect to or create.
            table_name (str): The name of the table where embeddings are stored.
            embedding_dimension (int): The dimensionality of the embedding vectors. Default is 384.
            generic_key (str): The key used to retrieve data from the container for embedding.
        """

        user: str
        password: str
        port: int
        host: str
        db_name: str
        table_name: str
        embedding_dimension: int = 384
        generic_key: str

    UIProperties = UIPropertiesMetadata(category="LlamaIndex", output_type=OutputTypes.MULTIMODAL)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """
        Initializes the `PostgresInsertNodes` class by setting up the connection to the
        PostgreSQL vector store.

        Args:
            attributes (TemplateAttributeType): A dictionary of attributes used for configuring
                the connection to the PostgreSQL database and vector store (db_name, host,
                password, port, user, table_name, embedding_dimension, generic_field_key).
        """
        super().__init__(attributes)
        self.table = self._connect_to_table()
        self.logger.debug(f"Connected to table '{self.attributes.table_name}'")

    def _connect_to_table(self) -> PGVectorStore:
        """
        Creates and connects to a PostgreSQL vector table using LlamaIndex's `PGVectorStore`.

        This method initializes a `PGVectorStore` instance with the given database connection
        details and returns it to interact with the vector table in the database.

        Returns:
            PGVectorStore: An instance of the `PGVectorStore` that allows interacting with the
                PostgreSQL vector table.
        """
        vector_store = connect_to_table(
            db_name=self.attributes.db_name,
            host=self.attributes.host,
            password=self.attributes.password,
            port=self.attributes.port,
            user=self.attributes.user,
            table_name=self.attributes.table_name,
            dimension=self.attributes.embedding_dimension,
        )
        return vector_store

    def insert_embedding(self, nodes: list[TextNode]) -> None:
        """
        Inserts embeddings (from `TextNode` objects) into the PostgreSQL vector table.

        This method takes a list of `TextNode` objects, which represent the text data
        to be embedded into vectors, and inserts these embeddings into the configured
        vector store table.

        Args:
            nodes (list[TextNode]): A list of `TextNode` objects from the LlamaIndex library
                that contain text to be embedded and inserted into the database.
        """
        self.table.add(nodes)

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Executes the process of inserting embeddings from a `DataContainer` into the database.

        This method retrieves the necessary data from the provided `DataContainer` using
        the `generic_field_key`, converts it into `TextNode` objects, and inserts these
        embeddings into the vector store.

        Args:
            container (DataContainer): A container holding data (e.g., text data) to be
                embedded and inserted into the PostgreSQL database.

        Returns:
            DataContainer: The same `DataContainer` passed into the method, typically with
                additional information or modifications made during the execution process.
        """
        nodes: list[TextNode] = self._get_generic_data(container, self.attributes.generic_key)
        self.insert_embedding(nodes)

        return container
