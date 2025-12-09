# -*- coding: utf-8 -*-
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
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

from sinapsis_llama_index.helpers.pgvector_helpers import connect_to_table, ensure_postgres_db_exists
from sinapsis_llama_index.helpers.schemas import VectorStoreDBConfig


class LLaMAIndexInsertNodes(Template):
    """Template for inserting embeddings into a PostgreSQL vector database using `PGVectorStore.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: LLaMAIndexInsertNodes
      class_name: LLaMAIndexInsertNodes
      template_input: InputTemplate
      attributes:
        db_config:
          user: my_user
          password: my_password
          port: 5432
          host: localhost
          db_name: my_db
          table_name: my_embeddings
        input_nodes_key: InputTemplate
    """

    class AttributesBaseModel(TemplateAttributes):
        """Attributes for the LLaMAIndexInsertNodes template.

        Attributes:
            db_config (VectorStoreDBConfig): Configuration for connecting to the PostgreSQL database,
                handling credentials via environment variables if needed.
            input_nodes_key (str): Key used to access the nodes to insert into the store.
        """

        db_config: VectorStoreDBConfig = Field(default_factory=VectorStoreDBConfig)
        input_nodes_key: str

    UIProperties = UIPropertiesMetadata(
        category="LlamaIndex",
        output_type=OutputTypes.MULTIMODAL,
        tags=[
            Tags.DATABASE,
            Tags.EMBEDDINGS,
            Tags.LLAMAINDEX,
            Tags.POSTGRESQL,
            Tags.VECTORS,
        ],
    )

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """Initializes the `PostgresInsertNodes` class by setting up the connection to the PostgreSQL vector store.

        Args:
            attributes (TemplateAttributeType): A dictionary of attributes used for configuring
                the connection to the PostgreSQL database and vector store (db_name, host,
                password, port, user, table_name, embedding_dimension, generic_field_key).
        """
        super().__init__(attributes)
        self.vector_store: PGVectorStore | None = None

    def init_vector_store(self, embedding_dimension: int) -> PGVectorStore:
        """Initialize the vector store for storing and retrieving embeddings.

        This method connects to the database and initializes the vector store. It can be overridden by subclasses
        to provide custom vector store initialization logic.

        Returns:
                VectorStore: The initialized vector store.
        """
        ensure_postgres_db_exists(**self.attributes.db_config.model_dump(exclude={"table_name"}))
        vector_store = connect_to_table(dimension=embedding_dimension, **self.attributes.db_config.model_dump())
        return vector_store

    def insert_embedding(self, nodes: list[TextNode]) -> None:
        """Inserts embeddings (from `TextNode` objects) into the PostgreSQL vector table.

        This method takes a list of `TextNode` objects, which represent the text data
        to be embedded into vectors, and inserts these embeddings into the configured
        vector store table.

        Args:
            nodes (list[TextNode]): A list of `TextNode` objects from the LlamaIndex library
                that contain text to be embedded and inserted into the database.
        """
        if self.vector_store is None:
            self.vector_store = self.init_vector_store(len(nodes[0].get_embedding()))
        self.vector_store.add(nodes)

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the process of inserting embeddings from a `DataContainer` into the database.

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
        nodes: list[TextNode] = self._get_generic_data(container, self.attributes.input_nodes_key)
        if nodes:
            self.insert_embedding(nodes)

        return container
