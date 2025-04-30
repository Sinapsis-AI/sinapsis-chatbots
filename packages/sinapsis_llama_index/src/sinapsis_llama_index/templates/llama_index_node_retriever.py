# -*- coding: utf-8 -*-
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_llama_index.helpers.llama_index_pg_retriever import LLaMAIndexPGRetriever, connect_to_table


class LLaMAIndexNodeRetriever(Template):
    """A Template for retrieving nodes from a database using embeddings.
    It initializes the vector store and sets up the retrieval system.

    This class is designed to work with a database schema and embedding
    models to retrieve relevant nodes based on text content.

    """

    class AttributesBaseModel(TemplateAttributes):
        """Attributes of the template

        embedding_model_name (str): name of the embedding model
        query_mode (str): Method to search the embeddings. Options include default, sparse,
        hybrid, text, among others
        top_k (int): Number of nodes to retrieve from the search
        db_name (str): Name of the database where search is done
        table_name (str): Name of the table where search is done
        database_dimension (int): Dimension of the vector table
        trust_remote_code (bool) Flag to indicate if remote code should be trusted
        user (str): Username to connect to database
        password (str): Password to connect to database

        """

        embedding_model_name: str
        query_mode: str = "default"
        top_k: int = 2
        db_name: str
        table_name: str
        database_dimension: int
        trust_remote_code: bool = True
        user: str
        password: str

    UIProperties = UIPropertiesMetadata(category="LlamaIndex", output_type=OutputTypes.MULTIMODAL)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.vector_database = self.init_vector_store()
        self.embedding_model = HuggingFaceEmbedding(
            self.attributes.embedding_model_name, device="cuda", trust_remote_code=self.attributes.trust_remote_code
        )
        self.retriever = self.init_retriever()

    def init_vector_store(self) -> PGVectorStore:
        """
        Initialize the vector store for storing and retrieving embeddings.

        This method connects to the database and initializes the vector store. It can be overridden by subclasses
        to provide custom vector store initialization logic.

        Returns:
                VectorStore: The initialized vector store.
        """
        vector_store = connect_to_table(
            db_name=self.attributes.db_name,
            table_name=self.attributes.table_name,
            dimension=self.attributes.database_dimension,
            password=self.attributes.password,
            user=self.attributes.user,
        )
        return vector_store

    def init_retriever(self) -> LLaMAIndexPGRetriever:
        """
        Initialize the vector retrieval system.

        This method initializes the vector retrieval system, which uses the vector store and the embedding model.
        It can be overridden by subclasses to provide custom synthesizer initialization logic.

        Returns:
                VectorDBRetriever: The initialized vector retrieval system.
        """
        retriever = LLaMAIndexPGRetriever(
            self.vector_database,
            self.embedding_model,
            query_mode=self.attributes.query_mode,
            similarity_top_k=self.attributes.top_k,
        )
        return retriever

    def execute(self, container: DataContainer) -> DataContainer:
        context_nodes = []
        for text in container.texts:
            retrieved_nodes = self.retriever._retrieve(QueryBundle(query_str=text.content))

            for node in retrieved_nodes:
                context_nodes.append(node.text)

                self._set_generic_data(container, context_nodes)
        return container
