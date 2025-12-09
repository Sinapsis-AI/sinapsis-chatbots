# -*- coding: utf-8 -*-
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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

from sinapsis_llama_index.helpers.llama_index_pg_retriever import LLaMAIndexPGRetriever
from sinapsis_llama_index.helpers.pgvector_helpers import connect_to_table
from sinapsis_llama_index.helpers.schemas import HFEmbeddingConfig, RetrieveArgs, VectorStoreDBConfig


class LLaMAIndexNodeRetriever(Template):
    """A Template for retrieving nodes from a database using embeddings.

    It initializes the vector store and sets up the retrieval system.

    This class is designed to work with a database schema and embedding
    models to retrieve relevant nodes based on text content.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: LLaMAIndexNodeRetriever
      class_name: LLaMAIndexNodeRetriever
      template_input: InputTemplate
      attributes:
        embedding_config:
          model_name: nomic-ai/nomic-embed-text-v1.5
          trust_remote_code : True
          device: cpu
          embed_batch_size: 8
        db_config:
          user: my_user
          password: my_password
          port: 5432
          host: localhost
          db_name: my_db
          table_name: my_embeddings
        retrieve_args:
          query_mode: default
          top_k: 4
          threshold: 0.5
    """

    class AttributesBaseModel(TemplateAttributes):
        """Attributes for the LLaMAIndexNodeRetriever template.

        Attributes:
            embedding_config (HFEmbeddingConfig): Configuration for the HuggingFace embedding model,
                including model_name, device, trust_remote_code, etc.
            db_config (VectorStoreDBConfig): Configuration for connecting to the PostgreSQL database,
                handling credentials via environment variables if needed.
            retrieve_args (RetrieveArgs): Parameters controlling the retrieval process, such as
                query mode, top_k results to fetch, and the similarity threshold.
        """

        embedding_config: HFEmbeddingConfig
        db_config: VectorStoreDBConfig = Field(default_factory=VectorStoreDBConfig)
        retrieve_args: RetrieveArgs = Field(default_factory=RetrieveArgs)

    UIProperties = UIPropertiesMetadata(
        category="LlamaIndex",
        output_type=OutputTypes.MULTIMODAL,
        tags=[
            Tags.DATABASE,
            Tags.EMBEDDINGS,
            Tags.HUGGINGFACE,
            Tags.LLAMAINDEX,
            Tags.POSTGRESQL,
            Tags.RETRIEVAL,
            Tags.VECTORS,
        ],
    )

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.embedding_model, embedding_dimension = self._init_embed_model()
        self.vector_store = self.init_vector_store(embedding_dimension)
        self.retriever = self.init_retriever()

    def init_vector_store(self, embedding_dimension: int) -> PGVectorStore:
        """Initialize the vector store for storing and retrieving embeddings.

        This method connects to the database and initializes the vector store. It can be overridden by subclasses
        to provide custom vector store initialization logic.

        Returns:
                VectorStore: The initialized vector store.
        """
        vector_store = connect_to_table(dimension=embedding_dimension, **self.attributes.db_config.model_dump())
        return vector_store

    @staticmethod
    def _get_embedding_dimension(model: HuggingFaceEmbedding) -> int:
        """Determines the embedding dimension of the HuggingFace model.

        Args:
            model (HuggingFaceEmbedding): The initialized LlamaIndex HuggingFaceEmbedding wrapper.

        Returns:
            int: The determined embedding dimension
        """
        dimension = None
        if hasattr(model, "_model") and hasattr(model._model, "get_sentence_embedding_dimension"):
            dimension = model._model.get_sentence_embedding_dimension()
            if dimension:
                return dimension
        test_embedding = model.get_text_embedding("test")
        return len(test_embedding)

    def _init_embed_model(self) -> tuple[HuggingFaceEmbedding, int]:
        """Initialize the embedding model.

        This method initializes the embedding model using the HuggingFace API. It can be overridden by subclasses
        to provide custom embedding model initialization.

        Returns:
            HuggingFaceEmbedding: The initialized embedding model.
        """
        model = HuggingFaceEmbedding(**self.attributes.embedding_config.model_dump(exclude_none=True))
        dimension = self._get_embedding_dimension(model=model)
        return model, dimension

    def init_retriever(self) -> LLaMAIndexPGRetriever:
        """Initialize the vector retrieval system.

        This method initializes the vector retrieval system, which uses the vector store and the embedding model.
        It can be overridden by subclasses to provide custom synthesizer initialization logic.

        Returns:
                VectorDBRetriever: The initialized vector retrieval system.
        """
        retriever = LLaMAIndexPGRetriever(
            self.vector_store, self.embedding_model, **self.attributes.retrieve_args.model_dump()
        )
        return retriever

    def execute(self, container: DataContainer) -> DataContainer:
        """Retrieves relevant text nodes for each input using vector search.

        Args:
            container (DataContainer): Contains text packets to process

        Returns:
            DataContainer: Container with retrieved nodes stored under instance_name in each packet's generic_data
        """
        for text in container.texts:
            retrieved_nodes = self.retriever._retrieve(QueryBundle(query_str=text.content))
            context_nodes = [node.text for node in retrieved_nodes]
            text.generic_data[self.instance_name] = context_nodes
        return container
