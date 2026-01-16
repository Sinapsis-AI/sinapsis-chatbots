from pydantic import Field
from sinapsis_chatbots_base.helpers.tags import Tags
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    UIPropertiesMetadata,
)

from sinapsis_llama_index.helpers.pgvector_helpers import clear_table
from sinapsis_llama_index.helpers.schemas import VectorStoreDBConfig


class LLaMAIndexClearTable(Template):
    """A template that connects to a specified PostgreSQL `PGVectorStore` table and deletes all its contents.

    This is a destructive, irreversible operation that removes the data inside the
    vector store.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: LLaMAIndexClearTable
      class_name: LLaMAIndexClearTable
      template_input: InputTemplate
      attributes:
        db_config:
          user: my_user
          password: my_password
          port: 5432
          host: localhost
          db_name: my_db
          table_name: my_embeddings
        embedding_dimension: 384
    """

    class AttributesBaseModel(TemplateAttributes):
        """Attributes for the LLaMAIndexClearTable template.

        Attributes:
            db_config (VectorStoreDBConfig): Configuration for connecting to the PostgreSQL database,
                handling credentials via environment variables if needed.
            embedding_dimension (int): The dimension of the vector table. Required to properly interface with the table.
        """

        db_config: VectorStoreDBConfig = Field(default_factory=VectorStoreDBConfig)
        embedding_dimension: int = 384

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

    def execute(self, container: DataContainer) -> DataContainer:
        """Executes the table clearing operation.

        Connects to the PostgreSQL vector store table specified in `db_config`
        and deletes all rows, effectively clearing its contents.

        Args:
            container (DataContainer): The input DataContainer.

        Returns:
            DataContainer: The original, unmodified DataContainer.
        """
        clear_table(dimension=self.attributes.embedding_dimension, **self.attributes.db_config.model_dump())
        self.logger.info(f"Successfully cleared table '{self.attributes.db_config.table_name}'.")

        return container
