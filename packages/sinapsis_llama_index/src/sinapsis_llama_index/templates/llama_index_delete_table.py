from pydantic import Field
from sinapsis_chatbots_base.helpers.tags import Tags
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    UIPropertiesMetadata,
)

from sinapsis_llama_index.helpers.pgvector_helpers import drop_table
from sinapsis_llama_index.helpers.schemas import VectorStoreDBConfig


class LLaMAIndexDeleteTable(Template):
    """A template that permanently drops (deletes) a specific `PGVectorStore` table from PostgreSQL.

    This is a destructive, irreversible operation that removes the table schema
    and all data it contains.

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: LLaMAIndexDeleteTable
      class_name: LLaMAIndexDeleteTable
      template_input: InputTemplate
      attributes:
        db_config:
          user: my_user
          password: my_password
          port: 5432
          host: localhost
          db_name: my_db
          table_name: my_embeddings
    """

    class AttributesBaseModel(TemplateAttributes):
        """Attributes for the LLaMAIndexDeleteTable template.

        Attributes:
            db_config (VectorStoreDBConfig): Configuration for connecting to the PostgreSQL database,
                handling credentials via environment variables if needed.
        """

        db_config: VectorStoreDBConfig = Field(default_factory=VectorStoreDBConfig)

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
        """Executes the table dropping (deletion) operation.

        Connects to the PostgreSQL database and issues a command to
        permanently drop the table specified in `db_config.table_name`.

        Args:
            container (DataContainer): The input DataContainer.

        Returns:
            DataContainer: The original, unmodified DataContainer.
        """
        drop_table(**self.attributes.db_config.model_dump())
        self.logger.info(f"Successfully deleted table '{self.attributes.db_config.table_name}'.")

        return container
