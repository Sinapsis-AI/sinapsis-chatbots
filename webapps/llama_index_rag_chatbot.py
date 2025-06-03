# -*- coding: utf-8 -*-
import gradio as gr
import yaml  # type:ignore[import-untyped]
from sinapsis.webapp.agent_gradio_helper import add_logo_and_title, css_header
from sinapsis.webapp.chatbot_base import BaseChatbot, generic_agent_builder
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP
from sinapsis_core.utils.logging_utils import sinapsis_logger
from sinapsis_llama_index.helpers.llama_index_pg_retriever import delete_table
from sinapsis_llama_index.helpers.rag_env_vars import FEED_DB_DEFAULT_PATH, FEED_DB_FROM_PDF_PATH

CONFIG_FILE = (
    AGENT_CONFIG_PATH or "packages/sinapsis_llama_index/src/sinapsis_llama_index/configs/unsloth_with_context.yaml"
)

FEED_DB_CONFIG_FROM_PDF = (
    FEED_DB_FROM_PDF_PATH or "packages/sinapsis_llama_index/src/sinapsis_llama_index/configs/feed_db_pdf.yaml"
)
FEED_DB_DEFAULT_CONFIG = (
    FEED_DB_DEFAULT_PATH or "packages/sinapsis_llama_index/src/sinapsis_llama_index/configs/feed_db_git_all_repos.yaml"
)


def clear_database() -> gr.Markdown:
    """Clears the vector table used by the retriever, based on the main agent configuration.

    Returns:
        gr.Markdown: A markdown message indicating completion of the operation.
    """
    with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
        config_dict: dict = yaml.safe_load(config_file)
    config_file.close()

    templates = config_dict.get("templates")
    for template in templates:
        if template.get("template_name", False) == "LLaMAIndexNodeRetriever":
            database_attrs = template.get("attributes")
            database_name = database_attrs.get("db_name")
            table_name = database_attrs.get("table_name")
            dimension = database_attrs.get("database_dimension")
            user = database_attrs.get("user")
            password = database_attrs.get("password")

            delete_table(database_name, table_name, user, password, dimension)
    return gr.Markdown("### Finished clearing context")


class RAGChatbot(BaseChatbot):
    """A chatbot class that integrates Retrieval-Augmented Generation (RAG) capabilities.

    This class extends the `BaseChatbot` and includes methods for uploading documents
    to a RAG system.
    """

    @staticmethod
    def upload_default_vals() -> gr.Markdown:
        """Uploads default values (Github repos) to the RAG system.

        This method uses a generic agent builder to configure the RAG system with default
        documents (Github repos content) and then passes a `DataContainer` to initialize
        the system.
        """
        agent = generic_agent_builder(FEED_DB_DEFAULT_CONFIG)
        agent()
        sinapsis_logger.debug("Finished uploading default documents")

        return gr.Markdown("#### Finished uploading default documents")

    @staticmethod
    def upload_doc(file_path: str) -> gr.Markdown:
        """Uploads a PDF document to feed the RAG system.

        This method updates the template attribute with the provided PDF file and
        passes it to the RAG system through the generic agent builder.

        Args:
            file_path (str): The file path to the PDF document to upload to the system.
        """
        agent = generic_agent_builder(FEED_DB_CONFIG_FROM_PDF)
        agent.update_template_attribute("LangchainPDFReader", "pypdfloader_init", {"file_path": file_path})
        agent.reset_state("LangchainPDFReader")
        agent()
        sinapsis_logger.debug("Finished uploading PDF document")
        return gr.Markdown("#### Finished uploading PDF document")

    @staticmethod
    def make_status_visible() -> gr.Markdown:
        """Displays a temporary status message indicating document upload is in progress.

        Returns:
            gr.Markdown: A status message.
        """
        return gr.Markdown("#### Uploading documents...")

    def app_interface(self) -> gr.Blocks:
        """Builds the full Gradio UI layout for the chatbot RAG application.

        Returns:
            gr.Blocks: Gradio Blocks layout for the complete application.
        """
        with gr.Blocks(css=css_header(), title=self.config.app_title) as chatbot_interface:
            add_logo_and_title(self.config.app_title)
            with gr.Row():
                upload_file_to_feed_llm = gr.UploadButton(
                    label="Upload a PDF file to feed your RAG system",
                    scale=1,
                    file_types=[".pdf"],
                )
                default_values = gr.Button(value="Upload default documents")
                clear_db = gr.Button("Clear context")
            status_msg = gr.Markdown(visible=True)
            upload_file_to_feed_llm.upload(self.make_status_visible, outputs=[status_msg]).then(
                self.upload_doc, inputs=[upload_file_to_feed_llm], outputs=[status_msg]
            )
            default_values.click(self.make_status_visible, outputs=[status_msg]).then(
                self.upload_default_vals, outputs=[status_msg]
            )
            clear_db.click(clear_database, outputs=status_msg)
            self.add_app_components()

        return chatbot_interface


if __name__ == "__main__":
    sinapsis_chatbot = RAGChatbot(CONFIG_FILE, config={"app_title": "Sinapsis RAG chatbot"})
    sinapsis_chatbot.launch(share=GRADIO_SHARE_APP)
