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
    AGENT_CONFIG_PATH or "packages/sinapsis_llama_index/src/sinapsis_llama_index/configs/llama_index_rag_chat.yaml"
)

FEED_DB_CONFIG_FROM_PDF = (
    FEED_DB_FROM_PDF_PATH or "packages/sinapsis_llama_index/src/sinapsis_llama_index/configs/feed_db_pdf.yaml"
)
FEED_DB_DEFAULT_CONFIG = (
    FEED_DB_DEFAULT_PATH or "packages/sinapsis_llama_index/src/sinapsis_llama_index/configs/feed_db_git.yaml"
)


def clear_database():
    """Method to clear a database using the main configuration file from the agent"""
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
    """
    A chatbot class that integrates Retrieval-Augmented Generation (RAG) capabilities.

    This class extends the `SimpleLLMChatbot` and includes methods for uploading documents
    to a RAG system.
    """

    @staticmethod
    def upload_default_vals() -> gr.Markdown:
        """
        Uploads default values (Wikipedia data) to the RAG system.

        This method uses a generic agent builder to configure the RAG system with default
        documents (Wikipedia in this case) and then passes a `DataContainer` to initialize
        the system.
        """

        agent = generic_agent_builder(FEED_DB_DEFAULT_CONFIG)
        agent()
        sinapsis_logger.debug("Finished uploading default documents")

        return gr.Markdown("#### Finished uploading default documents")

    @staticmethod
    def upload_doc(file_path: str) -> gr.Markdown:
        """
        Uploads a PDF document to feed the RAG system.

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
        "Updates status_text to indicate that the model is ready."
        return gr.Markdown("#### Uploading documents...")

    def __call__(self) -> gr.Blocks:
        """
        Creates the Gradio interface for the RAG chatbot.

        This method sets up a Gradio interface with file upload functionality and a button
        for uploading default documents. It also integrates the chatbot's core functionality
        into the interface using Gradio components.

        Returns:
            gr.Interface: The Gradio interface for interacting with the chatbot.
        """
        with gr.Blocks(css=css_header()) as chatbot_interface:
            add_logo_and_title(self.app_title)
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
            self.app_interface()

        return chatbot_interface


if __name__ == "__main__":
    sinapsis_chatbot = RAGChatbot(CONFIG_FILE, app_title="Sinapsis LLaMA-Index RAG chatbot")
    sinapsis_chatbot().launch(share=GRADIO_SHARE_APP)
