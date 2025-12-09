# -*- coding: utf-8 -*-
from typing import Any

import gradio as gr
from sinapsis.webapp.chatbot_base import BaseChatbot, ChatbotConfig, generic_agent_builder
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP, SINAPSIS_CACHE_DIR
from sinapsis_core.utils.logging_utils import sinapsis_logger
from sinapsis_llama_index.helpers.rag_env_vars import FEED_DB_DEFAULT_PATH, FEED_DB_FROM_PDF_PATH

CONFIG_FILE = (
    AGENT_CONFIG_PATH
    or "packages/sinapsis_llama_index/src/sinapsis_llama_index/configs/default/llama_cpp_rag_chat.yaml"
)

FEED_DB_CONFIG_FROM_PDF = (
    FEED_DB_FROM_PDF_PATH or "packages/sinapsis_llama_index/src/sinapsis_llama_index/configs/default/feed_db_pdf.yaml"
)
FEED_DB_DEFAULT_CONFIG = (
    FEED_DB_DEFAULT_PATH or "packages/sinapsis_llama_index/src/sinapsis_llama_index/configs/default/feed_db_git.yaml"
)

TABLE_DELETE_CONFIG = "packages/sinapsis_llama_index/src/sinapsis_llama_index/configs/delete_table.yaml"


def clear_database() -> gr.Markdown:
    """Clears the vector table used by the retriever, based on the main agent configuration.

    Returns:
        gr.Markdown: A markdown message indicating completion of the operation.
    """
    agent = generic_agent_builder(TABLE_DELETE_CONFIG)
    agent()
    return gr.Markdown("### Finished clearing context")


class RAGChatbot(BaseChatbot):
    """A chatbot class that integrates Retrieval-Augmented Generation (RAG) capabilities.

    This class extends the `BaseChatbot` and includes methods for uploading documents
    to a RAG system.
    """

    def __init__(self, config_file: str, config: ChatbotConfig | dict[str, Any] = ChatbotConfig()) -> None:
        super().__init__(config_file, config)
        self.chatbot_height = "60vh"

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

    def _add_rag_components(self) -> None:
        """Adds RAG (Retrieval-Augmented Generation) components to the interface."""
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

    def _inject_header_components(self) -> None:
        """Extends the parent's method by adding RAG-specific buttons."""
        super()._inject_header_components()
        self._add_rag_components()


if __name__ == "__main__":
    config = ChatbotConfig(
        app_title="Sinapsis RAG chatbot",
        examples=[
            "What does the document say about data privacy policies?",
            "Summarize the section related to financial projections.",
            "What does the report say about user feedback and surveys?",
        ],
    )
    sinapsis_chatbot = RAGChatbot(CONFIG_FILE, config)
    sinapsis_chatbot.launch(share=GRADIO_SHARE_APP, allowed_paths=[SINAPSIS_CACHE_DIR])
