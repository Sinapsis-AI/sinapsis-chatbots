# -*- coding: utf-8 -*-
from typing import Any

import gradio as gr
from sinapsis.webapp.chatbot_base import BaseChatbot, ChatbotConfig, ChatKeys
from sinapsis_core.cli.run_agent_from_config import generic_agent_builder
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP, SINAPSIS_CACHE_DIR

CONFIG_FILE = AGENT_CONFIG_PATH or "webapps/configs/llama_simple_chat_with_history.yaml"
DELETE_HISTORY_CONFIG = "webapps/configs/remove_chat_history.yaml"


class HistoryChatKeys(ChatKeys):
    """Extends `ChatKeys` to include references to templates used for chat history operations.

    This class centralizes template key names used for fetching, saving, and deleting
    chat history entries within the agent.
    """

    chat_fetcher_template: str = "ChatHistoryFetcher"
    chat_saver_template: str = "ChatHistorySaver"
    chat_remover_template: str = "ChatHistoryRemover"


class ChatbotWithHistory(BaseChatbot):
    """A chatbot class that supports persistent chat history using agent templates.

    This class enables automatic management of user-specific conversation history,
    including fetching previous interactions, saving new turns, and deleting current sessions.
    """

    def __init__(self, config_file: str, config: ChatbotConfig | dict[str, Any] = ChatbotConfig()) -> None:
        super().__init__(config_file, config)
        self._verify_templates()
        self.deletion_agent = generic_agent_builder(DELETE_HISTORY_CONFIG)

    def _verify_templates(self) -> None:
        """Ensures that required chat history templates are present in the agent configuration.

        Raises:
            ValueError: If any of the required memory templates are missing.
        """
        required_templates = [
            HistoryChatKeys.chat_fetcher_template,
            HistoryChatKeys.chat_saver_template,
        ]
        missing = [t for t in required_templates if t not in self.agent.topological_sort]

        if missing:
            raise ValueError(f"Missing required templates in agent: {missing}")

    def clear_user_conversation(self, user_id: str, session_id: str | None = None) -> None:
        """Clears stored conversation data for the specified user/session.

        Args:
            user_id (str): Unique user identifier.
            session_id (str | None, optional):  Optional session ID to scope clearing. Defaults to None.

        Returns:
            Any: Result of the clear operation (to be defined by subclasses).
        """
        filters = {"user_id": user_id}
        if session_id:
            filters["session_id"] = session_id

        self.deletion_agent.update_template_attribute(HistoryChatKeys.chat_remover_template, "filters", filters)
        self.deletion_agent()
        gr.Info("Conversation removed!")


if __name__ == "__main__":
    sinapsis_chatbot = ChatbotWithHistory(CONFIG_FILE)
    sinapsis_chatbot.launch(share=GRADIO_SHARE_APP, allowed_paths=[SINAPSIS_CACHE_DIR])
