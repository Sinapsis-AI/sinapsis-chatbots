# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Any

import gradio as gr
from sinapsis.webapp.chatbot_base import BaseChatbot, ChatbotConfig, ChatInterfaceComponents, ChatKeys
from sinapsis_core.cli.run_agent_from_config import generic_agent_builder
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP, SINAPSIS_CACHE_DIR

CONFIG_FILE = (
    AGENT_CONFIG_PATH or "packages/sinapsis_mem0/src/sinapsis_mem0/configs/managed/openai_simple_chat_with_mem0.yaml"
)
DELETE_MEMORIES_CONFIG = "packages/sinapsis_mem0/src/sinapsis_mem0/configs/managed/delete_memories.yaml"


class MemoryChatKeys(ChatKeys):
    """Extends `ChatKeys` to reference memory-related templates.

    This class serves as a centralized place to manage key names for different types of data
    that may be used in chat interactions.
    """

    memo_search_template: str = "Mem0Search"
    memo_add_template: str = "Mem0Add"
    memo_delete_template: str = "Mem0Delete"


class ChatbotWithMem0(BaseChatbot):
    """A base class for integrating Sinapsis agents with a Gradio-based chatbot UI.

    Supports multimodal user inputs (text, audio, images), optional memory features, and
    optional user authentication backed by PostgreSQL.
    """

    def __init__(self, config_file: str, config: ChatbotConfig | dict[str, Any] = ChatbotConfig()) -> None:
        super().__init__(config_file, config)
        self._verify_templates()
        self.deletion_agent = generic_agent_builder(DELETE_MEMORIES_CONFIG)
        self._setup_memory_templates()

    def _verify_templates(self) -> None:
        """Ensures that required memory templates are present in the agent configuration.

        Raises:
            ValueError: If any of the required memory templates are missing.
        """
        required_templates = [
            MemoryChatKeys.memo_search_template,
            MemoryChatKeys.memo_add_template,
        ]
        missing = [t for t in required_templates if t not in self.agent.topological_sort]

        if missing:
            raise ValueError(f"Missing required memory templates in agent: {missing}")

    def _get_template_attributes(self, template_name: str) -> dict[str, Any]:
        """Retrieves the metadata attributes for a specific template used by the agent.

        Args:
            template_name (str): The name of the agent template.

        Returns:
            dict[str, Any]: Metadata attributes of the specified template.
        """
        return self.agent.topological_sort[template_name].metadata.attributes

    def _setup_memory_templates(self) -> None:
        """Initializes memory-related configuration from agent templates and sets up deletion support."""
        search_attrs = self._get_template_attributes(MemoryChatKeys.memo_search_template)
        add_attrs = self._get_template_attributes(MemoryChatKeys.memo_add_template)
        self.use_managed = search_attrs["use_managed"]
        self.version = search_attrs["search_kwargs"].get("version", "v1")
        self.search_kwargs = deepcopy(search_attrs["search_kwargs"])
        self.add_kwargs = deepcopy(add_attrs["add_kwargs"])

    def clear_user_memories(self, user_id: str, session_id: str | None = None) -> None:
        """Core memory deletion logic that accepts explicit IDs.

        Args:
            user_id (str): The user identifier whose memories should be deleted
            session_id (str): Optional specific session to target
        """
        delete_kwargs = {"user_id": user_id}
        if session_id and self.use_managed:
            delete_kwargs["run_id"] = session_id

        self.deletion_agent.update_template_attribute(
            MemoryChatKeys.memo_delete_template, "delete_kwargs", delete_kwargs
        )
        self.deletion_agent()

    def handle_clear_memories(self, session_state: str, user_id: str) -> None:
        """Triggers deletion of all memory records associated with a specific user.

        Args:
            session_state (str): Session ID UUID
            user_id (str): The user ID whose memories should be deleted. If None, uses `request.username`.
        """
        self.clear_user_memories(user_id, session_state)
        gr.Info("Your memories have been deleted successfully!")

    def _add_buttons(self, chat_components: ChatInterfaceComponents):
        """Extends the parent method by adding one additional button to handle memories deletion.

        Args:
            chat_components (ChatInterfaceComponents): Container with components needed
                to enable/disable input based on chatbot state.
        """
        super()._add_buttons(chat_components)
        delete_memories_btn = gr.Button("Delete my memories")
        delete_memories_btn.click(self.handle_clear_memories, inputs=[chat_components.session_state], show_api=False)

    def _update_user_memories(self, user_id: str, session_id: str | None = None) -> None:
        """Updates the agent memory-related template configurations with the current user and session.

        Args:
            user_id (str): Unique user identifier for filtering and associating memory records.
            session_id (str | None, optional): Optional session ID to scope memory to a particular interaction session.
        """

        mem_kwargs = {"user_id": user_id}
        if self.use_managed and session_id:
            mem_kwargs["run_id"] = session_id

        if self.version == "v2":
            self.search_kwargs["filters"] = mem_kwargs
        else:
            self.search_kwargs.update(mem_kwargs)

        self.add_kwargs.update(mem_kwargs)
        self.agent.update_template_attribute(MemoryChatKeys.memo_search_template, "search_kwargs", self.search_kwargs)
        self.agent.update_template_attribute(MemoryChatKeys.memo_add_template, "add_kwargs", self.add_kwargs)

    def generate_user_response(
        self, message: dict[str, Any], user_id: str, session_id: str | None = None
    ) -> dict[str, Any]:
        """Generates a chatbot response for the given user message and ID.

        Combines packet generation and agent execution to transform user input
        into a formatted response while maintaining user association.

        Args:
            message: Dictionary containing user input (text/files/audio)
            user_id: Unique identifier for user/conversation tracking
            session_id (str | None): Session ID UUID

        Returns:
            Dictionary containing response elements (text/content/files)
        """
        self._update_user_memories(user_id, session_id)
        return super().generate_user_response(message, user_id, session_id)


if __name__ == "__main__":
    sinapsis_chatbot = ChatbotWithMem0(CONFIG_FILE)
    sinapsis_chatbot.launch(share=GRADIO_SHARE_APP, allowed_paths=[SINAPSIS_CACHE_DIR])
