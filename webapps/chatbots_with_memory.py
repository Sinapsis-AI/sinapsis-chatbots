# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Any

from sinapsis.webapp.chatbot_base import BaseChatbot, ChatbotConfig, ChatKeys
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP, SINAPSIS_CACHE_DIR
from sinapsis_core.utils.logging_utils import sinapsis_logger


class MemoryChatKeys(ChatKeys):
    """Defines key names used for referencing various chat-related data types.

    This class serves as a centralized place to manage key names for different types of data
    that may be used in chat interactions. These keys are typically used to map data in structured formats.
    """

    memo_search_template: str = "Mem0Search"
    memo_add_template: str = "Mem0Add"


class ChatbotWithMem0(BaseChatbot):
    """A base class for integrating Sinapsis agents with a Gradio-based chatbot UI.

    Supports multimodal user inputs (text, audio, images), optional memory features, and
    optional user authentication backed by PostgreSQL.
    """

    def __init__(self, config_file: str, config: ChatbotConfig | dict[str, Any] = ChatbotConfig()) -> None:
        super().__init__(config_file, config)
        self.username = ""
        self.enable_memories = self._validate_memory_enablement(self.config.enable_memories)
        if self.enable_memories:
            self._setup_memory_templates()
        self._setup_working_directory()
        # self.auth_users = self._load_authenticated_users()

    def _validate_memory_enablement(self, requested_enable: bool) -> bool:
        """Validates memory feature enablement by checking for required templates.

        Args:
            requested_enable (bool): Whether memory features are requested.

        Returns:
            bool: True if memory is both requested and supported by the agent, False otherwise.
        """
        if not requested_enable:
            sinapsis_logger.debug("Memory features disabled.")

        templates_exist = all(
            template in self.agent.topological_sort
            for template in [MemoryChatKeys.memo_search_template, MemoryChatKeys.memo_add_template]
        )
        actual_enable = requested_enable and templates_exist

        if requested_enable and not templates_exist:
            sinapsis_logger.warning(
                "Memory features requested but required templates (Mem0Search/Mem0Add) not found. "
                "Memories will be disabled."
            )
        else:
            sinapsis_logger.debug("Memory features enabled.")

        return actual_enable

    def _get_template_attributes(self, template_name: str) -> dict[str, Any]:
        """Retrieves metadata attributes for a specific agent template.

        Args:
            template_name (str): The name of the agent template.

        Raises:
            ValueError: If the template is not found in the agent.

        Returns:
            dict[str, Any]: Metadata attributes of the specified template.
        """
        if template_name not in self.agent.topological_sort:
            raise ValueError(f"Template {template_name} not found in agent")
        return self.agent.topological_sort[template_name].metadata.attributes

    def _setup_memory_templates(self) -> None:
        """Initializes memory-related settings by extracting required parameters from the agent templates."""
        search_attrs = self._get_template_attributes(MemoryChatKeys.memo_search_template)
        add_attrs = self._get_template_attributes(MemoryChatKeys.memo_add_template)
        self.use_managed = search_attrs["use_managed"]
        self.version = search_attrs["search_kwargs"].get("version", "v1")
        self.search_kwargs = deepcopy(search_attrs["search_kwargs"])
        self.add_kwargs = deepcopy(add_attrs["add_kwargs"])

    def _update_user_memories(self, username: str) -> bool:
        """Updates agent memory parameters based on the current user session.

        Args:
            username (str): The username to bind memory tracking to.

        Returns:
            bool: True if the user was updated, False if already current.
        """
        if username == self.username:
            return False

        self.username = username
        if self.version == "v2":
            self.search_kwargs["filters"] = {"user_id": self.username}
        else:
            self.search_kwargs["user_id"] = self.username
        self.add_kwargs["user_id"] = self.username
        self.agent.update_template_attribute(MemoryChatKeys.memo_search_template, "search_kwargs", self.search_kwargs)
        self.agent.update_template_attribute(MemoryChatKeys.memo_add_template, "add_kwargs", self.add_kwargs)

        return True

    def process_msg(
        self, message: dict[str, Any], history: list[dict[str, Any]], conv_id: str, container: DataContainer
    ) -> dict[str, Any]:
        """Processes a new user message, optionally updating memory state.

        Args:
            message (dict[str, Any]): Current message from the user.
            history (list[dict[str, Any]]): Current session history.
            conv_id (str): Conversation id value of the chat
            container (DataContainer): incoming data container
        Returns:
            dict[str, Any]: Output to be shown in the chatbot.
        """
        _ = container

        username = conv_id  # request.username if request.username else "default_user"

        if self.enable_memories:
            update = self._update_user_memories(username)
            if update:
                sinapsis_logger.info(f"Updated user to: {username}")
        response = super().process_msg(message, history, conv_id)

        return response


CONFIG_FILE = (
    AGENT_CONFIG_PATH or "packages/sinapsis_mem0/src/sinapsis_mem0/configs/managed/openai_simple_chat_with_mem0.yaml"
)
if __name__ == "__main__":
    sinapsis_chatbot = BaseChatbot(CONFIG_FILE)  # , "Sinapsis Claude Chatbot")
    sinapsis_chatbot.launch(share=GRADIO_SHARE_APP, allowed_paths=[SINAPSIS_CACHE_DIR])
