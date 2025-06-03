# -*- coding: utf-8 -*-
from sinapsis.webapp.chatbot_base import BaseChatbot
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP, SINAPSIS_CACHE_DIR

CONFIG_FILE = (
    AGENT_CONFIG_PATH or "packages/sinapsis_anthropic/src/sinapsis_anthropic/configs/anthropic_simple_chat.yaml"
)
if __name__ == "__main__":
    sinapsis_chatbot = BaseChatbot(CONFIG_FILE)  # , "Sinapsis Claude Chatbot")
    sinapsis_chatbot.launch(share=GRADIO_SHARE_APP, allowed_paths=[SINAPSIS_CACHE_DIR])
