# -*- coding: utf-8 -*-
from sinapsis.webapp.chatbot_base import BaseChatbot, ChatbotConfig
from sinapsis_core.utils.env_var_keys import AGENT_CONFIG_PATH, GRADIO_SHARE_APP, SINAPSIS_CACHE_DIR

CONFIG_FILE = AGENT_CONFIG_PATH or "packages/sinapsis_llama_cpp/src/sinapsis_llama_cpp/configs/llama_simple_chat.yaml"
if __name__ == "__main__":
    config = ChatbotConfig(
        app_title="Sinapsis LLaMA Chatbot",
        examples=[
            "Explain quantum computing in simple terms",
            "Write a short poem about artificial intelligence",
            "How would I summarize a 10-page PDF about climate change?",
        ],
    )
    sinapsis_chatbot = BaseChatbot(CONFIG_FILE, config)
    sinapsis_chatbot.launch(share=GRADIO_SHARE_APP, allowed_paths=[SINAPSIS_CACHE_DIR])
