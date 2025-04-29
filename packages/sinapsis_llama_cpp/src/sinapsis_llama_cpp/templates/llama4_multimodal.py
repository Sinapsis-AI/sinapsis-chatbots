# -*- coding: utf-8 -*-

from PIL import Image
from sinapsis_core.data_containers.data_packet import DataContainer

from sinapsis_llama_cpp.templates.llama_4_text_to_text import LLama4TextToText, LLamaMultiModalKeys


class LLama4MultiModal(LLama4TextToText):
    """Template for multi modal chat processing using the
    LLama 4 model This template provides support for text-to-text and image-to-text
    conversational chatbots and all the LLama4 models that have been released

    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: LLama4
      class_name: LLama4
      template_input: InputTemplate
      attributes:
        llm_model_name: "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        n_ctx: 9000
        role: assistant
        system_prompt: You are an AI and Python expert and you should reason in every response you provide
        chat_format: chatml
        context_max_len: 6
        pattern: null
        keep_before: true
        device_map: auto
        max_new_tokens: 256
        hf_access_token: null
        extra_args:
          torch_dtype: auto
          max_memory:
            0: "8GiB"
            cpu: "10GiB"

    """

    @staticmethod
    def process_packets(messages: list, container: DataContainer) -> list:
        for image in container.images:
            img = Image.fromarray(image.content)
            messages.append({LLamaMultiModalKeys.type: LLamaMultiModalKeys.image, LLamaMultiModalKeys.image: img})
        return messages
