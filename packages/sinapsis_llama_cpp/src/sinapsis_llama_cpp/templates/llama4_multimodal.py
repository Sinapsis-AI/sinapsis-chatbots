# -*- coding: utf-8 -*-
from PIL import Image
from sinapsis_chatbots_base.helpers.tags import Tags
from sinapsis_core.data_containers.data_packet import DataContainer

from sinapsis_llama_cpp.templates.llama4_text_to_text import LLama4TextToText, LLamaMultiModalKeys

LLama4MultiModalUIProperties = LLama4TextToText.UIProperties
LLama4MultiModalUIProperties.tags.extend([Tags.MULTIMODAL, Tags.IMAGE_TO_TEXT])


class LLama4MultiModal(LLama4TextToText):
    """Template for multi modal chat processing using the LLama 4 model.

    This template provides support for text-to-text and image-to-text
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
        init_args:
          llm_model_name: "meta-llama/Llama-4-Scout-17B-16E-Instruct"
          device_map: auto
          torch_dtype: auto
          max_memory:
            0: "8GiB"
            cpu: "10GiB"
        completion_args:
          max_new_tokens: 256
        role: assistant
        system_prompt: You are an AI and Python expert, and you should reason in every response you provide
        chat_format: chatml
        pattern: null
        keep_before: true
    """

    UIProperties = LLama4MultiModalUIProperties

    @staticmethod
    def extract_additional_content(container: DataContainer) -> list:
        """Extracts images from the container.

        Args:
            container (DataContainer): The container holding potential extra data.

        Returns:
            list: A list of dictionaries representing the additional image items.
        """
        content = []
        for image in container.images:
            img = Image.fromarray(image.content)
            content.append({LLamaMultiModalKeys.type: LLamaMultiModalKeys.image, LLamaMultiModalKeys.image: img})
        return content
