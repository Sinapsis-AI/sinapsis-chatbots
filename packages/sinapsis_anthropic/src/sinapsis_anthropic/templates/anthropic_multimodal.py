# -*- coding: utf-8 -*-
from sinapsis_chatbots_base.helpers.llm_keys import LLMChatKeys
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_generic_data_tools.helpers.encode_img_base64 import convert_image_ndarray_to_base64

from sinapsis_anthropic.helpers.anthropic_keys import AnthropicKeys
from sinapsis_anthropic.templates.anthropic_text_generation import AnthropicTextGeneration


class AnthropicMultiModal(AnthropicTextGeneration):
    """Template for multi-modal chat processing using Anthropic's Claude models.

    This template provides support for text-to-text and image-to-text conversational
    chatbots using Anthropic's Claude models that support multi-modal inputs. It enables
    processing of both text and image inputs to generate text responses.

    Usage example:
      agent:
        name: my_claude_agent
        templates:
          - template_name: InputTemplate
            class_name: InputTemplate
            attributes: {}
          - template_name: AnthropicMultiModal
            class_name: AnthropicMultiModal
            template_input: InputTemplate
            attributes:
                llm_model_name: claude-3-opus-20240229
                max_tokens: 4000
                temperature: 1

    The class handles conversion of image data to base64 format required by the Anthropic API,
    and properly formats multi-modal conversations for the API.

    This template extends AnthropicTextGeneration with additional capabilities for handling
    image inputs alongside text.
    """

    @staticmethod
    def process_content(container: DataContainer) -> list:
        """
        Processes images from a DataContainer and converts them to the Anthropic API format.

        This method extracts images from the provided container and converts each image
        to a base64-encoded representation formatted according to Anthropic's multimodal
        API requirements.

        Args:
            container (DataContainer): Container holding image data in the 'images' attribute
                                      where each image is an ImagePacket with ndarray content

        Returns:
            list: A list of dictionaries, each representing an image in Anthropic's API format
                 with 'type', 'source.type', 'source.media_type', and 'source.data' fields
        """
        content = []
        for image in container.images:
            img = convert_image_ndarray_to_base64(image.content)
            content.append(
                {
                    AnthropicKeys.type: AnthropicKeys.image,
                    AnthropicKeys.source: {
                        AnthropicKeys.type: AnthropicKeys.base64,
                        AnthropicKeys.media_type: f"{AnthropicKeys.image}/{AnthropicKeys.jpeg}",
                        AnthropicKeys.data: img,
                    },
                }
            )
        return content

    def generate_response(self, container: DataContainer) -> DataContainer:
        """
        Processes a list of `TextPacket` objects, generating a response for each
        text packet.

        Args:
            container (DataContainer): Container where the incoming message and possibly images
            that need to be processed.

        Returns:
            DataContainer: Updated container with the generated text responses from the LLM.
        """

        self.logger.debug("Chatbot in progress")
        responses = []
        messages = []
        for packet in container.texts:
            prompt = self.process_content(container)
            prompt.append({AnthropicKeys.type: AnthropicKeys.text, AnthropicKeys.text: packet.content})
            message = self.generate_dict_msg(LLMChatKeys.user_value, prompt)
            messages.append(message)

            response = self.infer(messages)

            self.logger.debug("End of interaction.")
            responses.append(TextPacket(content=response))

        container.texts.extend(responses)

        return container
