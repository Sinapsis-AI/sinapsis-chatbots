# -*- coding: utf-8 -*-
from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class LLMChatKeys:
    """
    A class to hold constants for the keys used in chat interactions with an LLM (Large Language Model).

    These keys represent the standard fields in a chat interaction, such as the role of the participant
    and the content of the message. They are typically used when constructing input messages or
    processing the output from an LLM.
    """

    role: Literal["role"] = "role"
    content: Literal["content"] = "content"
    choices: Literal["choices"] = "choices"
    message: Literal["message"] = "message"
    llm_responses: Literal["llm_responses"] = "llm_responses"
    system_value: Literal["system"] = "system"
    user_value: Literal["user"] = "user"
    assistant_value: Literal["assistant"] = "assistant"
