# -*- coding: utf-8 -*-
from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class AnthropicKeys:
    """
    A class to hold constants for the keys used in Anthropic message create method.
    """

    model: Literal["model"] = "model"
    max_tokens: Literal["max_tokens"] = "max_tokens"
    messages: Literal["messages"] = "messages"
    system: Literal["system"] = "system"
    temperature: Literal["temperature"] = "temperature"
    tools: Literal["tools"] = "tools"
    name: Literal["name"] = "name"
    web_search: Literal["web_search"] = "web_search"
    web_search_20250305: Literal["web_search_20250305"] = "web_search_20250305"
    type: Literal["type"] = "type"
    text: Literal["text"] = "text"
    thinking: Literal["thinking"] = "thinking"
    budget_tokens: Literal["budget_tokens"] = "budget_tokens"
    enabled: Literal["enabled"] = "enabled"
    image: Literal["image"] = "image"
    document: Literal["document"] = "document"
    source: Literal["source"] = "source"
    media_type: Literal["media_type"] = "media_type"
    data: Literal["data"] = "data"
    base64: Literal["base64"] = "base64"
    jpeg: Literal["jpeg"] = "jpeg"
    png: Literal["png"] = "png"
