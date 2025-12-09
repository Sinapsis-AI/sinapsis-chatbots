# -*- coding: utf-8 -*-
from enum import Enum


class Tags(Enum):
    """Tags for LLM templates."""

    ANTHROPIC = "anthropic"
    CHATBOTS = "chatbots"
    CODE = "code"
    CONVERSATIONAL = "conversational"
    CONTEXT = "context"
    DATABASE = "database"
    DOCUMENTS = "documents"
    EMBEDDINGS = "embeddings"
    HUGGINGFACE = "huggingface"
    IMAGE_TO_TEXT = "image_to_text"
    LLAMA = "llama"
    LLAMAINDEX = "llamaindex"
    LLM = "large_language_models"
    MULTIMODAL = "multimodal"
    POSTGRESQL = "postgresql"
    RETRIEVAL = "retrieval"
    RETRIEVAL_AG = "retrieval_augmented_generation"
    TEXT = "text"
    TEXT_COMPLETION = "text_completion"
    TEXT_TO_TEXT = "text_to_text"
    QUERY = "query"
    QUERY_CONTEXTUALIZATION = "query_contextualization"
    VECTORS = "vectors"
    MCP = "mcp"
    STREAMING = "streaming"
