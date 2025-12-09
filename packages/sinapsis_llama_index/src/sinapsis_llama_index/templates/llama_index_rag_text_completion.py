# -*- coding: utf-8 -*-
from typing import Any, cast

from llama_index.core import get_response_synthesizer
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.postgres import PGVectorStore
from pydantic import BaseModel, Field
from sinapsis_chatbots_base.helpers.postprocess_text import postprocess_text
from sinapsis_chatbots_base.helpers.tags import Tags
from sinapsis_chatbots_base.templates.llm_text_completion_base import (
    LLMInitArgs,
)
from sinapsis_core.template_base.base_models import TemplateAttributes, TemplateAttributeType
from sinapsis_llama_cpp.helpers.llama_init_model import init_llama_model
from sinapsis_llama_cpp.templates.llama_text_completion import LLaMATextCompletion, LLaMATextCompletionAttributes

from sinapsis_llama_index.helpers.llama_index_pg_retriever import LLaMAIndexPGRetriever
from sinapsis_llama_index.helpers.pgvector_helpers import connect_to_table
from sinapsis_llama_index.helpers.schemas import HFEmbeddingConfig, RetrieveArgs, VectorStoreDBConfig

LLaMAIndexRAGTextCompletionUIProperties = LLaMATextCompletion.UIProperties
LLaMAIndexRAGTextCompletionUIProperties.tags.extend(
    [Tags.LLAMAINDEX, Tags.QUERY_CONTEXTUALIZATION, Tags.RETRIEVAL, Tags.RETRIEVAL_AG]
)


class LLaMACPPInitArgs(LLMInitArgs):
    """LLaMA-specific arguments for initializing the `llama_cpp.Llama` model.

    Attributes:
        llm_model_name (str): The name or path of the LLM model to use
                            (e.g. 'TheBloke/Llama-2-7B-GGUF').
        llm_model_file (str): The specific GGUF model file (e.g., 'llama-2-7b.Q2_K.gguf').
        context_window (int): The context window size. Defaults to `DEFAULT_CONTEXT_WINDOW`.
        model_kwargs (dict[str, Any] | None): Additional keyword arguments passed directly to the underlying
            `llama_cpp.Llama` constructor. Defaults to `None`.
        verbose (bool): Enable verbose logging from llama.cpp. Defaults to `True`.
    """

    llm_model_file: str
    context_window: int = DEFAULT_CONTEXT_WINDOW
    model_kwargs: dict[str, Any] | None = None
    verbose: bool = True


class LLaMACPPCompletionArgs(BaseModel):
    """LLaMA-specific arguments for `create_chat_completion`, inheriting base parameters.

    Attributes:
        temperature (float): Controls randomness. 0.0 = deterministic, >0.0 = random. Defaults to `0.2`.
        max_new_tokens (int): The maximum number of new tokens to generate.
        generate_kwargs (dict[str, Any] | None): Additional keyword arguments passed directly to the underlying
            `llama_cpp.Llama.create_chat_completion` call during generation. Defaults to `None`.
    """

    temperature: float
    max_new_tokens: int = DEFAULT_NUM_OUTPUTS
    generate_kwargs: dict[str, Any] | None = None


class LLaMARAGAttributes(LLaMATextCompletionAttributes):
    """Attributes for configuring a LLaMA-based Retrieval-Augmented Generation (RAG) system.

    Inherits from `RAGAttributes` and `LLaMAAttributes` to provide the necessary
    configuration parameters for a RAG system that integrates the LLaMA model.
    This includes settings for both retrieval-based augmentation and model-specific
    parameters.

    Attributes:
        init_args (LLaMACPPInitArgs): Arguments for initializing the LlamaCPP model wrapper,
            including model name/file, context window, and underlying llama_cpp settings.
        completion_args (LLaMACPPCompletionArgs): Arguments controlling the generation process
            within the LlamaCPP wrapper, like temperature and max_new_tokens.
        chat_history_key (str | None): Key in the packet's generic_data to find
            the conversation history.
        rag_context_key (str | None): Key in the packet's generic_data to find
            RAG context to inject.
        system_prompt (str | Path | None): The system prompt (or path to one)
            to instruct the model.
        pattern (str | None): A regex pattern used to post-process the model's response.
        keep_before (bool): If True, keeps text before the 'pattern' match; otherwise,
            keeps text after.
        embedding_config (HFEmbeddingConfig): Configuration for the HuggingFace embedding model,
            including model_name, device, trust_remote_code, etc.
        db_config (VectorStoreDBConfig): Configuration for connecting to the PostgreSQL database,
            handling credentials via environment variables if needed.
        retrieve_args (RetrieveArgs): Parameters controlling the retrieval process, such as
            query mode, top_k results to fetch, and the similarity threshold.
    """

    init_args: LLaMACPPInitArgs
    completion_args: LLaMACPPCompletionArgs
    embedding_config: HFEmbeddingConfig
    db_config: VectorStoreDBConfig = Field(default_factory=VectorStoreDBConfig)
    retrieve_args: RetrieveArgs = Field(default_factory=RetrieveArgs)


class LLaMAIndexRAGTextCompletion(LLaMATextCompletion):
    """Template for configuring and initializing a LLaMA-based Retrieval-Augmented Generation (RAG) system.

    This class manages the setup of a Retrieval-Augmented Generation (RAG) system
    by integrating the LLaMA model for generative tasks alongside retrieval-based
    augmentations, typically using external knowledge sources. It handles downloading
    and initializing the model while configuring relevant retrieval augmentations
    using provided attributes using the llama-index framework for the context retrieval and
    response generation.


    Usage example:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
      class_name: InputTemplate
      attributes: {}
    - template_name: LLaMARAGChat
      class_name: LLaMAIndexRAGTextCompletion
      template_input: InputTemplate
      attributes:
        init_args:
          llm_model_name: 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'
          llm_model_file: 'mistral-7b-instruct-v0.2.Q2_K.gguf'
          context_window: 4096
          model_kwargs:
            n_gpu_layers: -1
            n_threads: 4
        completion_args:
          max_new_tokens: 256
          temperature: 0.5
        role: assistant
        keep_before: true
        embedding_config:
            model_name: BAAI/bge-small-en
            trust_remote_code: true
            device: cuda
        db_config:
          db_name: vector_db
          table_name: llama2
        retrieve_args:
          query_mode: default
          top_k: 2
        system_prompt: You are an expert in AI.

    """

    AttributesBaseModel: TemplateAttributes = LLaMARAGAttributes
    UIProperties = LLaMAIndexRAGTextCompletionUIProperties

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """Initializes the RAG system with the provided attributes.

        This method sets up the vector store, retrieval system, query engine, and
        llm model which are essential elements for the retrieval-augmented generation process.

        """
        super().__init__(attributes)
        self.embedding_model, embedding_dimension = self._init_embed_model()
        self.vector_store = self.init_vector_store(embedding_dimension)
        self.retriever = self.init_retriever()
        self.query_engine = self.create_query_engine()

    def reset_llm_state(self) -> None:
        """Resets the internal state, ensuring that no memory, context, or cached information persists.

        This method calls `reset()` on the model to clear its internal state and `reset_llm_context()`
        to reset any additional context management mechanisms.

        Subclasses may override this method to implement model-specific reset behaviors if needed.
        """
        self.llm._model.reset()

    def init_llm_model(self) -> LlamaCPP:
        """Initializes the LLaMA model using the downloaded model path and the configuration attributes.

        This method downloads the LLaMA model from the Hugging Face Hub using the
        model name and file attributes, then sets up the model with parameters
        such as the number of tokens, temperature, and GPU/CPU settings. The model
        is then returned as an initialized instance of `LlamaCPP`, which is designed
        to handle large-scale models efficiently.

        Returns:
            LlamaCPP: An instance of the LlamaCPP model, initialized with the
                      specified configuration.
        """
        init_args: dict = self.attributes.init_args.model_dump(
            exclude_none=True, exclude={"llm_model_file", "llm_model_name"}
        )
        init_args.update(self.attributes.completion_args.model_dump(exclude_none=True))
        return init_llama_model(
            self.attributes.init_args.llm_model_name,
            self.attributes.init_args.llm_model_file,
            init_args,
            model_type="LlamaCPP",
        )

    def init_vector_store(self, embedding_dimension: int) -> PGVectorStore:
        """Initialize the vector store for storing and retrieving embeddings.

        This method connects to the database and initializes the vector store. It can be overridden by subclasses
        to provide custom vector store initialization logic.

        Returns:
                VectorStore: The initialized vector store.
        """
        vector_store = connect_to_table(dimension=embedding_dimension, **self.attributes.db_config.model_dump())
        return vector_store

    @staticmethod
    def _get_embedding_dimension(model: HuggingFaceEmbedding) -> int:
        """Determines the embedding dimension of the HuggingFace model.

        Args:
            model (HuggingFaceEmbedding): The initialized LlamaIndex HuggingFaceEmbedding wrapper.

        Returns:
            int: The determined embedding dimension
        """
        dimension = None
        if hasattr(model, "_model") and hasattr(model._model, "get_sentence_embedding_dimension"):
            dimension = model._model.get_sentence_embedding_dimension()
            if dimension:
                return dimension
        test_embedding = model.get_text_embedding("test")
        return len(test_embedding)

    def _init_embed_model(self) -> tuple[HuggingFaceEmbedding, int]:
        """Initialize the embedding model.

        This method initializes the embedding model using the HuggingFace API. It can be overridden by subclasses
        to provide custom embedding model initialization.

        Returns:
            HuggingFaceEmbedding: The initialized embedding model.
        """
        model = HuggingFaceEmbedding(**self.attributes.embedding_config.model_dump(exclude_none=True))
        dimension = self._get_embedding_dimension(model=model)
        return model, dimension

    def init_retriever(self) -> LLaMAIndexPGRetriever:
        """Initialize the vector retrieval system.

        This method initializes the vector retrieval system, which uses the vector store and the embedding model.
        It can be overridden by subclasses to provide custom synthesizer initialization logic.

        Returns:
                VectorDBRetriever: The initialized vector retrieval system.
        """
        retriever = LLaMAIndexPGRetriever(
            self.vector_store, self.embedding_model, **self.attributes.retrieve_args.model_dump()
        )
        return retriever

    def create_query_engine(self) -> RetrieverQueryEngine:
        """Creates a query engine using the retriever.

        This method initializes a query engine that uses the retriever and LLM for executing queries.
        It can be overridden by subclasses if custom query engine behavior is needed.

        Returns:
            RetrieverQueryEngine: The query engine for executing queries.
        """
        return RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            llm=self.llm,
            response_synthesizer=get_response_synthesizer(llm=self.llm, response_mode=ResponseMode.REFINE),
        )

    def get_response(self, input_message: str | list[dict]) -> str | None:
        """This method uses the query engine to process the provided query string and generate a response.

        Args:
            input_message (str | list): The input text or prompt to which the model
            will respond.

        Returns:
            str|None: The model's response as a string, or None if no response
                is generated.
        """
        full_query = ""

        for message in input_message:
            message = cast(dict, message)
            if message.get("content", False):
                full_query += message.get("content", False)

        response = self.query_engine.query(full_query)
        self.logger.debug(response)
        if response:
            return postprocess_text(str(response), self.attributes.pattern, self.attributes.keep_before)
        return None
