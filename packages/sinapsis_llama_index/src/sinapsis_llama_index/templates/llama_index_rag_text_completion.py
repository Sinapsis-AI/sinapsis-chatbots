# -*- coding: utf-8 -*-
from typing import cast

from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.postgres import PGVectorStore
from sinapsis_chatbots_base.helpers.postprocess_text import postprocess_text
from sinapsis_core.template_base.base_models import TemplateAttributes, TemplateAttributeType
from sinapsis_llama_cpp.helpers.llama_init_model import init_llama_model
from sinapsis_llama_cpp.templates.llama_text_completion import LLaMATextCompletion, LLaMATextCompletionAttributes
from sinapsis_llama_index.helpers.llama_index_pg_retriever import LLaMAIndexPGRetriever, connect_to_table


class LLaMARAGAttributes(LLaMATextCompletionAttributes):
    """Attributes for configuring a LLaMA-based Retrieval-Augmented Generation (RAG) system.

    Inherits from `RAGAttributes` and `LLaMAAttributes` to provide the necessary
    configuration parameters for a RAG system that integrates the LLaMA model.
    This includes settings for both retrieval-based augmentation and model-specific
    parameters.

    Attributes:
        llm_model_name (str): The name of the LLaMA model to use for generation.
        llm_model_file (str): The specific model file to be used for initialization.

        max_tokens (int): The maximum number of tokens to generate in a response.
        temperature (float): Controls the randomness of the model's output.
        n_gpu_layers (int): Number of layers to execute on the GPU.
        n_threads (int): Number of CPU threads to use for model inference.
        embedding_model_name (str): The name of the embedding model to use.
        db_name (str): The name of the database where embeddings are stored.
        table_name (str): The name of the table in the database for embeddings.
        database_dimension (int): The dimension of the database's embeddings.
        query_mode (str): The search strategy used for querying ('default' or other strategies).
        top_k (int): The number of top results to return during retrieval.
        pattern (str | None): A regex pattern to match delimiters. Defaults to handling `<|...|>` and `</...>`.
        keep_before (bool): If True, returns the portion before the first match; if False,
            returns the portion after the first match.

        trust_remote_code (bool): Whether to allow custom models defined on the HuggingFace Hub
            to execute their own code. Defaults to False.
    """

    embedding_model_name: str
    context_window: int = 7000
    db_name: str
    table_name: str
    database_dimension: int
    query_mode: str
    top_k: int
    trust_remote_code: bool = False


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
      class_name: LLaMARAGChat
      template_input: InputTemplate
      attributes:
        llm_model_name: bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF
        llm_model_file: DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf
        max_tokens: 8000
        n_ctx: 9024
        temperature: 0.6
        n_threads: 8
        n_gpu_layers: 20
        embedding_model_name: BAAI/bge-small-en
        db_name: vector_db
        table_name: llama2
        database_dimension: 384.0
        query_mode: default
        top_k: 2
        keep_before: false
        system_prompt: You are an expert in AI.
        trust_remote_code: true

    """

    AttributesBaseModel: TemplateAttributes = LLaMARAGAttributes

    def __init__(self, attributes: TemplateAttributeType) -> None:
        """
        Initializes the RAG system with the provided attributes.

        This method sets up the vector store, retrieval system, query engine, and
        llm model which are essential elements for the retrieval-augmented generation process.

        """
        super().__init__(attributes)
        self.vector_store = self.init_vector_store()
        self.retriever = self.init_retriever()
        self.query_engine = self.create_query_engine()

    def reset_llm_state(self) -> None:
        """
        Resets the internal state of the language model, ensuring that no memory, context,
        or cached information from previous interactions persists in the current session.

        This method calls `reset()` on the model to clear its internal state and `reset_llm_context()`
        to reset any additional context management mechanisms.

        Subclasses may override this method to implement model-specific reset behaviors if needed.
        """
        self.llm._model.reset()

    def init_llm_model(self) -> LlamaCPP:
        """
        Initializes the LLaMA model using the downloaded model path and the configuration attributes.
        This method downloads the LLaMA model from the Hugging Face Hub using the
        model name and file attributes, then sets up the model with parameters
        such as the number of tokens, temperature, and GPU/CPU settings. The model
        is then returned as an initialized instance of `LlamaCPP`, which is designed
        to handle large-scale models efficiently.

        Returns:
            LlamaCPP: An instance of the LlamaCPP model, initialized with the
                      specified configuration.
        """
        return init_llama_model(self.attributes.model_dump(exclude_none=True))

    def init_vector_store(self) -> PGVectorStore:
        """
        Initialize the vector store for storing and retrieving embeddings.

        This method connects to the database and initializes the vector store. It can be overridden by subclasses
        to provide custom vector store initialization logic.

        Returns:
            VectorStore: The initialized vector store.
        """
        vector_store = connect_to_table(
            db_name=self.attributes.db_name,
            table_name=self.attributes.table_name,
            dimension=self.attributes.database_dimension,
        )
        return vector_store

    def _init_embed_model(self) -> HuggingFaceEmbedding:
        """
        Initialize the embedding model.

        This method initializes the embedding model using the HuggingFace API. It can be overridden by subclasses
        to provide custom embedding model initialization.

        Returns:
            HuggingFaceEmbedding: The initialized embedding model.
        """
        hugging_face_model = HuggingFaceEmbedding(
            self.attributes.embedding_model_name, trust_remote_code=self.attributes.trust_remote_code
        )

        return hugging_face_model

    def init_retriever(self) -> LLaMAIndexPGRetriever:
        """
        Initialize the vector retrieval system.

        This method initializes the vector retrieval system, which uses the vector store and the embedding model.
        It can be overridden by subclasses to provide custom synthesizer initialization logic.

        Returns:
            VectorDBRetriever: The initialized vector retrieval system.
        """
        embed_model = self._init_embed_model()
        retriever = LLaMAIndexPGRetriever(
            self.vector_store,
            embed_model,
            query_mode=self.attributes.query_mode,
            similarity_top_k=self.attributes.top_k,
        )
        return retriever

    def create_query_engine(self) -> RetrieverQueryEngine:
        """
        Creates a query engine using the retriever.

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
        """
        This method uses the query engine to process the provided query string and generate a response.
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
