# -*- coding: utf-8 -*-
from typing import Any, Literal

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.postgres import PGVectorStore
from sinapsis_core.utils.logging_utils import sinapsis_logger


class LLaMAIndexPGRetriever(BaseRetriever):
    """Retriever over a PostgreSQL vector store.

    This class uses a vector store and an embedding model to perform retrieval
    of relevant documents based on a query. It interacts with a PostgreSQL
    vector store, retrieves the top-k most similar nodes, and returns them
    along with their similarity scores.
    """

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: Literal["default", "sparse", "hybrid", "text_search", "semantic_hybrid"] = "default",
        similarity_top_k: int = 1,
        threshold: float = 0.85,
    ) -> None:
        """Initializes the VectorDBRetriever with required parameters.

        Args:
            vector_store (PGVectorStore): The vector store used for querying.
            embed_model (Any): The model used to embed queries into vector representations.
            query_mode (str, optional): The query mode for retrieval (default is 'default').
            similarity_top_k (int, optional): The number of top similar results to return (default is 2).
            threshold (float): Threshold for embedding similarity
        """
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self.threshold = threshold
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieves the most relevant nodes from the vector store based on the query.

        This method processes the query, generates its embedding, and queries the vector
        store for similar nodes. It then pairs each node with its corresponding similarity score
        and returns them in a list of `NodeWithScore` objects.

        Args:
            query_bundle (QueryBundle): The query object containing the query string and metadata.

        Returns:
            list[NodeWithScore]: A list of `NodeWithScore` objects, each containing a node
                                  and its similarity score.
        """
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        if query_result.nodes:
            for node, similarity in zip(query_result.nodes, query_result.similarities):
                if similarity > self.threshold:
                    nodes_with_scores.append(NodeWithScore(node=node, score=similarity))
        sinapsis_logger.debug(f"Nodes with scores: {nodes_with_scores}")
        return nodes_with_scores
