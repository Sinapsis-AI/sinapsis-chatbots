# -*- coding: utf-8 -*-

from typing import Any

import numpy as np
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
        query_mode: str = "default",
        similarity_top_k: int = 2,
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
        best_node = None
        best_score = float("-inf")
        for index, node in enumerate(query_result.nodes):
            if query_result.similarities is not None:
                score = np.mean(query_result.similarities)
                nodes_with_scores.append(NodeWithScore(node=node, score=score))
                if score is not None and score > self.threshold and score > best_score:
                    best_node = node
                    best_score = score

        if best_node is not None:
            sinapsis_logger.debug(f"Best Node: {best_node}")
            return [NodeWithScore(node=best_node, score=best_score)]
        sinapsis_logger.debug(f"Nodes with scores: {nodes_with_scores}")
        return nodes_with_scores


def connect_to_table(
    db_name: str,
    table_name: str,
    user: str,
    password: str,
    dimension: int = 384,
    host: str = "localhost",
    port: str = "5432",
) -> PGVectorStore:
    """
    Creates and connects to a PostgreSQL vector table using LlamaIndex's `PGVectorStore`.

    This method initializes a `PGVectorStore` instance with the given database connection
    details and returns it to interact with the vector table in the database.

    Args:
        db_name (str): name of the database to connect to
        table_name (str): Name of the table
        dimension (int): Dimension of the vector database
        host (str): Host direction for the database
        port (str): Port where the database is hosted
        user (str): Username for the database connection
        password (str): Password for the database connection

    Returns:
        PGVectorStore: An instance of the `PGVectorStore` that allows interacting with the
            PostgreSQL vector table.
    """
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
        table_name=table_name,
        embed_dim=dimension,
    )
    return vector_store


def delete_table(
    db_name: str,
    table_name: str,
    user: str,
    password: str,
    dimension: int = 384,
    host: str = "localhost",
    port: str = "5432",
) -> None:
    """
    Method to clear a table from a PGVector database
    Args:
        db_name (str): name of the database to connect to
        table_name (str): Name of the table
        dimension (int): Dimension of the vector database
        host (str): Host direction for the database
        port (str): Port where the database is hosted
        user (str): Username for the database connection
        password (str): Password for the database connection



    """
    database = connect_to_table(db_name, table_name, user, password, dimension, host, port)
    database.clear()
