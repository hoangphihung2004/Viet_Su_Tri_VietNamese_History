from dotenv import load_dotenv
import os
from typing import List, Dict, Any
import cohere

load_dotenv()


class RerankModel:
    """
    A client for Cohere's Rerank API to re-order documents.

    This class serves as a direct wrapper for Cohere's reranking endpoint.
    It takes a user query and a list of candidate documents and uses a
    powerful language model to compute a semantic relevance score for each
    document. This is typically used as a second-stage refinement in a
    retrieval pipeline (e.g., after an initial vector or BM25 search) to
    improve the precision of search results.

    Attributes:
        __client (cohere.Client): The initialized Cohere client instance.
        __model (str): The name of the Cohere reranking model to be used.
    """
    def __init__(self, api_key: str = os.getenv("COHERE_API_KEY"), model_name: str = os.getenv("RERANK_MODEL_NAME")):
        """"
        Initializes the RerankModel client.

        Args:
            api_key (str, optional): The API key for authenticating with the
                Cohere service. Defaults to the value of the "COHERE_API_KEY"
                environment variable.
            model_name (str, optional): The specific reranking model to use
                (e.g., "rerank-english-v3.0"). Defaults to the value of the
                "RERANK_MODEL_NAME" environment variable.
        """
        print("[RerankModel] Initializing Cohere reranker...")
        self.__client = cohere.Client(api_key)
        self.__model = model_name
        print("[RerankModel] Cohere reranker initialized.")

    def reranking(self, user_query: str, docs: List[Dict[str, Any]], limits: int = 10) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of documents based on relevance to a user query.

        This method sends the query and document texts to the Cohere API,
        receives relevance scores, and re-orders the original documents
        accordingly. A new key, "Rerank_Score", is added to each document
        dictionary.

        Args:
            user_query (str): The user's search query.
            docs (List[Dict[str, Any]]): A list of document dictionaries. Each
                dictionary must contain a "Content" key holding the text to be
                reranked.
            limits (int, optional): The maximum number of top-ranked documents
                to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: A new list containing the documents sorted by
                their "Rerank_Score" in descending order, truncated to the
                specified limit. If the API call fails, it returns the
                original list of documents, truncated to the limit.
        """
        if not docs:
            return []

        try:
            doc_texts = [doc["Content"] for doc in docs]
            response = self.__client.rerank(
                model=self.__model,
                query=user_query,
                documents=doc_texts,
                top_n=min(limits, len(docs))
            )

            # Initialize all documents with default score
            for doc in docs:
                doc["Rerank_Score"] = 0.0

            # Update scores for reranked documents
            for result in response.results:
                docs[result.index]["Rerank_Score"] = result.relevance_score

            # Sort by rerank score and return top_k
            sorted_docs = sorted(docs, key=lambda x: x.get("Rerank_Score", 0.0), reverse=True)
            return sorted_docs[:limits]

        except Exception as e:
            print(f"[reranking] Error: {e}")
            return docs[:limits]  # Return original docs if reranking fails


class RerankService:
    """
    A high-level service wrapper for the reranking functionality.

    This class provides a clean abstraction layer over the `RerankModel`. Its
    purpose is to decouple the main application logic from the specific
    implementation details of the Cohere client, making it easier to integrate,
    test, and potentially swap out the reranking provider in the future.

    Attributes:
        __rerank_model (RerankModel): An instance of the underlying reranking
            model client.
    """
    def __init__(self):
        """
        Initializes the RerankService.

        This creates an instance of the underlying `RerankModel` which will be
        used to perform the actual reranking operations.
        """
        print("[RAGService] Initializing RerankService with Cohere reranker...")
        self.__rerank_model = RerankModel()
        print("[RAGService] RerankService initialized.")

    def handle_reranking(self, user_query: str, documents: List[Dict[str, Any]], limits: int = 10) -> List[Dict[str, Any]]:
        """
        Processes a reranking request for a given query and documents.

        This method serves as the primary public interface for the reranking
        service. It validates the input and delegates the reranking task to
        the underlying `RerankModel`.

        Args:
            user_query (str): The user's search query.
            documents (List[Dict[str, Any]]): A list of document dictionaries
                to be reranked. Each dictionary must have a "Content" key.
            limits (int, optional): The maximum number of documents to return.
                Defaults to 10.

        Returns:
            List[Dict[str, Any]]: The reranked and sorted list of documents.
                Returns an empty list if the input query or documents are empty,
                or if an unexpected error occurs.
        """
        try:
            if not user_query or not documents:
                print("[handle_reranking] Empty query or documents")
                return []
            return self.__rerank_model.reranking(user_query=user_query, docs=documents, limits=limits)
        except Exception as e:
            print(f"[handle_reranking] Error: {e}")
            return []
