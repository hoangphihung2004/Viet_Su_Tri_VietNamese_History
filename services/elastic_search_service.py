"""
ElasticSearch
=============

A wrapper class for performing BM25 keyword-based document retrieval using OpenSearch.

This class connects to an OpenSearch cluster (compatible with Elasticsearch API)
and allows performing multi-field text search across indexed documents.

Environment Variables (via .env):
---------------------------------
- BONSAI_ELASTICSEARCH_URL: The OpenSearch server URL.
- ES_INDEX_NAME: The index to search from.
"""

import os
from opensearchpy import OpenSearch
from dotenv import load_dotenv
from typing import List
load_dotenv()


class ElasticSearch:
    """
    Provides a simplified interface to perform BM25 search queries on an OpenSearch index.
    """
    def __init__(self, opensearch_url: str = os.getenv("BONSAI_ELASTICSEARCH_URL"),
                 index_name: str = os.getenv("ES_INDEX_NAME")):
        """
        Initializes the OpenSearch client with the specified URL and index.

        Args:
            opensearch_url (str): The OpenSearch server endpoint.
            index_name (str): The name of the index to query.
        """
        print("[ElasticSearch] Initializing ElasticSearch...")
        self.__client = OpenSearch(
            hosts=[opensearch_url],
            http_compress=True,
            use_ssl=True,
            verify_certs=True,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=20
        )
        self.__index_name = index_name
        self.__opensearch_url = opensearch_url
        print("[ElasticSearch] ElasticSearch initialized.")

    def bm25_search(self, user_query: str, limits: int = 100) -> List[str]:
        """
        Executes a BM25-based keyword search over the specified OpenSearch index.

        The search uses a `multi_match` query across the "title" and "content" fields,
        with "title" given higher weight (boost factor 2).

        Args:
            user_query (str): The user input or query string to search for.
            limits (int): Maximum number of document IDs to return. Default is 100.

        Returns:
            List[str]: A list of document IDs (as strings) matching the query, ranked by relevance.

        Notes:
            - If the user query is empty or invalid, an empty list is returned.
            - If an exception occurs during search, it is logged and an empty list is returned.
        """
        if user_query is None or user_query.strip() == "":
            return []

        search_body = {
            "size": limits,
            "query": {
                "multi_match": {
                    "query": user_query,
                    "fields": ["title^2", "content"]
                }
            },
            "_source": False
        }

        try:
            response = self.__client.search(index=self.__index_name, body=search_body)
            hits = response['hits']['hits']
            ids_list = [hit['_id'] for hit in hits]

            return ids_list
        except Exception as e:
            print(f"[ElasticSearch] Error: {e}")
            return []


