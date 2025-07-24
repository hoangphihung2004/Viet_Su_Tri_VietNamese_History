"""
QdrantService
=============

A unified service to interact with a Qdrant vector database.

This class abstracts all major operations needed to manage and search collections
inside Qdrant for use in a Vietnamese history retrieval-augmented generation (RAG) system.
It supports:
- Collection creation/deletion
- Batched data insertion
- Pure vector similarity search (semantic)
- Filtering search by `table_type` and candidate document IDs

Environment Variables (via .env):
---------------------------------
- QDRANT_HOST (str): Hostname of the Qdrant instance
- QDRANT_PORT (str or int): Port of the Qdrant instance
- QDRANT_COLLECTION_NAME (str): Collection name to store vectors

Dependencies:
-------------
- qdrant-client
- dotenv
- tqdm
- embedding_data.model_embedding_initialization.ModelEmbedding
"""

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchAny, HasIdCondition
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from embedding_data.model_embedding_initialization import ModelEmbedding
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()


class QdrantService:
    """
    A service wrapper for Qdrant vector database, optimized for semantic search
    in a Vietnamese historical document context.
    """
    def __init__(self):
        """
        Initializes the QdrantService with a connection to the database and an embedding model.
        """
        print("[QdrantService] Initializing QdrantService...")
        self.client = QdrantClient(
            host=os.getenv('QDRANT_HOST'),
            port=os.getenv('QDRANT_PORT'),
            timeout=30,
        )
        self.__collection_name = os.getenv('QDRANT_COLLECTION_NAME')
        self.__embedding_model = ModelEmbedding()
        print("[QdrantService] QdrantService initialized.")

    def create_collection(self):
        """
        Creates a fresh collection in Qdrant with Cosine distance.

        If the collection already exists, it will be deleted first.
        This operation is useful when re-indexing or re-building the database.
        """
        if self.client.collection_exists(collection_name=self.__collection_name):
            print(f"Collection {self.__collection_name} exists. Deleting...")
            self.client.delete_collection(collection_name=self.__collection_name)
            print("Deleted old collection.\n")

        self.client.create_collection(
            collection_name=self.__collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            shard_number=1
        )
        print(f"Successfully created collection '{self.__collection_name}'.")

    def insert_data(self, points, batch_size=64):
        """
        Inserts (or upserts) points into the Qdrant collection in batches.

        Args:
            points (List[models.PointStruct]): A list of Qdrant PointStruct objects.
                Each must have an ID, a vector, and a payload (e.g. content, title).
            batch_size (int): Number of points to insert per request. Default is 64.
        """
        for batch in tqdm(range(0, len(points), batch_size), desc="Inserting data"):
            batch = points[batch:batch + batch_size]
            self.client.upsert(
                collection_name=self.__collection_name,
                points=batch,
                wait=True
            )

    def search_similarity(self, user_query: str, candidate_ids: List[str], limits: int = 10) -> List[Dict[str, Any]]:
        """
        Performs semantic search over the given candidate document IDs using vector similarity.

        This method uses a pre-trained embedding model to encode the query and returns
        the top-k semantically similar documents from Qdrant.

        Args:
            user_query (str): The user's search query.
            candidate_ids (List[str]): A list of document IDs (from BM25 or ElasticSearch).
            limits (int): Maximum number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of matching documents, each with fields like:
                - Chuck_Index
                - Title
                - URL
                - Table_Type
                - Content
        """
        try:

            embed_query = self.__embedding_model.get_embedding_query(text=user_query)

            # Stage 2: Vector Re-ranking
            search_points = self.client.query_points(
                collection_name=self.__collection_name,
                query=embed_query,
                query_filter=models.Filter(must=[HasIdCondition(has_id=candidate_ids)]),
                limit=limits,
                with_payload=True
            )

            search_points = list(search_points)[0][1]

            search_results = []
            for point in search_points:
                search_results.append({
                    "Chuck_Index": point.payload.get("chunk_index"),
                    "Title": point.payload.get("title"),
                    "URL": point.payload.get("url"),
                    "Table_Type": point.payload.get("table_type"),
                    "Content": point.payload.get("content")
                })
            return search_results

        except Exception as e:
            print(f"[search_similarity] Error: {e}")
            return []

    @staticmethod
    def __filters(table_type: Optional[List[str]] = None) -> Optional[Filter]:
        """
        Constructs a Qdrant filter for the given table types.

        Args:
            table_type (Optional[List[str]]): A list of valid table_type values to include.

        Returns:
            Optional[Filter]: A Qdrant `Filter` object if `table_type` is specified,
                              otherwise None.
        """
        if not table_type:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="table_type",
                    match=MatchAny(any=table_type)
                )
            ]
        )
