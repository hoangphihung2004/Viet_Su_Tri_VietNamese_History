"""
Data Migration Script: Qdrant to OpenSearch

This script facilitates the migration of document data (specifically ID, title, and content)
from a Qdrant vector database collection to an OpenSearch index.

Purpose:
It is designed for a two-stage (or "re-ranker") search architecture where:
1.  OpenSearch, with its powerful BM25 algorithm, serves as the fast, first-stage retriever
    to recall a set of relevant candidate documents based on keywords.
2.  Qdrant is used for the second-stage, high-precision semantic re-ranking of the candidates
    retrieved from OpenSearch.

Features:
-   Optimized for Vietnamese: Creates an OpenSearch index with a custom analyzer
    (using ICU components) to handle Vietnamese diacritics (accents) and stop words,
    enabling robust, accent-insensitive full-text search.
-   Efficient Batch Processing: Uses Qdrant's `scroll` API and OpenSearch's `bulk` helper
    for efficient, memory-safe data transfer.
-   Configuration-driven: All connection details and parameters are managed via a `.env` file.
-   Robust Error Handling: Includes connection checks and logs detailed information
    about the migration process.

Setup:
    Add the following environment variables to the `.env` file:
    BONSAI_ELASTICSEARCH_URL="<your_opensearch_url>"
    QDRANT_HOST="<your_qdrant_host>"
    QDRANT_PORT="<your_qdrant_port>"
    QDRANT_COLLECTION_NAME="<your_qdrant_collection>"
    ES_INDEX_NAME="<your_desired_opensearch_index_name>"
"""

import os
import logging
from typing import Dict, Any
from opensearchpy import OpenSearch, helpers
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

load_dotenv()

# Configure logging for detailed progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:
    """Manages application configuration by loading environment variables from a .env file."""

    def __init__(self):
        """
        Initializes the configuration object.

        Loads environment variables and validates that essential ones are present.

        Raises:
            ValueError: If required environment variables are not set.
        """
        self.es_url = os.getenv("BONSAI_ELASTICSEARCH_URL")
        self.qdrant_host = os.getenv("QDRANT_HOST")
        self.qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        self.qdrant_collection = os.getenv('QDRANT_COLLECTION_NAME')
        self.es_index = os.getenv('ES_INDEX_NAME')
        self.batch_size = 256

        if not all([self.es_url, self.qdrant_collection]):
            raise ValueError(
                "Missing required environment variables. Please set BONSAI_ELASTICSEARCH_URL and "
                "QDRANT_COLLECTION_NAME in your .env file.")


class ElasticsearchManager:
    """A manager class to handle all interactions with the OpenSearch cluster."""

    def __init__(self, es_url, index_name):
        """
        Initializes the OpenSearch client.

        Args:
            es_url (str): The connection URL for the OpenSearch cluster.
            index_name (str): The name of the index to operate on.

        Raises:
            Exception: If the connection to the cluster fails.
        """
        self.index_name = index_name
        try:
            # Use the OpenSearch client, which is compatible with many Elasticsearch providers
            self.es = OpenSearch(
                hosts=[es_url],
                http_compress=True,  # Enable gzip compression for better performance
                use_ssl=True,
                verify_certs=True,
                ssl_assert_hostname=False,  # Often needed for cloud-hosted providers
                ssl_show_warn=False,
                request_timeout=60
            )
            # Kiểm tra kết nối
            self.es.info()
            logging.info("Successfully connected to the OpenSearch cluster.")
        except Exception as e:
            logging.error(f"Failed to connect to the OpenSearch cluster: {e}")
            raise

    def create_index_for_vietnamese_bm25(self):
        """
        Creates an OpenSearch index with custom settings and mappings for Vietnamese text.

        If the index already exists, this operation is skipped. The custom analyzer
        supports accent-insensitive search and uses a custom list of Vietnamese stop words.
        """
        if self.es.indices.exists(index=self.index_name):
            logging.warning(f"Index '{self.index_name}' already exists. Skipping creation.")
            return

        logging.info(f"Creating new index '{self.index_name}' optimized for Vietnamese search...")
        # Define custom settings and mappings for Vietnamese language analysis
        settings_and_mappings = {
            "settings": {
                "analysis": {
                    "analyzer": {"vietnamese_analyzer":
                                     {"tokenizer": "icu_tokenizer",
                                      "filter": ["lowercase", "icu_folding", "vietnamese_stop"]
                                      }
                                 },
                    "filter": {"vietnamese_stop":
                                   {"type": "stop",
                                    "stopwords": ["bị", "bởi", "cả", "các", "cái", "cần", "càng", "chỉ",
                                                  "chiếc", "cho", "chứ", "chưa", "có", "có thể", "cứ",
                                                  "của", "cùng", "cũng", "đã", "đang", "đây", "để",
                                                  "đến", "đều", "đi", "được", "do", "đó", "gì", "khi",
                                                  "không", "là", "lại", "lên", "lúc", "mà", "mỗi", "một",
                                                  "nên", "nếu", "ngay", "nhiều", "như", "nhưng", "những",
                                                  "nơi", "nữa", "phải", "qua", "ra", "rằng", "rất",
                                                  "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thì",
                                                  "trên", "trước", "từ", "từng", "và", "vẫn", "vào",
                                                  "vậy", "vì", "việc", "với"]
                                    }
                               }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "vietnamese_analyzer"},
                    "content": {"type": "text", "analyzer": "vietnamese_analyzer"}
                }
            }
        }
        try:
            self.es.indices.create(index=self.index_name, body=settings_and_mappings)
            logging.info(f"Successfully created index '{self.index_name}' with Vietnamese analyzer.")
        except Exception as e:
            logging.error(f"Error creating index: {e}")
            raise

    def bulk_insert(self, actions):
        """
        Performs a bulk insert operation into the OpenSearch index.

        Args:
            actions (List[Dict[str, Any]]): A list of bulk actions to perform.

        Returns:
            Tuple[int, int]: A tuple containing the number of successful operations
                             and the number of failed operations.
        """
        try:
            success, errors = helpers.bulk(self.es, actions, index=self.index_name, raise_on_error=False)
            if errors:
                logging.warning(f"Encountered {len(errors)} errors during bulk insert. First error: {errors[0]}")
            return success, len(errors)
        except Exception as e:
            logging.error(f"A critical error occurred during the bulk operation: {e}")
            return 0, len(actions)


class DataMigrator:
    """Orchestrates the data migration process from Qdrant to OpenSearch."""

    def __init__(self, config):
        """
       Initializes the migrator, setting up clients for both databases.

       Args:
           config (Config): The application configuration object.

       Raises:
           Exception: If it cannot connect to Qdrant or find the specified collection.
       """
        self.config = config
        self.es_manager = ElasticsearchManager(config.es_url, config.es_index)
        try:
            self.qdrant_client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port, timeout=30)
            # Verify that the collection exists before proceeding
            self.qdrant_client.get_collection(collection_name=config.qdrant_collection)
            logging.info(f"Successfully connected to Qdrant and found collection '{config.qdrant_collection}'.")
        except Exception as e:
            logging.error(f"Failed to connect to Qdrant or find collection '{config.qdrant_collection}': {e}.")
            raise

    @staticmethod
    def _transform_point_to_action(point: PointStruct) -> Dict[str, Any]:
        """
        Transforms a Qdrant Point object into an OpenSearch bulk action dictionary.

        Args:
            point (PointStruct): The point object retrieved from Qdrant.

        Returns:
            Dict[str, Any]: A dictionary formatted for the OpenSearch bulk API.
        """
        payload = point.payload
        source_data = {"id": point.id, "title": payload.get("title", ""), "content": payload.get("content", "")}
        # The document _id in OpenSearch is set to the Qdrant point ID.
        # This is crucial for linking results from OpenSearch back to Qdrant for re-ranking.
        return {"_op_type": "index", "_id": point.id, "_source": source_data}

    def run(self) -> None:
        """
        Executes the main migration loop.

        It first ensures the target OpenSearch index exists with the correct settings,
        then scrolls through all points in the Qdrant collection, transforms them, and
        ingests them into OpenSearch in batches.
        """
        self.es_manager.create_index_for_vietnamese_bm25()
        offset = None
        total_success, total_failed = 0, 0
        logging.info("Bắt đầu di chuyển dữ liệu...")
        while True:
            # Scroll through the Qdrant collection to fetch points in batches
            try:
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=self.config.qdrant_collection, limit=self.config.batch_size,
                    offset=offset, with_payload=["title", "content"])  # Scroll through the Qdrant collection to fetch points in batches
            except Exception as e:
                logging.error(f"Error scrolling data from Qdrant: {e}")
                break
            if not points:
                # No more points to process
                break
            # Transform points to bulk actions and ingest into OpenSearch
            actions = [self._transform_point_to_action(p) for p in points]
            success_count, error_count = self.es_manager.bulk_insert(actions)
            total_success += success_count
            total_failed += error_count
            logging.info(
                f"Processed batch: {success_count} successful, {error_count} failed. "
                f"Total successful: {total_success}"
            )
            if next_offset is None:
                logging.info("Successfully scrolled through all data from Qdrant.")
                break
            offset = next_offset
        logging.info(f"Total documents successfully migrated: {total_success}")
        logging.info(f"Total documents failed to migrate: {total_failed}")


def main():
    """The main entry point for the script."""
    try:
        config = Config()
        migrator = DataMigrator(config)
        migrator.run()
    except (ValueError, Exception) as e:
        logging.critical(f"Script terminated due to a critical error: {e}")


if __name__ == "__main__":
    main()
