from embedding_data.embedding_data_to_qdrant import EmbeddingDataToQdrant
from sqlserver.load_data import LoadData
from services.qdrant_service import QdrantService


def main():
    """
    Orchestrates the end-to-end data ingestion pipeline.

    This function executes the complete ETL (Extract, Transform, Load) process:
    1.  Extract: Loads all data tables from the source SQL Server database.
    2.  Transform: Iterates through each table, converting its rows into
        vectorized, chunked data points suitable for Qdrant.
    3.  Load: Inserts these data points into a pre-defined Qdrant collection.

    Side Effects:
        - A new Qdrant collection is created if it does not already exist.
        - Data from the source database is processed and inserted into the
          Qdrant collection.
    """
    all_data = LoadData().getdata()

    qdrant = QdrantService()
    qdrant.create_collection()
    for key, value in all_data.items():
        table_type = key
        print(table_type)
        collection = EmbeddingDataToQdrant(table_type=table_type, table=value).get_all_list_embedded()
        qdrant.insert_data(collection)


if __name__ == "__main__":
    main()
