from embedding_data.model_embedding_initialization import ModelEmbedding
import uuid
from tqdm import tqdm
from qdrant_client.models import PointStruct
from embedding_data.text_chunker import TextSplitter


def create_meta_data(id_=None, vec=None, chunk_index=None, table_type=None, title=None, url=None, content=None):
    """
    Creates a Qdrant PointStruct object from provided data.

    This helper function standardizes the creation of data points for Qdrant
    by packaging a vector and its associated metadata (payload) into a
    single, structured object.

    Args:
        id_ (str, optional): The unique identifier for the data point.
        vec (list[float], optional): The vector embedding of the content.
        chunk_index (int, optional): The sequential index of the text chunk
            from its original source document.
        table_type (str, optional): The category or source table of the data
            (e.g., "DiSanVanHoa", "NhanVat").
        title (str, optional): The title of the original source document.
        url (str, optional): The URL of the original source document.
        content (str, optional): The text content of the chunk.

    Returns:
        PointStruct: A Qdrant PointStruct object ready for upserting into a
            collection.
    """
    meta = PointStruct(
        id=id_,
        vector=vec,
        payload={
            "chunk_index": chunk_index,
            "table_type": table_type,
            "title": title,
            "url": url,
            "content": content
        }
    )
    return meta


class EmbeddingDataToQdrant:
    """
    Processes and embeds tabular data for ingestion into Qdrant.

    This class orchestrates the entire pipeline from a raw data table (e.g., a
    Pandas DataFrame) to a list of Qdrant-ready PointStructs. The process
    includes:
    1. Ingesting a DataFrame and a `table_type` for context.
    2. Iterating through each row of the DataFrame.
    3. Splitting the 'content' field of each row into smaller text chunks.
    4. Prepending contextual information (table type and title) to each chunk
       to enhance the semantic quality of the embedding.
    5. Generating a vector embedding for each context-enriched chunk.
    6. Packaging the UUID, vector, and metadata into a PointStruct.

    Attributes:
        model_embedding (ModelEmbedding): An instance of the embedding model wrapper.
        text_splitter (TextSplitter): An instance of the text splitting utility.
        table_type (str): The category of the data being processed.
        add_info (str): A contextual prefix string derived from `table_type`,
                        used to enrich text before embedding.
        table (pd.DataFrame): The input data table.
        list_embedded (list[PointStruct]): A list that accumulates the
            processed PointStruct objects.
    """
    def __init__(self, table_type, table):
        """
        Initializes the EmbeddingDataToQdrant processor.

        Args:
            table_type (str): The name of the data category (e.g., "DiSanVanHoa",
                              "QuanSu"). This is used to generate a contextual prefix.
            table (pd.DataFrame): The input DataFrame. It must contain 'title',
                                  'url', and 'content' columns.
        """
        self.model_embedding = ModelEmbedding()

        self.text_splitter = TextSplitter()

        # Generate a contextual prefix based on the table type.
        # This helps the embedding model differentiate between similar terms
        # in different domains (e.g., "battle" in a military context vs. a
        # sports context).
        self.table_type = table_type
        if table_type == "DiSanVanHoa":
            self.add_info = "Thông tin di sản văn hóa "
        elif table_type == "GiaiDoan":
            self.add_info = "Thông tin giai đoạn "
        elif table_type == "QuanSu":
            self.add_info = "Thông tin quân sự "
        elif table_type == "ThoiKy":
            self.add_info = "Thông tin thời kỳ "
        else:
            self.add_info = "Thông tin của nhân vật "
        self.table = table
        self.list_embedded = []

    def process_single_doc(self):
        """
        Processes all rows in the provided DataFrame.

        This method iterates through the `table`, splits the content of each
        document, generates embeddings for each chunk, and populates the
        `self.list_embedded` attribute with the resulting `PointStruct` objects.
        A progress bar is displayed using tqdm.
        """
        for idx, row in tqdm(self.table.iterrows()):
            title = row['title']
            url = row['url']
            content = row['content']
            # Split the document content into chunks
            all_chunk = self.text_splitter.split_text(content)
            for i, chunk in enumerate(all_chunk):
                # Enrich the chunk with context for better embedding
                chunk = self.add_info + title + ": " + chunk

                # Generate the vector embedding
                chunk_embedded = self.__embedding_text(chunk)

                # Create the final PointStruct object
                info = create_meta_data(
                    id_=str(uuid.uuid4()),
                    vec=chunk_embedded[0],
                    chunk_index=i,
                    table_type=self.table_type,
                    title=title,
                    url=url,
                    content=chunk  # Store the enriched content
                )
                self.list_embedded.append(info)

    def __embedding_text(self, chunk):
        """
        Private helper method to embed a single piece of text.

        Args:
            chunk (str): The text chunk to embed.

        Returns:
            list[list[float]]: The resulting vector embedding from the model,
                               typically encapsulated in a list.
        """
        return self.model_embedding.get_embedding_context(chunk)

    def get_all_list_embedded(self):
        """
        Public entry point to execute the full processing pipeline and retrieve the results.

        This method first calls `process_documents()` to perform the embedding
        and then returns the complete list of generated `PointStruct` objects.

        Returns:
            list[PointStruct]: A list of all generated PointStructs, ready for
                               upload to a Qdrant collection.
        """
        self.process_single_doc()
        return self.list_embedded
