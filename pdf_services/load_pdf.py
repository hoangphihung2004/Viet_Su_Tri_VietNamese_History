from typing import List
from langchain_core.documents import Document
from embedding_data.text_chunker import TextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from embedding_data.model_embedding_initialization import ModelEmbedding


class EmbeddingData:
    """
    Provides a pipeline for processing PDF files into a FAISS vector store.

    This class encapsulates the entire process of loading data from PDF files,
    splitting the text into semantically coherent chunks, generating vector
    embeddings for these chunks, and finally indexing them in an in-memory
    FAISS database for efficient similarity searching.
    """
    def __init__(self):
        """
        Initializes the EmbeddingData processor.

        This sets up the necessary components for the pipeline, including a
        text splitter with pre-defined chunking parameters and the embedding
        model itself.
        """
        self.__text_splitter = TextSplitter(chunk_size=550, overlap=120)
        self.__model_embedding = ModelEmbedding().return_model()

    def __split_data(self, docs: List[Document]) -> List[Document]:
        """
        Private helper method to split a list of documents into smaller chunks.

        Args:
            docs (List[Document]): The list of LangChain Document objects to be
                split.

        Returns:
            List[Document]: A new list containing smaller Document chunks derived
                from the input documents.
        """
        splits = self.__text_splitter.split_documents(docs)
        return splits

    def __embedding_data(self, splits: List[Document]) -> FAISS:
        """
        Private helper method to embed document chunks and create a FAISS index.

        This method takes a list of document chunks, extracts their text
        content, generates embeddings using the initialized model, and creates
        an in-memory FAISS vector store.

        Args:
            splits (List[Document]): The list of document chunks to be embedded.

        Returns:
            FAISS: An in-memory FAISS vector store containing the indexed
                embeddings of the provided text chunks.
        """
        texts = [doc.page_content for doc in splits]
        database = FAISS.from_texts(texts=texts, embedding=self.__model_embedding)
        return database

    def process_pdf(self, pdf_paths: List[str]) -> FAISS:
        """
        Orchestrates the full pipeline from PDF file paths to a FAISS database.

        This is the main public method of the class. It takes a list of file
        paths to PDF documents and performs the following steps:
        1. Loads the content from all specified PDFs.
        2. Splits the loaded content into smaller, manageable chunks.
        3. Embeds these chunks and indexes them into a new FAISS vector store.

        Args:
            pdf_paths (List[str]): A list of local file paths to the PDF
                documents that need to be processed.

        Returns:
            FAISS: A searchable, in-memory FAISS vector store containing the
                embedded content from all provided PDFs.
        """
        all_docs = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
        splits = self.__split_data(all_docs)
        database = self.__embedding_data(splits)
        return database

