from langchain.memory import ConversationBufferMemory
from typing import List
from pdf_services.load_pdf import EmbeddingData
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


class PDFRAGService:
    """
    Implements a stateful, conversational RAG service for querying PDFs.

    This class provides a complete solution for building a "chat with your PDF"
    application. It manages the entire lifecycle:
    1.  Processing and embedding PDF documents into a vector store.
    2.  Setting up a retriever to fetch relevant context from the documents.
    3.  Maintaining conversation history for contextual follow-up questions.
    4.  Generating answers using a powerful generative model (Google Gemini).

    The service is stateful, meaning it must first process PDFs via the
    `process_pdf` method before it can answer questions.

    Attributes:
        __retriever: The LangChain retriever component responsible for fetching
            relevant documents from the vector store based on a query.
        __db: The in-memory FAISS vector store containing the embedded PDF content.
        __embedding_data (EmbeddingData): An instance of the class responsible for
            the PDF-to-vector-store conversion logic.
        __gemini_service (ChatGoogleGenerativeAI): The configured generative AI
            model instance for generating answers.
        __memory (ConversationBufferMemory): The memory component that stores
            and manages the conversation history.
    """
    def __init__(self, model_name: str = os.getenv("GOOGLE_MODEL_NAME"),
                 google_api_key: str = os.getenv("GOOGLE_API_KEY_EXTRA"),
                 temperature: float = 0.4):
        """
        Initializes the PDFRAGService and its components.

        Args:
            model_name (str, optional): The name of the Google Gemini model to
                use. Defaults to the value of the "GOOGLE_MODEL_NAME"
                environment variable.
            google_api_key (str, optional): The API key for Google's AI
                services. Defaults to the value of the "GOOGLE_API_KEY_EXTRA"
                environment variable.
            temperature (float, optional): The creativity/randomness of the
                model's output. A lower value (e.g., 0.4) makes the output more
                deterministic and factual. Defaults to 0.4.
        """
        self.__retriever = None
        self.__db = None
        self.__embedding_data = EmbeddingData()
        self.__gemini_service = ChatGoogleGenerativeAI(model=model_name,
                                                       google_api_key=google_api_key,
                                                       temperature=temperature)
        self.__memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        print("[PDFRAGService] All components initialized successfully.")

    def process_pdf(self, pdf_paths: List[str], limits: int = 10) -> None:
        """
        Processes a list of PDF files and prepares them for querying.

        This method ingests the content of the provided PDFs, creates a vector
        store, and initializes the retriever. This is a mandatory prerequisite
        that must be called before using the `generate_answer` method.

        Args:
            pdf_paths (List[str]): A list of local file paths to the PDF
                documents to be indexed.
            limits (int, optional): The number of relevant document chunks (top-k)
                to retrieve from the vector store for each query. Defaults to 5.

        Side Effects:
            - Populates `self.__db` with a FAISS vector store.
            - Initializes `self.__retriever` to be used for answering questions.
        """
        self.__db = self.__embedding_data.process_pdf(pdf_paths)
        self.__retriever = self.__db.as_retriever(search_kwargs={"k": limits})
        print("[PDFRAGService] PDF processed & Retriever initialized.")

    def generate_answer(self, question: str):
        """
        Generates an answer to a given question using the indexed PDFs.

        This method leverages the previously configured retriever and the
        conversational memory to generate a context-aware answer. The
        `process_pdf` method must have been called successfully prior to this.

        Args:
            question (str): The user's question to be answered.

        Returns:
            str: The generated answer from the language model.

        Raises:
            AttributeError: If the retriever has not been initialized (i.e., if
            `process_pdf` was not called first).
        """
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.__gemini_service,
            retriever=self.__retriever,
            return_source_documents=False,
            verbose=False,
            memory=self.__memory,
        )

        result = qa_chain.invoke({"question": question})
        return result["answer"]


if __name__ == "__main__":
    paths = [r"C:\Users\USER\Downloads\Multistage_feature_fusion_knowledge_distillation.pdf"]
    rag = PDFRAGService()
    rag.process_pdf(paths)
    while True:
        enter = input("Enter: ")
        res = rag.generate_answer(enter)
        print(res)

