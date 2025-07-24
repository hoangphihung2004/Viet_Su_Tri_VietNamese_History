import os
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from dotenv import load_dotenv

# Load environment variables from a .env file at the root of the project
load_dotenv()


class ModelEmbedding:
    """
    A wrapper for HuggingFaceEmbeddings to streamline model initialization and usage.

    This class simplifies the process of using sentence-transformer models from
    Hugging Face by:
    1.  Loading the model name from an environment variable for easy configuration.
    2.  Automatically detecting and utilizing available GPU (CUDA) for performance.
    3.  Providing separate, clearly-named methods for embedding documents (for
        storage/indexing) and queries (for searching), which is a best
        practice for bi-encoder models.

    Attributes:
        model (HuggingFaceEmbeddings): The underlying LangChain embedding model
            instance that performs the embedding operations.
    """
    def __init__(self, model_name=os.getenv("MODEL_EMBEDDING")):
        """
        Initializes the ModelEmbedding wrapper.

        It sets up the embedding model by selecting the best available hardware
        and loading the specified model from Hugging Face.

        Args:
            model_name (str, optional): The Hugging Face model identifier
                (e.g., "bkai-foundation-models/vietnamese-bi-encoder").
                If not provided, it defaults to the value of the
                "MODEL_EMBEDDING" environment variable.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {'device': device}
        self.model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    def return_model(self):
        """
        Returns the underlying LangChain HuggingFaceEmbeddings model instance.

        This can be useful for accessing advanced functionalities of the
        LangChain model directly.

        Returns:
            HuggingFaceEmbeddings: The configured instance of the embedding model.
        """
        return self.model

    def get_embedding_context(self, text):
        """
        Generates an embedding for a single document or piece of context.

        This method is intended for embedding text that will be stored in a
        vector database. It uses the `embed_documents` method from the
        underlying model, which is optimized for corpus processing.

        Args:
            text (str): The document content to embed.

        Returns:
            list[list[float]]: A list containing a single vector embedding for
                               the provided text.
        """
        return self.model.embed_documents([text])

    def get_embedding_query(self, text):
        """
        Generates an embedding for a single search query.

        This method is intended for embedding user queries at search time. It
        uses the `embed_query` method, as some bi-encoder models apply
        different processing to queries versus documents.

        Args:
            text (str): The search query to embed.

        Returns:
            list[float]: The vector embedding for the query.
        """
        return self.model.embed_query(text)


