from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """
    A standardized wrapper for LangChain's RecursiveCharacterTextSplitter.

    This utility class provides a pre-configured interface for splitting text
    into smaller chunks. It is designed to be used consistently throughout an
    application to process both raw text strings and lists of LangChain
    `Document` objects.

    The underlying `RecursiveCharacterTextSplitter` attempts to split text
    along a prioritized list of separators (e.g., double newlines, newlines,
    periods) to keep semantically related content, like paragraphs or sentences,
    intact.

    Attributes:
        chunk_size (int): The maximum number of characters allowed in a chunk.
        overlap (int): The number of characters from the end of a preceding
            chunk to include at the beginning of the subsequent chunk. This
            helps maintain context across chunk boundaries.
        text_splitter (RecursiveCharacterTextSplitter): The configured instance
            of the LangChain text splitter.
    """
    def __init__(self, chunk_size=600, overlap=200):
        """
        Initializes the TextSplitter with specified chunking parameters.

        Args:
            chunk_size (int, optional): The target maximum size for each text
                chunk. Defaults to 600.
            overlap (int, optional): The size of the overlap between
                consecutive chunks. Defaults to 200.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            # These separators are tried in order.
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_text(self, text):
        """
        Splits a single raw text string into a list of smaller chunks.

        This method is ideal for processing a single piece of text where no
        metadata is involved.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of text chunks.
        """
        chunks = self.text_splitter.split_text(text)
        return chunks

    def split_documents(self, documents):
        """
        Splits a list of LangChain Document objects into smaller chunks.

        This method is designed to work with LangChain's `Document` format.
        Crucially, it preserves the metadata of the original documents and
        propagates it to the resulting chunks, which is essential for
        maintaining source information.

        Args:
            documents (List[Document]): A list of Document objects to be split.

        Returns:
            List[Document]: A new list of Document objects, where each element
                            is a chunk of an original document.
        """
        chunks = self.text_splitter.split_documents(documents)
        return chunks

