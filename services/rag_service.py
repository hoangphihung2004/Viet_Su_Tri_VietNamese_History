"""
RAGService
==========

A Retrieval-Augmented Generation (RAG) pipeline that integrates multiple services
for processing user queries and generating context-aware answers, particularly
tailored to Vietnamese history.

This service follows an optimized multi-step pipeline:
1. Query understanding and transformation.
2. Keyword-based candidate retrieval (ElasticSearch).
3. Vector-based semantic search (Qdrant).
4. Document reranking (RerankService).
5. Final answer generation (GeminiService).

Components:
-----------
- GeminiService: Handles query transformation and final answer generation via LLM.
- QdrantService: Performs similarity search over embedded documents.
- ConversationHistory: Maintains chat context across user turns.
- ElasticSearch: Retrieves candidate documents using BM25 keyword search.
- RerankService: Re-ranks retrieved documents based on semantic relevance.
"""

from services.gemini_service import GeminiService
from services.qdrant_service import QdrantService
from services.conversation_history import ConversationHistory
from services.elastic_search_service import ElasticSearch
from typing import Dict, Any
from re_ranking.re_ranking import RerankService


class RAGService:
    """
    Implements the full RAG pipeline with modular services for query handling,
    retrieval, reranking, and response generation.
    """

    def __init__(self):
        """
        Initializes the RAGService and all dependent services.
        """
        print("[RAGService] Initializing component services...")
        self.__gemini_service = GeminiService()
        self.__qdrant_service = QdrantService()
        self.__conversation_history = ConversationHistory()
        self.__rerank_service = RerankService()
        self.__elastic_search = ElasticSearch()
        print("[RAGService] All components initialized successfully.")

    def __add_history(self, user_query: str, answer: str) -> None:
        """
        Adds the user query and AI-generated answer to the conversation history.

        Args:
            user_query (str): The input query from the user.
            answer (str): The AI-generated answer to be stored in memory.
        """
        self.__conversation_history.add_user_message(user_query)
        self.__conversation_history.add_ai_message(answer)

    def handling_query(self, user_query: str) -> Dict[str, Any]:
        """
        Handles a user query through the complete RAG pipeline.

        Steps:
        1. Query transformation and classification (e.g., greeting, off-topic).
        2. Enhanced query generation based on detected entity and category.
        3. Document retrieval using ElasticSearch (BM25).
        4. Semantic similarity search using Qdrant over retrieved candidates.
        5. Reranking of documents using the RerankService.
        6. Answer generation using GeminiService.
        7. Conversation history update (if applicable).

        Args:
            user_query (str): The raw input query from the user.

        Returns:
            Dict[str, Any]: A dictionary containing the final answer and optional document URL.
                {
                    "Answer": <str>,
                    "URL": <str or None>
                }
        """
        # BƯỚC 1: Phân tích query trong một lần gọi LLM duy nhất
        query_transform = self.__gemini_service.query_transform(user_query=user_query,
                                                                history=self.__conversation_history)

        # Handle greeting or off-topic queries early
        if query_transform["status"] == "greeting":
            return {"Answer": query_transform["normalized_query"], "URL": None}

        if query_transform["status"] == "off_topic":
            answer_text = "Xin lỗi, tôi là trợ lý chuyên về lịch sử Việt Nam. Tôi không thể trả lời các câu hỏi ngoài lĩnh vực này."
            return {"Answer": answer_text, "URL": None}

        # Extract structured info from transformed query
        query_normalize = query_transform["normalized_query"].strip()
        query_entity = query_transform["search_entity"].strip()
        table_type = query_transform["category"].strip()

        # Validate required fields
        if not query_entity or not table_type:
            answer_text = "Xin lỗi, tôi chưa hiểu rõ câu hỏi của bạn. Bạn có thể hỏi lại chi tiết hơn được không?"
            return {"Answer": answer_text, "URL": None}

        # Step 3: Enhance the query with semantic hint
        enhanced_query = self.__enhance_query(user_query=query_entity, table_type=table_type)

        # Step 4: Retrieve candidate document IDs using keyword search
        candidate_ids = self.__elastic_search.bm25_search(user_query=enhanced_query, limits=120)

        if not candidate_ids:
            answer_text = "Rất tiếc, tôi không tìm thấy tài liệu nào liên quan đến chủ đề này trong kho dữ liệu của mình."
            return {"Answer": answer_text, "URL": None}

        # Step 5: Semantic search over filtered candidate documents
        retrieved_docs = self.__qdrant_service.search_similarity(user_query=enhanced_query,
                                                                 candidate_ids=candidate_ids,
                                                                 limits=70)

        if not retrieved_docs:
            answer_text = "Rất tiếc, tôi không tìm thấy tài liệu nào liên quan đến chủ đề này trong kho dữ liệu của mình."
            return {"Answer": answer_text, "URL": None}

        # Step 6: Re-rank top documents
        reranked_docs = self.__rerank_service.handle_reranking(user_query=enhanced_query,
                                                               documents=retrieved_docs,
                                                               limits=self.__get_top(table_type=table_type))

        # Step 7: Generate the final answer using reranked documents
        final_answer_obj = self.__gemini_service.generate_answer_question(
            user_query=query_normalize,
            docs=reranked_docs
        )

        # Optionally update conversation history
        if final_answer_obj["URL"] is None:
            return final_answer_obj

        self.__add_history(query_normalize, final_answer_obj["Answer"])

        return final_answer_obj

    @staticmethod
    def __get_top(table_type: str) -> int:
        top_k = {"COMPARE": 30,
                 "STAGE": 20}
        return top_k.get(table_type, 13)

    @staticmethod
    def __enhance_query(user_query: str, table_type: str) -> str:
        """
        Adds semantic hint to the query based on the content category.

        Args:
            user_query (str): The base query or entity name.
            table_type (str): Category of the content (e.g., 'FIGURE', 'HERITAGE').

        Returns:
            str: A semantically enhanced query with a domain-specific prefix.
        """
        hints = {
            "MILITARY": "Thông tin quân sự",
            "PERIOD": "Thông tin thời kỳ",
            "STAGE": "Thông tin giai đoạn",
            "FIGURE": "Thông tin của nhân vật",
            "HERITAGE": "Thông tin di sản văn hóa"
        }
        hint = hints.get(table_type, "")
        return f"{hint} {user_query}".strip()


if __name__ == "__main__":
    rag = RAGService()

    while True:
        enter = input("Enter: ")
        res = rag.handling_query(enter)
        print(res)
