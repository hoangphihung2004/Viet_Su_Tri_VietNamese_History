"""
GeminiService - A Language Model Service for Vietnamese History Q&A

This class wraps Google Gemini (via LangChain) into a specialized interface
for processing and answering Vietnamese history-related questions. It builds
two LangChain processing chains:

1. `transform_chain`: Classifies, normalizes, and extracts query intent.
2. `answer_chain`: Generates precise, contextual answers using chunked data.

The GeminiService is designed for:
- Understanding ambiguous or informal user input
- Resolving context using dialogue history
- Generating structured output suitable for search and answer generation

Usage:
    gemini = GeminiService()
    result = gemini.query_transform("ông ấy là ai", history)
    answer = gemini.generate_answer_question("ông ấy là ai", docs)

Dependencies:
    - LangChain
    - Google Generative AI via LangChain
    - dotenv for environment management
    - ConversationHistory class
"""

import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from services.conversation_history import ConversationHistory

load_dotenv()


class GeminiService:
    def __init__(self, model_name: str = os.getenv("GOOGLE_MODEL_NAME")):
        """
        Initializes the GeminiService.

        Sets up the underlying Google Generative AI model via LangChain, and
        prepares two main prompt chains:
            - transform_chain for query understanding
            - answer_chain for generating factual answers

        Args:
            model_name (str): The LLM model name. Fetched from environment if not provided.
        """
        print("[GeminiService] Initializing GeminiService...")
        self.__llm_model = ChatGoogleGenerativeAI(model=model_name, temperature=0.4,
                                                  google_api_key=os.getenv("GOOGLE_API_KEY"))

        self.__build_chains()
        print("[GeminiService] GeminiService initialized.")

    def get_models(self):
        """
        Returns the underlying LLM model instance.

        Useful for debugging, manual prompting, or advanced LangChain chaining.

        Returns:
            ChatGoogleGenerativeAI: The model instance.
        """
        return self.__llm_model

    def __build_chains(self):
        """
        Constructs LangChain pipelines (chains) for:

        - `transform_chain`: Analyzes the user question, uses conversation history,
                             classifies status, extracts search entity, and determines category.

        - `answer_chain`: Synthesizes a final response using document chunks, filtering unrelated data,
                          and formats a concise, source-linked answer.

        This method uses templated prompts tailored for Vietnamese history Q&A.
        """
        str_parser = StrOutputParser()

        transform_prompt = ChatPromptTemplate.from_template(
            """
            Bạn là một AI chuyên gia phân tích và xử lý câu hỏi về lịch sử Việt Nam.
            Nhiệm vụ của bạn là phân tích toàn diện câu hỏi của người dùng, dựa trên lịch sử hội thoại, và trả về một đối tượng JSON có cấu trúc duy nhất.

            **QUY TRÌNH PHÂN TÍCH GỒM 4 BƯỚC BẮT BUỘC:**

            **BƯỚC 1: PHÂN LOẠI TỔNG QUÁT (`status`)**
            - Nếu câu hỏi chỉ là lời chào ("xin chào", "hello", "bạn là ai"): đặt status là "greeting".
            - Nếu câu hỏi không liên quan đến lịch sử Việt Nam (thời tiết, cà phê, lập trình): đặt status là "off_topic".
            - Nếu là câu hỏi hợp lệ về lịch sử: đặt status là "ok".

            **BƯỚC 2: CHUẨN HÓA CÂU HỎI (`normalized_query`)**
            - Luôn thực hiện: Tự động thêm dấu, sửa lỗi chính tả, chuẩn hóa từ ngữ.
            - Nếu câu hỏi mơ hồ (dùng "nó", "họ", "sự kiện đó"), hãy dựa vào `LỊCH SỬ HỘI THOẠI` để điền ngữ cảnh còn thiếu.
            - Nếu câu hỏi không dùng tiếng Việt, hãy dịch sang tiếng Việt.
            - Nếu status là "greeting", hãy tạo một lời chào thân thiện.

            **BƯỚC 3: TRÍCH XUẤT THỰC THỂ TÌM KIẾM (`search_entity`)**
            - Áp dụng khi status là "ok".
            - Trích xuất 1 hay nhiều thực thể có trong câu hỏi (ví dụ như mối quan hệ giữa A và B thì thực thể lấy ra là A, B).
            - Từ `normalized_query`, chỉ trích xuất **thực thể chính**: tên nhân vật, triều đại, sự kiện, di sản.
            - **Quan trọng**: Bỏ hoàn toàn các từ để hỏi ("ai", "ở đâu"), tính từ, và mệnh đề mô tả ("nguyên nhân", "tóm tắt", "chi tiết", "thành công"). Mục tiêu là lấy ra từ khóa cốt lõi để tìm kiếm.
            - Nếu status là "greeting" hoặc "off_topic", để trường này là chuỗi rỗng "".

            **BƯỚC 4: ĐỊNH TUYẾN DỮ LIỆU (`category`)**
            - Áp dụng khi status là "ok".
            - Xác định loại bảng dữ liệu phù hợp nhất để truy vấn dựa trên `normalized_query`:
                - **MILITARY**: thông tin về các cuộc đấu tranh, chiến tranh, trận đánh (như "Trận Bạch Đằng", "Khởi nghĩa Lam Sơn").
                - **PERIOD**: thông tin tổng quan về các thời kỳ lịch sử lớn (như "Thời Bắc thuộc", "Thời kỳ phong kiến").
                - **STAGE**: thông tin về các giai đoạn nhỏ trong từng thời kỳ (ví dụ "Giai đoạn đầu nhà Trần").
                - **FIGURE**: thông tin về nhân vật lịch sử (như "Trần Hưng Đạo", "Nguyễn Trãi").
                - **HERITAGE**: thông tin về các di sản văn hóa (như "Lễ hội cầu mưa", "Nhã nhạc cung đình Huế").
                - **YEAR**: hỏi cụ thể năm nào đó (như "năm 937 xảy ra cái gì")
                - **COMPARE**: thông tin về so sánh cái gì đó (như so sánh sự khác nhau giữa nhà Ngô và nhà Lý)
            - **Quy tắc đặc biệt**: Nếu người dùng chỉ rõ tìm trong bảng nào (ví dụ: "...trong bảng Quân Sự"), hãy tuân thủ và trả về đúng loại bảng đó.
            - Nếu status là "greeting" hoặc "off_topic", để trường này là `null`.

            -----------------
            **--- VÍ DỤ MINH HỌA ---**

            Lịch sử: "Trần Hưng Đạo quê ở đâu"
            Câu hỏi hiện tại: "ông ấy sinh năm bao nhiêu"
            --> JSON Output:
            {{"status": "ok", "normalized_query": "Trần Hưng Đạo sinh năm bao nhiêu", "search_entity": "Trần Hưng Đạo", "category": "FIGURE"}}

            Câu hỏi hiện tại: "tìm kiếm nhân vật Trần Hưng Đạo trong bảng Quân Sự"
            --> JSON Output:
            {{"status": "ok", "normalized_query": "tìm kiếm nhân vật Trần Hưng Đạo trong bảng Quân Sự", "search_entity": "Trần Hưng Đạo", "category": "MILITARY"}}

            Câu hỏi hiện tại: "tóm tắt chi tiết giai đoạn bắc thyocoj lần 1"
            --> JSON Output:
            {{"status": "ok", "normalized_query": "tóm tắt chi tiết giai đoạn Bắc thuộc lần I", "search_entity": "Bắc thuộc lần I", "category": "PERIOD"}}

            Câu hỏi hiện tại: "Bạn thích uống cà phê không?"
            --> JSON Output:
            {{"status": "off_topic", "normalized_query": "Bạn thích uống cà phê không?", "search_entity": "", "category": null}}

            Câu hỏi hiện tại: "Bạn là ai?"
            --> JSON Output:
            {{"status": "greeting", "normalized_query": "Xin chào, tôi là trợ lý chuyên về lịch sử Việt Nam. Rất vui được giúp đỡ bạn.", "search_entity": "", "category": null}}
            -----------------

            **DỮ LIỆU ĐẦU VÀO:**

            LỊCH SỬ HỘI THOẠI GẦN ĐÂY:
            {history}

            CÂU HỎI HIỆN TẠI:
            {user_query}

            -----------------
            **YÊU CẦU OUTPUT:**
            Chỉ trả về MỘT đối tượng JSON hợp lệ duy nhất tuân thủ theo các quy tắc và ví dụ trên. Không thêm bất kỳ văn bản giải thích hay ký tự ```json nào khác.
            {{"status": .., "normalized_query": .., "search_entity": .., "category": ..}}
            """
        )

        self.transform_chain = transform_prompt | self.__llm_model | str_parser

        # --- Chain 2: Answer Generation ---
        # This chain synthesizes a final answer from a list of retrieved document chunks.
        # It is instructed to be accurate, cite sources, and handle cases where
        # no relevant information is found.
        answer_prompt = ChatPromptTemplate.from_template(
            """
            Bạn là một chuyên gia lịch sử Việt Nam. Nhiệm vụ của bạn là đọc kỹ các đoạn nội dung (chunk) dưới đây để trả lời câu hỏi của người dùng về lịch sử Việt Nam.

            Quy tắc bắt buộc:
            1. Mỗi chunk bao gồm:
               - chunk_index: vị trí chunk trong bài viết
               - Title: tiêu đề bài viết
               - URL: liên kết đến bài viết
               - Content: nội dung đoạn trích
            2. Các chunk có thể cùng thuộc một bài viết (cùng Title và URL). Nếu vậy, hãy tổng hợp các chunk này lại để tạo câu trả lời đầy đủ, tránh lặp lại.
            3. Nếu chunk nào không liên quan đến câu hỏi, hãy loại bỏ hoàn toàn chunk đó khỏi quá trình tổng hợp.
            4. Chỉ được sử dụng thông tin còn lại trong các chunk này để trả lời. Tuyệt đối không thêm hoặc bịa đặt thông tin ngoài dữ liệu đã cung cấp.
            5. Trả lời ngắn gọn, đúng trọng tâm câu hỏi. Tránh lan man, không thêm thông tin ngoài phạm vi câu hỏi nếu không cần thiết.
            6. Câu trả lợi dựa trên tài liệu tôi cung cấp thì không được ghi câu trả lời đó thuộc chunk nào.
            7. Nếu không có chunk nào chứa thông tin cần thiết để trả lời câu hỏi, hãy trả về JSON như sau:
               {{
                 "Answer": "Cảm ơn câu hỏi của bạn. Hiện tại, kho dữ liệu của tôi chưa có thông tin về chủ đề này. Tôi sẽ ghi nhận để bổ sung trong tương lai.",
                 "URL": None
               }}
            ------------------------------------
            *** LƯU Ý QUAN TRỌNG: KHỚP NGỮ CẢNH, KHÔNG CHỈ KHỚP TÊN ***

            Phải đảm bảo chunk nói về đúng nhân vật trong đúng vai trò/bối cảnh được hỏi.
            Ví dụ: Câu hỏi về "Thầy Vũ" sẽ không được trả lời bằng thông tin về "Tướng quân Vũ".

            Nếu không có chunk nào khớp đúng ngữ cảnh, hãy áp dụng quy tắc 5.   
            ------------------------------------
            Yêu cầu kết quả:
            - Trả lời dưới dạng JSON với hai trường:
            {{
              "Answer": "Câu trả lời ngắn gọn, chính xác, đúng trọng tâm câu hỏi.",
              "URL": "danh sách URL của bài viết liên quan đến câu trả lời, đảm bảo các URL không trùng lặp (duy nhất)."
            }}
            - Nếu câu trả lời tổng hợp từ nhiều chunk nhưng cùng một bài viết, chỉ cần ghi lại Title và URL của bài đó.
            - Không liệt kê lại nguyên văn các chunk.
            ------------------------------------

            Dữ liệu cung cấp:
            {list_chunks}

            ------------------------------------
            Câu hỏi của người dùng:
            {question}

            ------------------------------------
            Hãy phân tích kỹ các chunk, loại bỏ các chunk không liên quan, sau đó trả lời theo đúng định dạng JSON ở trên.
            """
        )

        self.answer_chain = answer_prompt | self.__llm_model | str_parser

    def query_transform(self, user_query: str, history: ConversationHistory) -> Dict[str, Any]:
        """
        Analyzes and transforms a raw user query using Gemini LLM.

        The method:
            - Uses chat history to resolve ambiguity.
            - Classifies query type (greeting, off-topic, valid).
            - Normalizes and cleans user input.
            - Extracts core entity (e.g., names, events, periods).
            - Identifies search category (e.g., FIGURE, MILITARY, etc.).

        Args:
            user_query (str): Raw input from the user.
            history (ConversationHistory): The ongoing dialogue history object.

        Returns:
            Dict[str, Any]: A structured dictionary:
                {
                    "status": "ok" | "greeting" | "off_topic",
                    "normalized_query": "...",
                    "search_entity": "...",
                    "category": "FIGURE" | "PERIOD" | "HERITAGE" | ... | null
                }
            If an error occurs or query is off-topic, a default fallback structure is returned.
        """

        try:
            response = self.transform_chain.invoke({
                "history": history.format_for_prompt(),
                "user_query": user_query
            })

            cleaned_response = re.sub(r"```(?:json)?|```", "", response).strip()

            result_dict = json.loads(cleaned_response)

            return result_dict

        except Exception as e:
            print(f"[combine_query] Error: {e}")
            return {"status": "off_topic", "normalized_query": user_query, "search_entity": "", "category": None}

    def generate_answer_question(self, user_query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a contextual answer using retrieved document chunks.

        This method:
            - Formats document chunks with metadata (Title, URL, Content).
            - Sends them with the user query to the LLM.
            - Asks the LLM to filter irrelevant content and synthesize a short answer.
            - Enforces strict grounding rules: do not invent facts.

        Args:
            user_query (str): The original user question.
            docs (List[Dict[str, Any]]): List of chunks with required fields:
                {
                    "Chuck_Index": int,
                    "Title": str,
                    "URL": str,
                    "Content": str
                }

        Returns:
            Dict[str, Any]: A JSON-like dictionary with:
                {
                    "Answer": "The short, focused answer to the question.",
                    "URL": ["source_url_1", "source_url_2", ...] | None
                }

            If no valid information is found or an error occurs, a friendly fallback answer is returned.
        """
        if not docs:
            return {
                "Answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý phản hồi từ hệ thống. Vui lòng thử lại.",
                "URL": None
            }

        list_chunks_str = "[\n" + ",\n".join([
            json.dumps({
                "chunk_index": doc["Chuck_Index"],
                "Title": doc["Title"],
                "URL": doc["URL"],
                "Content": doc["Content"]
            }, ensure_ascii=False)
            for doc in docs
        ]) + "\n]"

        try:
            response_str = self.answer_chain.invoke({
                "list_chunks": list_chunks_str,
                "question": user_query
            })

            cleaned_response = re.sub(r"```(?:json)?|```", "", response_str).strip()

            result_dict = json.loads(cleaned_response)

            return result_dict
        except json.JSONDecodeError:
            return {
                "Answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý phản hồi từ hệ thống. Vui lòng thử lại.",
                "URL": None
            }
        except Exception as e:
            print(f"[generate_answer_fact_question] Error: {e}")
            return {
                "Answer": "Xin lỗi, tôi không thể tạo câu trả lời vào lúc này. Vui lòng thử lại.",
                "URL": None
            }
