from langchain.memory import ConversationBufferMemory

"""
    A simple conversation history manager that wraps LangChain's ConversationBufferMemory.

    This class provides a simplified interface to track user and AI messages.
    Its primary purpose is to maintain conversational context and provide a
    formatted history string that can be injected into LLM prompts. This allows
    the AI to be aware of the recent dialogue, enabling follow-up questions and
    contextual understanding.
"""


class ConversationHistory:

    def __init__(self):
        """
        Initializes the ConversationHistory object and its underlying memory buffer.
        """
        print("[ConversationHistory] Initializing ConversationHistory...")
        self.__memory = ConversationBufferMemory()
        print("[ConversationHistory] ConversationHistory initialized.")

    def add_user_message(self, message: str):
        """
        Adds a message from the user to the conversation history.

        Args:
            message (str): The user's input message to be stored.
        """
        self.__memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message: str):
        """
        Adds a message from the AI to the conversation history.

        Args:
            message (str): The AI's response message to be stored.
        """
        self.__memory.chat_memory.add_ai_message(message)

    def format_for_prompt(self, max_turns: int = 3) -> str:
        """
        Retrieves recent exchanges and formats them as a string for prompt injection.

        This method extracts the last `max_turns` of the conversation, where one
        turn consists of a user message and an AI response. The output is a clean,
        multi-line string perfect for providing context to an LLM.

        Args:
            max_turns (int): The number of recent conversational turns (user + AI pairs)
                             to include. Defaults to 3.

        Returns:
            str: A formatted string of the recent conversation history.

        Example:
            If the history contains several turns, and `max_turns=2`, the output
            might look like:

            User: Who led the Bach Dang battle?
            Assistant: It was General Tran Hung Dao.
            User: When did it happen?
            Assistant: The battle took place in 1288.
        """
        # Get the last N messages. Each turn has 2 messages (human, ai).
        messages = self.__memory.chat_memory.messages[-max_turns * 2:]
        formatted = []
        for msg in messages:
            # Assign a role based on the message type
            role = "User" if msg.type == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    def clear(self):
        """
        Clears all messages from the conversation history, resetting the memory.
        """
        self.__memory.clear()

