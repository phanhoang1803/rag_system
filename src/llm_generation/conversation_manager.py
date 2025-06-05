# intelligent_rag_system/src/llm_generation/conversation_manager.py

from typing import List, Dict, Any
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

class ConversationManager:
    """
    Manages conversation history for multi-turn interactions.
    Uses LangChain's ConversationBufferWindowMemory for a sliding window of messages.
    """
    def __init__(self, k: int = 5):
        """
        Initializes the conversation manager with a window of 'k' turns.
        Args:
            k: The number of previous turns to keep in memory.
        """
        self.memory = ConversationBufferWindowMemory(k=k, return_messages=True)
        print(f"ConversationManager initialized with a window of {k} turns.")

    def add_message(self, role: str, content: str):
        """
        Adds a message to the conversation history.
        Args:
            role: 'human' for user messages, 'ai' for assistant messages.
            content: The text content of the message.
        """
        if role == "human":
            self.memory.chat_memory.add_user_message(content)
        elif role == "ai":
            self.memory.chat_memory.add_ai_message(content)
        else:
            print(f"Warning: Invalid role '{role}'. Message not added to history.")

    def get_chat_history_for_llm(self) -> List[BaseMessage]:
        """
        Retrieves the formatted chat history suitable for an LLM prompt.
        """
        # self.memory.load_memory_variables({}) returns a dict like {'history': [msgs]}
        return self.memory.load_memory_variables({})["history"]

    def get_chat_history_raw(self) -> List[Dict[str, str]]:
        """
        Retrieves the raw chat history as a list of dictionaries.
        Useful for passing to the query rewriter or for displaying in UI.
        """
        raw_history = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                raw_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                raw_history.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                raw_history.append({"role": "system", "content": msg.content})
        return raw_history

    def clear_history(self):
        """Clears the entire conversation history."""
        self.memory.clear()
        print("Conversation history cleared.")

# Example Usage
if __name__ == "__main__":
    cm = ConversationManager(k=3) # Keep last 3 turns

    cm.add_message("human", "Hi there!")
    cm.add_message("ai", "Hello! How can I help you today?")
    print(f"\nHistory (LLM format, 1 turn): {cm.get_chat_history_for_llm()}")
    print(f"History (Raw format, 1 turn): {cm.get_chat_history_raw()}")

    cm.add_message("human", "What is Product A?")
    cm.add_message("ai", "Product A is a cloud storage solution.")
    print(f"\nHistory (LLM format, 2 turns): {cm.get_chat_history_for_llm()}")
    print(f"History (Raw format, 2 turns): {cm.get_chat_history_raw()}")

    cm.add_message("human", "Tell me more about its features.")
    cm.add_message("ai", "It offers 1TB storage and monthly backups.")
    print(f"\nHistory (LLM format, 3 turns - should show last 3): {cm.get_chat_history_for_llm()}")
    print(f"History (Raw format, 3 turns - should show last 3): {cm.get_chat_history_raw()}")

    cm.add_message("human", "What about its pricing?")
    cm.add_message("ai", "Product A costs $10/month.")
    print(f"\nHistory (LLM format, 4 turns - should show last 3): {cm.get_chat_history_for_llm()}")
    print(f"History (Raw format, 4 turns - should show last 3): {cm.get_chat_history_raw()}")

    cm.clear_history()
    print(f"\nHistory after clearing: {cm.get_chat_history_for_llm()}")