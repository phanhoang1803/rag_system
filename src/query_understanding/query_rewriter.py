# intelligent_rag_system/src/query_understanding/query_rewriter.py

"""
This module contains the QueryRewriter class, which rewrites user queries to improve retrieval.
"""
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, Any, List

from config.settings import settings

class QueryRewriter:
    """
    Rewrites or expands user queries for better retrieval using an LLM.
    Uses LangChain's Runnable interface for a modular and chainable approach.
    """
    def __init__(self,
                 model_name: str = settings.LLM_MODEL_NAME,
                 api_key: str = settings.GOOGLE_API_KEY,
                 ):
        self.llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                
                """
                You are an expert query rewriter. Your goal is to improve the user's query 
                to be more effective for information retrieval from a knowledge base. 
                Consider synonyms, broader terms, and implicit context. 
                If there's chat history, use it to make the current query more specific. 
                Only output the revised query, nothing else.
                """
            ),
            (
                "user",
                """
                Original Query: {original_query}
                Chat History: {chat_history}
                Revised Query:
                """
            )
        ])
        self.rewriter_chain = (
            {"original_query": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        print("QueryRewriter initialized.")
        
    def rewrite_query(self, original_query : str, chat_history : List[Dict[str, str]] = None) -> str:
        """
        Rewrites the original query, potentially incorporating chat history.

        Args:
            original_query: The user's initial query.
            chat_history: A list of dicts like [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                          representing the conversation history.

        Returns:
            The rewritten or expanded query string.
        """
        if chat_history is None:
            chat_history_str = "No chat history."
        else:
            # Format chat history as a string
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
            if not chat_history_str:
                chat_history_str = "No meaningful chat history."
        
        print(f"Rewriting query: '{original_query}' with history: '{chat_history_str[:50]}...'")
        
        # Invoke the chain. We pass a dictionary with 'original_query' and 'chat_history'
        # The RunnablePassthrough will correctly map these to the prompt template.
        rewritten_query = self.rewriter_chain.invoke({
            "original_query": original_query,
            "chat_history": chat_history_str
        })
        print(f"Rewritten query: '{rewritten_query}'")
        return rewritten_query.strip()

# Example Usage
if __name__ == "__main__":
    query_rewriter = QueryRewriter()

    # Case 1: No chat history
    query1 = "company policy for holidays"
    rewritten_query1 = query_rewriter.rewrite_query(query1)
    print(f"Original: '{query1}' -> Rewritten: '{rewritten_query1}'\n")

    # Case 2: With chat history
    chat_history = [
        {"role": "user", "content": "Tell me about Product A."},
        {"role": "assistant", "content": "Product A is a cloud storage solution with 1TB capacity."},
        {"role": "user", "content": "What about its premium version?"}
    ]
    query2 = "What about its premium version?"
    rewritten_query2 = query_rewriter.rewrite_query(query2, chat_history)
    print(f"Original: '{query2}' with history -> Rewritten: '{rewritten_query2}'\n")

    # Case 3: Short and ambiguous query
    query3 = "payroll"
    rewritten_query3 = query_rewriter.rewrite_query(query3)
    print(f"Original: '{query3}' -> Rewritten: '{rewritten_query3}'\n")