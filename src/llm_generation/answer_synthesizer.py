# intelligent_rag_system/src/llm_generation/answer_synthesizer.py
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any, Optional
from llama_index.core.schema import NodeWithScore, TextNode
from config.settings import settings

class AnswerSynthesizer:
    """
    Synthesizes a final answer using an LLM based on retrieved context (all as text nodes)
    and user query. It's designed to extract specific information from unstructured text
    that might have originated from structured sources (CSV, JSON).
    """
    def __init__(self, model_name: str = settings.LLM_MODEL_NAME, api_key: str = settings.GOOGLE_API_KEY):
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1)

        # CRITICAL: The system prompt is now highly detailed on how to extract and format information
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
             You are a highly capable and precise enterprise RAG assistant.
             Your goal is to answer the user's question accurately and concisely, 
             based ONLY on the provided context. If the context does not contain 
             enough information to answer the question, state that you don't know 
             or that the information is not available in the provided context. 
             Do NOT make up answers.
             Context:\n{context_str}\n\n
             Instructions:\n
             - Provide clear, direct answers.\n
             - **Extract specific details from the text:** If the query is about a product (e.g., price, features, ID), an employee (e.g., department, role, contact), or an FAQ, search the context carefully for these facts. 
             - **Format extracted details cleanly:**\n
             - For **Product information**: List details like 'Name: [Name], ID: [ID], Price: [Price], Features: [List of features]'.\n
             - For **Employee information**: List details like 'Name: [Name], Department: [Department], Role: [Role], Email: [Email]'.\n
             - For **FAQ answers**: Provide the direct answer found in the context.\n
             - For **General information**: Summarize or directly answer from the text.\n
             - If multiple pieces of information are relevant, present them clearly.\n
             - Cite sources by their 'file_name' or 'file_path' (and 'page_label' or 'row_idx' if available) when possible. Prioritize 'file_name' if both are available.\n
             - Avoid conversational filler like 'Based on the context provided...'.\n
             - Be brief and to the point."
             """
            ),
            (
            "user", 
            """Question: {query}"""
            )
        ])
        self.synthesis_chain = self.synthesis_prompt | self.llm | StrOutputParser()
        print("AnswerSynthesizer initialized.")

    def _format_context(self, nodes: List[NodeWithScore]) -> str:
        """Formats text nodes into a single string for the prompt."""
        formatted_context = []
        for i, node in enumerate(nodes):
            metadata_parts = []
            if node.metadata.get('file_name'):
                metadata_parts.append(f"File: {node.metadata['file_name']}")
            elif node.metadata.get('file_path'): # Fallback to file_path if name not present
                metadata_parts.append(f"Source: {node.metadata['file_path']}")
            if node.metadata.get('page_label'):
                metadata_parts.append(f"Page: {node.metadata['page_label']}")
            if node.metadata.get('row_idx') is not None: # For CSV/JSON rows, handle 0
                metadata_parts.append(f"Row: {node.metadata['row_idx']}")
            if node.metadata.get('file_type'): # E.g., 'csv', 'json', 'html', 'md', 'txt', 'pdf'
                 metadata_parts.append(f"Type: {node.metadata['file_type'].upper()} Document")

            metadata_str = ", ".join(metadata_parts) if metadata_parts else "Source: Unknown Document"
            formatted_context.append(f"--- Context Node {i+1} ({metadata_str}) ---\n{node.text}\n")
        return "\n".join(formatted_context)

    def synthesize_answer(self,
                          query: str,
                          retrieved_nodes: Optional[List[NodeWithScore]] = None) -> str:
        """
        Synthesizes an answer based on the query and provided retrieved nodes.
        All context is now expected to come from vector store nodes.

        Args:
            query: The user's original question.
            retrieved_nodes: List of NodeWithScore objects from vector/keyword search.

        Returns:
            The synthesized answer string.
        """
        full_context_str = ""

        if retrieved_nodes:
            full_context_str = self._format_context(retrieved_nodes)
        
        if not full_context_str.strip():
            full_context_str = "No relevant context found in the knowledge base."
            print("No relevant context provided to synthesizer.")

        print(f"Synthesizing answer for query: '{query}' with context (first 200 chars): {full_context_str[:200]}...")

        try:
            answer = self.synthesis_chain.invoke({
                "context_str": full_context_str,
                "query": query
            })
            return answer.strip()
        except Exception as e:
            print(f"Error during answer synthesis: {e}")
            return "An error occurred while generating the answer. Please try again."

# Example Usage
if __name__ == "__main__":
    from config.settings import settings # Ensure settings are accessible for API key

    synthesizer = AnswerSynthesizer()

    # Example Nodes (could be from any source: MD, PDF, CSV, JSON, HTML)
    # The key is how the data is represented as text within the Document/Node
    
    # Simulating a product from a CSV or JSON
    product_node = TextNode(
        text="Product: Cloud Storage Premium, ID: PRD002, Price: $25.00/month, Features: 2TB capacity, real-time sync, versioning, 99.9% uptime SLA.", 
        metadata={"file_name": "products.csv", "row_idx": 1, "file_type": "csv"}
    )
    dummy_product_node = NodeWithScore(
        node=product_node,
        score=0.9, 
    )
    
    # Simulating an employee from a CSV or JSON
    employee_node = TextNode(
        text="Employee Name: Alice Smith, Employee ID: EMP001, Department: Human Resources, Role: HR Manager, Email: alice.smith@example.com, Phone: 555-1234.", 
        metadata={"file_name": "employees.json", "file_type": "json"}
    )
    dummy_employee_node = NodeWithScore(
        node=employee_node,
        score=0.88, 
    )

    # Simulating a policy from a Markdown/Text file
    policy_node = TextNode(
        text="The company's remote work policy states that employees must maintain a safe and productive workspace at their approved remote location. International travel for remote work requires prior HR approval for stays exceeding 30 days.",
        metadata={"file_name": "remote_work_policy.md", "page_label": "1", "file_type": "md"}
    )
    dummy_policy_node = NodeWithScore(
        node=policy_node,
        score=0.85, 
    )
    
    # Simulating an FAQ from a Text file
    faq_node = TextNode(
        text="Question: How do I reset my password? Answer: You can reset your password by visiting the IT self-service portal at portal.example.com/password-reset. Follow the 'Forgot Password' link.",
        metadata={"file_name": "it_faqs.txt", "file_type": "txt"}
    )
    dummy_faq_node = NodeWithScore(
        node=faq_node,
        score=0.92, 
    )

    # All nodes are now treated as 'unstructured context' by the synthesizer
    all_dummy_nodes = [dummy_product_node, dummy_employee_node, dummy_policy_node, dummy_faq_node]

    # Test Case 1: Product information query
    query1 = "What are the features and price of Cloud Storage Premium?"
    answer1 = synthesizer.synthesize_answer(query1, retrieved_nodes=[dummy_product_node])
    print(f"\nQuery: '{query1}'\nAnswer: {answer1}")

    # Test Case 2: Employee information query
    query2 = "Tell me about Alice Smith and her contact details."
    answer2 = synthesizer.synthesize_answer(query2, retrieved_nodes=[dummy_employee_node])
    print(f"\nQuery: '{query2}'\nAnswer: {answer2}")

    # Test Case 3: General policy query
    query3 = "What is the remote work policy regarding international travel?"
    answer3 = synthesizer.synthesize_answer(query3, retrieved_nodes=[dummy_policy_node])
    print(f"\nQuery: '{query3}'\nAnswer: {answer3}")

    # Test Case 4: FAQ query
    query4 = "How do I reset my password?"
    answer4 = synthesizer.synthesize_answer(query4, retrieved_nodes=[dummy_faq_node])
    print(f"\nQuery: '{query4}'\nAnswer: {answer4}")

    # Test Case 5: Hybrid query, relying on LLM to parse from combined context
    query5 = "What are the features of Cloud Storage Premium, and what is the policy on international travel for remote work?"
    answer5 = synthesizer.synthesize_answer(query5, retrieved_nodes=[dummy_product_node, dummy_policy_node])
    print(f"\nQuery: '{query5}'\nAnswer: {answer5}")

    # Test Case 6: No relevant context
    query6 = "What is the capital of France?"
    answer6 = synthesizer.synthesize_answer(query6, retrieved_nodes=[])
    print(f"\nQuery: '{query6}' (no context)\nAnswer: {answer6}")