# intelligent_rag_system/src/rag_system/rag_pipeline.py
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import List, Dict, Any
from src.query_understanding.ner_extractor import NERExtractor
from src.query_understanding.query_rewriter import QueryRewriter
# from src.query_understanding.intent_recognizer import IntentRecognizer # REMOVED

from src.retrieval.retrievers import EnterpriseRetriever
from src.llm_generation.answer_synthesizer import AnswerSynthesizer
from src.llm_generation.conversation_manager import ConversationManager
from src.ingestion.vector_store import ChromaDBManager
from src.ingestion.embeddings import EmbeddingModelManager

class RAGPipeline:
    """
    Orchestrates the entire RAG process, unified to use the vector store
    for all data retrieval (structured and unstructured data treated uniformly).
    The LLM is responsible for extracting specific details from the retrieved text.
    """
    def __init__(self):
        print("Initializing RAGPipeline components...")
        self.ner_extractor = NERExtractor()
        self.query_rewriter = QueryRewriter()
        # self.intent_recognizer = IntentRecognizer() # REMOVED

        # Initialize core managers for retrieval (ONLY vector store related)
        self.chroma_db_manager = ChromaDBManager()
        self.embedding_model_manager = EmbeddingModelManager()
        
        self.retriever = EnterpriseRetriever(
            chroma_db_manager=self.chroma_db_manager,
            embedding_model_manager=self.embedding_model_manager
        )
        self.answer_synthesizer = AnswerSynthesizer()
        self.conversation_manager = ConversationManager(k=5) # Keep last 5 turns of history
        print("RAGPipeline initialized.")

    def run_pipeline(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the RAG pipeline for a given user query. All data retrieval
        is unified through the vector store, without explicit intent-based routing.

        Args:
            user_query: The raw query from the user.

        Returns:
            A dictionary containing the generated answer, source nodes, and other metadata.
        """
        print(f"\n--- RAG Pipeline Started for Query: '{user_query}' ---")

        # 1. Get current chat history for query rewriting
        chat_history_raw = self.conversation_manager.get_chat_history_raw()

        # 2. Query Understanding: Rewrite Query
        # Contextualizes the query based on chat history.
        rewritten_query = self.query_rewriter.rewrite_query(user_query, chat_history=chat_history_raw)

        # 3. Query Understanding: Extract Entities (still useful for context or future enhancements)
        # These extracted entities can be logged or potentially passed to the synthesizer's prompt
        # if the LLM needs more explicit guidance based on recognized entities.
        extracted_entities = self.ner_extractor.extract_entities(rewritten_query)
        print(f"Extracted entities: {extracted_entities}")

        # 4. Retrieval (Unified): Always perform hybrid search on the vector store
        print(f"Performing unified hybrid retrieval from vector store for query: '{rewritten_query}'")
        retrieved_nodes = self.retriever.retrieve(rewritten_query)

        # 5. Answer Synthesis
        print("Synthesizing final answer...")
        
        # The prompt in AnswerSynthesizer is now solely responsible for guiding the LLM
        # to extract and format specific details from the retrieved text nodes.
        final_answer = self.answer_synthesizer.synthesize_answer(
            user_query, # Use original query for synthesis
            retrieved_nodes=retrieved_nodes
        )

        # 6. Update conversation history
        self.conversation_manager.add_message("human", user_query)
        self.conversation_manager.add_message("ai", final_answer)

        print("--- RAG Pipeline Finished ---")
        return {
            "answer": final_answer,
            "source_nodes": [{"text": node.text, "metadata": node.metadata, "score": node.score} for node in retrieved_nodes],
            "rewritten_query": rewritten_query,
            "extracted_entities": extracted_entities,
            "chat_history": chat_history_raw
        }

# Example Usage
if __name__ == "__main__":
    # IMPORTANT: Ensure your ChromaDB is running and seeded with ALL your data
    # (including data from CSV/JSON files, which were loaded as Documents).
    # 1. Ensure you have run: `python run_ingestion.py` (from project root)
    # 2. Set your GOOGLE_API_KEY in the .env file.
    # Then run this script: `python src/rag_system/rag_pipeline.py`

    rag_pipeline = RAGPipeline()

    print("\n--- Running Sample Queries (All hitting unified vector store) ---")

    # Test 1: General policy search
    response1 = rag_pipeline.run_pipeline("What are the key points of the company's remote work policy regarding international travel?")
    print(f"\nResponse: {response1['answer']}")
    print(f"Sources: {len(response1['source_nodes'])} nodes.")

    # Test 2: Product information search (from what was CSV/JSON, now just text in vector store)
    response2 = rag_pipeline.run_pipeline("Tell me about the features and price of the Cloud Storage Premium product.")
    print(f"\nResponse: {response2['answer']}")
    print(f"Sources: {len(response2['source_nodes'])} nodes.")

    # Test 3: Employee information search (from what was CSV/JSON, now just text in vector store)
    response3 = rag_pipeline.run_pipeline("Who is Alice Smith and what department is she in? What's her role?")
    print(f"\nResponse: {response3['answer']}")
    print(f"Sources: {len(response3['source_nodes'])} nodes.")

    # Test 4: FAQ search (from what was TXT/MD, now just text in vector store)
    response4 = rag_pipeline.run_pipeline("What is the process for submitting an expense report?")
    print(f"\nResponse: {response4['answer']}")
    print(f"Sources: {len(response4['source_nodes'])} nodes.")

    # Test 5: Hybrid query (product + policy) - LLM will parse both from text nodes
    response5 = rag_pipeline.run_pipeline("What are the features of Enterprise AI Platform and what is our policy on data security?")
    print(f"\nResponse: {response5['answer']}")
    print(f"Sources: {len(response5['source_nodes'])} nodes.")

    # Test 6: Multi-turn (history should influence query)
    print("\n--- Multi-turn conversation test ---")
    _ = rag_pipeline.run_pipeline("What is Product A?") # Assuming Product A was indexed
    response6 = rag_pipeline.run_pipeline("Tell me more about its key features and availability.")
    print(f"\nResponse: {response6['answer']}")
    print(f"Rewritten Query (influenced by history): {response6['rewritten_query']}")
    print(f"Sources: {len(response6['source_nodes'])} nodes.")