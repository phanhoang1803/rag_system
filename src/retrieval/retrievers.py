# intelligent_rag_system/src/retrieval/retrievers.py
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank

from src.ingestion.vector_store import ChromaDBManager
from src.ingestion.embeddings import EmbeddingModelManager

from config.settings import settings

class EnterpriseRetriever:
    """
    Manages vector store retrieval strategies including vector search, keyword search,
    and hybrid search. All RAG data is now sourced uniformly from the vector store.
    """
    def __init__(self,
                 chroma_db_manager: ChromaDBManager,
                 embedding_model_manager: EmbeddingModelManager,
                 top_k: int = settings.TOP_K_RETRIEVAL):

        self.chroma_db_manager = chroma_db_manager
        self.top_k = top_k

        # Initialize LlamaIndex VectorStoreIndex for all text data
        self.vector_index: VectorStoreIndex = self._initialize_vector_index(embedding_model_manager)

        # Initialize individual retrievers
        self.vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=self.top_k
        )
        print("Vector Retriever initialized.")

        self.bm25_retriever = self._initialize_bm25_retriever()
        print("BM25 Retriever initialized.")

        # Initialize re-ranker
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", # A good general-purpose cross-encoder
            top_n=self.top_k # Re-rank and take top_n results
        )
        print("Re-ranker initialized.")


    def _initialize_vector_index(self, embedding_model_manager: EmbeddingModelManager) -> VectorStoreIndex:
        """Helper to initialize the LlamaIndex VectorStoreIndex."""
        embed_model = embedding_model_manager.get_embedding_model()
        storage_context = self.chroma_db_manager.get_storage_context()
        return VectorStoreIndex.from_vector_store(
            vector_store=storage_context.vector_store,
            embed_model=embed_model,
        )

    def _initialize_bm25_retriever(self) -> BM25Retriever:
        """
        Helper to initialize the BM25 Retriever by loading all existing nodes from ChromaDB.
        This can be memory intensive for very large collections; consider alternatives
        for production-scale BM25 if the full dataset doesn't fit in memory.
        """
        print("Loading all text nodes from ChromaDB for BM25 indexing...")
        all_content = self.chroma_db_manager.chroma_collection.get(
            ids=self.chroma_db_manager.chroma_collection.get()['ids'],
            include=['documents', 'metadatas']
        )
        bm25_documents = []
        if all_content and 'documents' in all_content:
            # Reconstruct LlamaIndex Document-like objects from Chroma's raw dict output
            for i, doc_text in enumerate(all_content['documents']):
                # Chroma stores metadata as dict, LlamaIndex expects it as dict
                bm25_documents.append(Document(text=doc_text, metadata=all_content['metadatas'][i]))
        
        if not bm25_documents:
            print("No documents found in ChromaDB for BM25 initialization. BM25 will be empty.")
            # Return a BM25Retriever with empty nodes to prevent errors
            return BM25Retriever(nodes=[]) 

        return BM25Retriever.from_defaults(nodes=bm25_documents, similarity_top_k=min(self.top_k, len(bm25_documents)))


    def retrieve(self, query: str) -> List[NodeWithScore]:
        """
        Performs a unified hybrid (vector + keyword) search across all indexed documents
        and re-ranks results. This is now the ONLY retrieval method for RAG data.
        """
        print(f"Performing unified hybrid retrieval for query: '{query}'")

        vector_results = self.vector_retriever.retrieve(query)
        bm25_results = self.bm25_retriever.retrieve(query)

        # Combine and de-duplicate results based on node ID
        combined_results_map = {node.node.id_: node for node in vector_results + bm25_results}
        combined_results = list(combined_results_map.values())

        print(f"Combined {len(vector_results)} vector results and {len(bm25_results)} BM25 results into {len(combined_results)} unique results.")

        # Re-rank the combined results
        re_ranked_nodes = self.reranker.postprocess_nodes(combined_results, query_str=query)
        print(f"Re-ranked to {len(re_ranked_nodes)} top results.")

        return re_ranked_nodes

# Example Usage (Requires ChromaDB to be running and seeded with all data)
if __name__ == "__main__":
    from config.settings import settings
    import os
    import shutil
    from pathlib import Path
    from llama_index.core.schema import Document

    # Ensure settings are accessible for API key and paths
    # Set a dummy API key for local testing if not in .env
    if not settings.GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not found in settings. Using a placeholder for example. Set it in .env for actual use.")
        settings.GOOGLE_API_KEY = "YOUR_DUMMY_API_KEY" # This will likely fail API calls

    # Ensure a dummy ChromaDB directory exists for testing
    temp_chroma_dir = Path("./temp_chroma_test_data")
    if temp_chroma_dir.exists():
        shutil.rmtree(temp_chroma_dir)
    os.makedirs(temp_chroma_dir)
    
    # Create a temporary ChromaDBManager for this test (not the actual one from run_ingestion)
    # The ChromaDBManager should point to the actual persisted data for real usage.
    # For this isolated test, we'll quickly create and add a node.
    temp_chroma_mgr = ChromaDBManager(persist_directory=str(temp_chroma_dir), collection_name="test_collection_retriever")
    temp_chroma_mgr.initialize_vector_store()
    
    # Create a dummy embedding model for this test
    temp_embed_model_mgr = EmbeddingModelManager()

    # Add a dummy node to the temporary ChromaDB for retrieval testing
    dummy_node_text = "This is a dummy document about company policies. It mentions remote work and benefits. Also, there's information about Product X with its price $10 and features."
    dummy_node_id = "dummy_policy_product_1"
    
    # Get the embedding for the dummy node
    try:
        dummy_embedding = temp_embed_model_mgr.get_embedding_model().get_text_embedding(dummy_node_text)
    except Exception as e:
        print(f"Could not get embedding for dummy node. Ensure GOOGLE_API_KEY is valid. Error: {e}")
        dummy_embedding = [0.0] * 768 # Fallback to dummy embedding if API fails

    dummy_doc_node = Document(
        text=dummy_node_text,
        metadata={"file_name": "dummy_policy_product.txt", "file_type": "txt"},
        id_=dummy_node_id,
        embedding=dummy_embedding # Assign embedding to node
    )
    # ChromaDB add expects a list of LlamaIndex BaseNode objects, but we're simulating a raw doc.
    # We need to manually add it to the collection for the BM25 retriever to pick it up.
    temp_chroma_mgr.chroma_collection.add(
        documents=[dummy_node_text],
        metadatas=[dummy_doc_node.metadata],
        ids=[dummy_node_id],
        embeddings=[dummy_embedding]
    )
    print(f"Added dummy node to temporary ChromaDB. Count: {temp_chroma_mgr.chroma_collection.count()}")


    print("Initializing EnterpriseRetriever for testing...")
    retriever = EnterpriseRetriever(chroma_db_manager=temp_chroma_mgr,
                                    embedding_model_manager=temp_embed_model_mgr)

    print("\n--- Testing Unified Retrieval (Hybrid Search for all data types) ---")
    
    queries = [
        "Tell me about remote work policies.",
        "What are the features and price of Product X?",
        "What is the company's benefits plan?" # This will likely return relevant sections from the dummy doc
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.retrieve(query)
        print(f"Retrieved {len(results)} nodes for '{query}':")
        for i, node_with_score in enumerate(results):
            file_info = node_with_score.metadata.get('file_name') or node_with_score.metadata.get('file_path', 'N/A')
            print(f"  {i+1}. Score: {node_with_score.score:.4f}, Content: {node_with_score.text[:100]}..., Source: {file_info}, Type: {node_with_score.metadata.get('file_type', 'N/A')}")

    # Clean up temporary ChromaDB data
    if temp_chroma_dir.exists():
        shutil.rmtree(temp_chroma_dir)
        print(f"Cleaned up temporary ChromaDB data at: {temp_chroma_dir}")
