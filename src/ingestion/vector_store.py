# intelligent_rag_system/src/ingestion/vector_store.py
from pathlib import Path
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from src.ingestion.embeddings import EmbeddingModel
import chromadb
from config.settings import settings
from typing import List

class ChromaDBManager:
    """
    Manages the ChromaDB vector store.
    Adheres to SRP by handling only vector database interactions.
    """
    def __init__(self, embedding_model_manager: EmbeddingModel,
                 persist_directory: str = settings.CHROMA_PERSIST_DIR,
                 collection_name: str = settings.CHROMA_COLLECTION_NAME,
                 ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chroma_client = None
        self.chroma_collection = None
        self.vector_store = None
        self.storage_context = None
        self.embed_model = embedding_model_manager.get_embedding_model()
        
    def initialize_vector_store(self) -> ChromaVectorStore:
        """
        Initializes the ChromaDB client, collection, and LlamaIndex VectorStore.
        """
        print(f"Initializing ChromaDB at: {self.persist_directory} for collection: {self.collection_name}")
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create the collection
        self.chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
        print(f"ChromaDB collection '{self.collection_name}' initialized. Current count: {self.chroma_collection.count()}")

        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        return self.vector_store
    
    def _generate_embeddings_for_nodes(self, nodes: List[TextNode]) -> List[TextNode]:
        """Generate embeddings for nodes using the embedding model."""
        if not self.embed_model:
            raise ValueError("Embedding model not set. Use set_embedding_model() first.")
        
        print(f"Generating embeddings for {len(nodes)} nodes...")
        for node in nodes:
            if not node.embedding:  # Only generate if embedding doesn't exist
                embedding = self.embed_model.get_text_embedding(node.get_content())
                node.embedding = embedding
        
        print(f"Embeddings generated for {len(nodes)} nodes.")
        return nodes
    
    def add_nodes(self, nodes: List[TextNode]):
        """Adds a list of TextNode objects to the ChromaDB vector store."""
        if not self.vector_store:
            self.initialize_vector_store() # Ensure it's initialized

        # Generate embeddings for nodes
        nodes_with_embeddings = self._generate_embeddings_for_nodes(nodes)

        print(f"Adding {len(nodes_with_embeddings)} nodes to ChromaDB...")
        try:
            self.vector_store.add(nodes_with_embeddings)
            print(f"Successfully added {len(nodes_with_embeddings)} nodes to ChromaDB.")
            print(f"New total ChromaDB count: {self.chroma_collection.count()}")
        except Exception as e:
            print(f"Error adding nodes to ChromaDB: {e}")
            raise
        
    def get_storage_context(self) -> StorageContext:
        """Returns the LlamaIndex StorageContext for the vector store."""
        if not self.storage_context:
            self.initialize_vector_store()
        return self.storage_context
    
    def clear_collection(self):
        """Clears all data from the ChromaDB collection."""
        if not self.chroma_client:
            self.initialize_vector_store() # Ensure client is loaded
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"ChromaDB collection '{self.collection_name}' cleared.")
            # Re-create it immediately
            self.chroma_collection = self.chroma_client.create_collection(self.collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        except Exception as e:
            print(f"Error clearing ChromaDB collection '{self.collection_name}': {e}")
            # If collection didn't exist, get_collection will create it later
            
# Example Usage (for testing ChromaDBManager directly)
if __name__ == "__main__":
    from llama_index.core.schema import Document
    from src.ingestion.chunking import DocumentChunker
    from src.ingestion.embeddings import EmbeddingModel
    from config.settings import settings
    import shutil

    # Clean up previous ChromaDB data for a fresh start
    if Path(settings.CHROMA_PERSIST_DIR).exists():
        print(f"Removing existing ChromaDB data at: {settings.CHROMA_PERSIST_DIR}")
        shutil.rmtree(settings.CHROMA_PERSIST_DIR)

    # Initialize embedding model first
    embed_model = EmbeddingModel().get_embedding_model()
    
    chroma_manager = ChromaDBManager(EmbeddingModel())
    vector_store = chroma_manager.initialize_vector_store()

    # Create dummy documents and nodes
    doc1 = Document(text="The quick brown fox jumps over the lazy dog.", metadata={"source": "animal_facts"})
    doc2 = Document(text="A car is a wheeled motor vehicle used for transportation.", metadata={"source": "vehicle_definitions"})
    doc3 = Document(text="The sun is the star at the center of the Solar System.", metadata={"source": "astronomy"})

    chunker = DocumentChunker(chunk_size=50, chunk_overlap=0)
    documents = [doc1, doc2, doc3]
    nodes = chunker.get_nodes_from_documents(documents)

    # Add nodes to ChromaDB
    chroma_manager.add_nodes(nodes)

    # Verify count
    print(f"Total nodes in ChromaDB after adding: {chroma_manager.chroma_collection.count()}")

    # Test retrieval (requires an embedding model)
    try:
        query_embedding = embed_model.get_query_embedding("What type of vehicle?")
        # Manually query ChromaDB for demonstration
        results = chroma_manager.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=['documents', 'metadatas']
        )
        print(f"\nChromaDB query results for 'What type of vehicle?':")
        if results and results['documents']:
            print(f"Retrieved Document: {results['documents'][0]}")
            print(f"Metadata: {results['metadatas'][0]}")
        else:
            print("No results found.")
    except Exception as e:
        print(f"Error during retrieval test: {e}")

    # Test clearing
    print("\nClearing ChromaDB collection...")
    chroma_manager.clear_collection()
    print(f"Total nodes in ChromaDB after clearing: {chroma_manager.chroma_collection.count()}")