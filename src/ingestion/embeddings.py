# intelligent_rag_system/src/ingestion/embeddings.py
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.embeddings import BaseEmbedding
from config.settings import settings
from typing import List
from llama_index.core.schema import TextNode

class EmbeddingModelManager:
    """
    Provides an interface for the embedding model.
    Adheres to SRP by focusing solely on embedding generation.
    Adheres to DIP by returning a BaseEmbedding abstract type.
    """
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME, api_key: str = settings.GOOGLE_API_KEY):
        self.model_name = model_name
        self.api_key = api_key
        self._embedding_model = None
        
    def get_embedding_model(self) -> BaseEmbedding:
        """Initializes and returns the GoogleGenerativeAIEmbedding model."""
        if self._embedding_model is None:
            print(f"Initializing embedding model: {self.model_name}")
            self._embedding_model = GoogleGenAIEmbedding(
                model_name=self.model_name,
                api_key=self.api_key
            )
        return self._embedding_model
    
    def generate_embeddings(self, text_nodes: List[TextNode]) -> List[TextNode]:
        """Generate embeddings for a list of TextNode objects."""
        embedding_model = self.get_embedding_model()
        embeddings = embedding_model.get_text_embedding_batch(
            [node.text for node in text_nodes]
        )
        for node, embedding in zip(text_nodes, embeddings):
            node.embedding = embedding
        return text_nodes
    
# Example Usage (for testing embedding model directly)
if __name__ == "__main__":
    embed_model_manager = EmbeddingModelManager()
    embed_model = embed_model_manager.get_embedding_model()

    text_to_embed = "This is a test sentence for embedding."
    print(f"Embedding text: '{text_to_embed}'...")
    try:
        embedding = embed_model.get_text_embedding(text_to_embed)
        print(f"Embedding generated. Length: {len(embedding)}")
        print(f"First 10 dimensions: {embedding[:10]}...")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        print("Please ensure your GOOGLE_API_KEY is correctly set in .env and you have access to the embedding model.")
