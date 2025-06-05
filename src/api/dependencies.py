# intelligent_rag_system/src/api/dependencies.py

from functools import lru_cache
from src.rag_system.rag_pipeline import RAGPipeline

@lru_cache() # Cache the instance for efficient reuse
def get_rag_pipeline() -> RAGPipeline:
    """
    Dependency injector for the RAGPipeline.
    Initializes the RAGPipeline once and reuses the instance across requests.
    """
    print("Initializing RAGPipeline for API...")
    return RAGPipeline()
