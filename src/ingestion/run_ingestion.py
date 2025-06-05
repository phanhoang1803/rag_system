import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import sys
import os
from pathlib import Path

# Add the project root to the Python path to allow imports from src/
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.ingestion.loaders import DocumentLoader
from src.ingestion.chunking import DocumentChunker
from src.ingestion.embeddings import EmbeddingModelManager
from src.ingestion.vector_store import ChromaDBManager
from src.ingestion.indexer import Indexer
from src.ingestion.mongodb_manager import MongoDBManager


def run_data_ingestion(clear_existing_index: bool = True):
    """
    Orchestrates the entire data ingestion and indexing pipeline.
    This function replaces the Prefect flow logic.
    """
    print("Starting Enterprise RAG Data Ingestion Process (without Prefect)...")

    # Initialize managers
    loader = DocumentLoader()
    chunker = DocumentChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    embed_model_manager = EmbeddingModelManager(model_name=settings.EMBEDDING_MODEL_NAME, api_key=settings.GOOGLE_API_KEY)
    vector_db_manager = ChromaDBManager(persist_directory=settings.CHROMA_PERSIST_DIR, 
                                        collection_name=settings.CHROMA_COLLECTION_NAME)
    
    # --- Ingest Data to ChromaDB ---
    # Prepare documents for indexing (from file system)
    all_documents_for_indexing = []
    print(f"Loading documents from {settings.RAW_DOCS_DIR}")
    all_documents_for_indexing.extend(loader.load_markdown_docs(settings.RAW_DOCS_DIR))
    all_documents_for_indexing.extend(loader.load_text_docs(settings.RAW_DOCS_DIR))
    all_documents_for_indexing.extend(loader.load_pdf_docs(settings.RAW_DOCS_DIR))
    print(f"Loading documents from {settings.RAW_WEB_CONTENT_DIR}")
    all_documents_for_indexing.extend(loader.load_html_docs(settings.RAW_WEB_CONTENT_DIR))
    print(f"Loading documents from {settings.RAW_STRUCTURED_DIR}")
    all_documents_for_indexing.extend(loader.load_json_docs(settings.RAW_STRUCTURED_DIR))
    all_documents_for_indexing.extend(loader.load_csv_docs(settings.RAW_STRUCTURED_DIR))
    print(f"Total documents loaded for vector indexing: {len(all_documents_for_indexing)}")

    if all_documents_for_indexing:
        # Initialize the Indexer and run the indexing process
        indexer = Indexer(chunker, embed_model_manager, vector_db_manager)
        indexer.index_documents(all_documents_for_indexing, clear_existing=clear_existing_index)
    else:
        print("No documents found for vector indexing. Skipping ChromaDB update.")

    print("Enterprise RAG Data Ingestion Process Complete.")

if __name__ == "__main__":
    # Call the main ingestion function
    # Set clear_existing_index=True to re-index everything each time (useful during development)
    # Set clear_existing_index=False for incremental updates (more complex logic needed in indexer for that)
    run_data_ingestion(clear_existing_index=True)