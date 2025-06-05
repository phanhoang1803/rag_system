# intelligent_rag_system/prefect_flows/ingestion_flow.py

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prefect import flow, task
from typing import List, Dict, Any

# Import your custom modules
from src.ingestion.loaders import DocumentLoader
from src.ingestion.chunking import DocumentChunker
from src.ingestion.embeddings import EmbeddingModel
from src.ingestion.vector_store import ChromaDBManager
from src.ingestion.indexer import Indexer
from src.ingestion.mongodb_manager import MongoDBManager
from config.settings import settings

# --- Initialize Managers as Prefect Task-like Objects (Singleton pattern for managers) ---
# This ensures that each manager instance is created once per flow run
@task
def get_document_loader() -> DocumentLoader:
    return DocumentLoader()

@task
def get_document_chunker() -> DocumentChunker:
    return DocumentChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

@task
def get_embedding_model_manager() -> EmbeddingModel:
    return EmbeddingModel(model_name=settings.EMBEDDING_MODEL_NAME, api_key=settings.GOOGLE_API_KEY)

@task
def get_chroma_db_manager() -> ChromaDBManager:
    return ChromaDBManager(embedding_model_manager=EmbeddingModel(), 
                           persist_directory=settings.CHROMA_PERSIST_DIR, 
                           collection_name=settings.CHROMA_COLLECTION_NAME)

@task
def get_mongodb_manager() -> MongoDBManager:
    manager = MongoDBManager(uri=settings.MONGODB_URI, db_name=settings.MONGODB_DB_NAME)
    manager.connect() # Ensure connection is established
    return manager


@task
def load_all_documents_for_indexing(loader: DocumentLoader) -> List[Any]:
    """Loads all relevant documents for indexing from file system sources."""
    all_documents = []
    print(f"Loading documents from {settings.RAW_DOCS_DIR}")
    all_documents.extend(loader.load_markdown_docs(settings.RAW_DOCS_DIR))
    all_documents.extend(loader.load_text_docs(settings.RAW_DOCS_DIR))
    all_documents.extend(loader.load_pdf_docs(settings.RAW_DOCS_DIR))
    print(f"Loading documents from {settings.RAW_WEB_CONTENT_DIR}")
    all_documents.extend(loader.load_html_docs(settings.RAW_WEB_CONTENT_DIR))
    all_documents.extend(loader.load_csv_docs(settings.RAW_STRUCTURED_DIR))
    all_documents.extend(loader.load_json_docs(settings.RAW_STRUCTURED_DIR))
    print(f"Total documents loaded for indexing: {len(all_documents)}")
    return all_documents

@flow(name="Enterprise RAG Data Ingestion")
def enterprise_rag_ingestion_flow(clear_existing_index: bool = True):
    """
    Main Prefect flow for ingesting data into the Enterprise RAG System.
    Orchestrates loading, chunking, embedding, and indexing.
    """
    print("Starting Enterprise RAG Data Ingestion Flow...")

    # Get instances of managers
    loader = get_document_loader()
    chunker = get_document_chunker()
    embed_model_manager = get_embedding_model_manager()
    vector_db_manager = get_chroma_db_manager()

    # Initialize Indexer with all its dependencies
    indexer = Indexer(chunker, embed_model_manager, vector_db_manager)

    # Task 2: Load documents for vector indexing
    documents_to_index = load_all_documents_for_indexing(loader=loader)

    # Task 3: Index documents into ChromaDB
    if documents_to_index:
        indexer.index_documents(documents_to_index, clear_existing=clear_existing_index)
    else:
        print("No documents found for vector indexing. Skipping ChromaDB update.")

    print("Enterprise RAG Data Ingestion Flow Finished.")

# --- Prefect Deployment (for scheduling) ---
# This section defines how Prefect should deploy and run your flow on a schedule.
# To register this deployment:
# 1. Start a Prefect agent: `prefect agent start --pool default` (or your custom pool)
# 2. Run: `python prefect_flows/ingestion_flow.py` (this script will register the deployment)
# You can then see and manage it in the Prefect Cloud UI.

if __name__ == "__main__":
    # For local testing without a Prefect server/agent
    # enterprise_rag_ingestion_flow(clear_existing_index=True)

    # To deploy this flow to Prefect Cloud (or a local server)
    # Ensure your PREFECT_API_KEY and PREFECT_API_URL are set in .env
    # and you're logged in `prefect cloud login` or `prefect profile use <your_profile>`

    # Command: 
    # prefect deploy -n "daily-enterprise-rag-ingestion" -q default --cron "0 7 * * *" --storage github --path prefect_flows/ --repo https://github.com/phanhoang1803/rag_system prefect_flows/ingestion_flow.py:enterprise_rag_ingestion_flow

    enterprise_rag_ingestion_flow.deploy(
        name="daily-enterprise-rag-ingestion",
        version="1.0",
        work_pool_name="default",
        cron="0 7 * * *",  # Every day at 7 AM
        tags=["data_ingestion", "rag"],
    )