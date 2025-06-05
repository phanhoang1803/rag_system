import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    BASE_DIR: Path = BASE_DIR
    
    # --- General Settings ---
    PROJECT_NAME: str = "Intelligent RAG System"
    PROJECT_DESCRIPTION: str = "An intelligent RAG system for enterprise knowledge base"
    PROJECT_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # --- Google Gemini API Settings ---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    
    # --- Perfect API Settings ---
    # PERFECT_API_URL: str = os.getenv("PERFECT_API_URL", "https://api.perfect.cloud/api")
    # PERFECT_API_KEY: str = os.getenv("PERFECT_API_KEY")
    
    # MongoDB Settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "enterprise_data")
    MONGODB_PRODUCTS_COLLECTION: str = os.getenv("MONGODB_PRODUCTS_COLLECTION", "products")
    MONGODB_EMPLOYEES_COLLECTION: str = os.getenv("MONGODB_EMPLOYEES_COLLECTION", "employees")
    MONGODB_FAQS_COLLECTION: str = os.getenv("MONGODB_FAQS_COLLECTION", "faqs")

    # --- Vector Store Settings ---
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_data")
    CHROMA_COLLECTION_NAME: str = "enterprise_documents"
    
    # --- LLM Settings ---
    LLM_MODEL_NAME: str = "gemini-2.0-flash-lite"
    EMBEDDING_MODEL_NAME: str = "models/embedding-001" # Default for GoogleGenerativeAI embeddings
    
    # --- RAG Settings ---
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 20
    TOP_K_RETRIEVAL: int = 5 # Number of documents to retrieve intially
    
    # --- Paths for raw data --
    RAW_DOCS_DIR: Path =  BASE_DIR / "data" / "raw" / "docs"
    RAW_STRUCTURED_DIR: Path = BASE_DIR / "data" / "raw" / "structured"
    RAW_WEB_CONTENT_DIR: Path = BASE_DIR / "data" / "raw" / "web_content"
    RAW_DATABASE_DIR: Path = BASE_DIR / "data" / "raw" / "database" # For the seed script
    
settings = Settings()


os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
