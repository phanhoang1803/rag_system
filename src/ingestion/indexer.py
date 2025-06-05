# intelligent_rag_system/src/ingestion/indexer.py
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pathlib import Path
from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import Settings

from src.ingestion.chunking import DocumentChunker
from src.ingestion.embeddings import EmbeddingModelManager
from src.ingestion.vector_store import ChromaDBManager
from config.settings import settings

class Indexer:
    """
    Orchestrates the indexing process for unstructured and semi-structured text data.
    Adheres to SRP by focusing on the indexing flow.
    Adheres to DIP by depending on abstract interfaces (BaseEmbedding, VectorStore).
    """
    def __init__(self,
                 chunker: DocumentChunker,
                 embedding_model_manager: EmbeddingModelManager,
                 vector_db_manager: ChromaDBManager):
        self.chunker = chunker
        self.embedding_model_manager = embedding_model_manager
        self.vector_db_manager = vector_db_manager
        self._index = None

    def get_index(self) -> VectorStoreIndex:
        """Returns the initialized LlamaIndex VectorStoreIndex."""
        if self._index is None:
            embed_model = self.embedding_model_manager.get_embedding_model()
            storage_context = self.vector_db_manager.get_storage_context()
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=storage_context.vector_store,
                embed_model=embed_model,
                # Optionally, you can specify service context here if you have custom LLM config
                # service_context=ServiceContext.from_defaults(llm=your_llm_instance)
            )
            print(f"LlamaIndex VectorStoreIndex initialized for collection '{self.vector_db_manager.collection_name}'.")
        return self._index

    def index_documents(self, documents: List[Document], clear_existing: bool = False):
        """
        Processes and indexes a list of LlamaIndex Document objects.
        """
        if not documents:
            print("No documents provided for indexing.")
            return

        if clear_existing:
            self.vector_db_manager.clear_collection()
            # Re-initialize the index after clearing
            self._index = None
            self.get_index()

        print(f"Starting indexing of {len(documents)} documents...")
        nodes = self.chunker.get_nodes_from_documents(documents)
        nodes = self.embedding_model_manager.generate_embeddings(nodes)
        
        self.vector_db_manager.add_nodes(nodes)
        print("Indexing complete.")

# Example Usage (for testing Indexer directly)
if __name__ == "__main__":
    from src.ingestion.loaders import DocumentLoader
    import shutil
    import os

    # Clean up previous ChromaDB data for a fresh start for this test
    if Path(settings.CHROMA_PERSIST_DIR).exists():
        print(f"Removing existing ChromaDB data at: {settings.CHROMA_PERSIST_DIR}")
        shutil.rmtree(settings.CHROMA_PERSIST_DIR)

    # Initialize components
    loader = DocumentLoader()
    chunker = DocumentChunker()
    embed_model_manager = EmbeddingModelManager()
    vector_db_manager = ChromaDBManager()

    indexer = Indexer(chunker, embed_model_manager, vector_db_manager)

    # Load some sample documents
    # Ensure you have 'company_policy.md' and 'product_manual.txt' in data/raw/docs/
    # Or create dummy files for this test if you haven't seeded data yet
    dummy_docs_path = settings.RAW_DOCS_DIR
    if not dummy_docs_path.exists():
        print(f"Creating dummy docs directory for test: {dummy_docs_path}")
        os.makedirs(dummy_docs_path)
        with open(dummy_docs_path / "company_policy.md", "w") as f:
            f.write("This is a test policy about remote work. Remote work is awesome. It has rules.")
        with open(dummy_docs_path / "product_manual.txt", "w") as f:
            f.write("Product Alpha is the best. It has features X, Y, and Z. Easy to use.")

    documents_to_index = []
    documents_to_index.extend(loader.load_markdown_docs(dummy_docs_path))
    documents_to_index.extend(loader.load_text_docs(dummy_docs_path))

    if not documents_to_index:
        print("No documents found or created for indexing test. Please ensure dummy files exist.")
    else:
        Settings.embed_model = embed_model_manager.get_embedding_model()
        Settings.llm = None
        # Index the documents, clearing existing data
        indexer.index_documents(documents_to_index, clear_existing=True)

        # Get the query engine for basic verification
        query_engine = indexer.get_index().as_query_engine()

        print("\nTesting query engine...")
        try:
            response = query_engine.query("What are the rules for remote work?")
            print(f"Query Response: {response}")
            # You might want to inspect response.source_nodes to see retrieved chunks
        except Exception as e:
            print(f"Error during query test: {e}")