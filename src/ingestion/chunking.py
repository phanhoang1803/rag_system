# intelligent_rag_system/src/ingestion/chunking.py
from typing import List
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter
from config.settings import settings

class DocumentChunker:
    """
    Handles the splitting of large documents into smaller, semantically meaningful chunks (nodes).
    Adheres to SRP by focusing only on chunking logic.
    """
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def get_nodes_from_documents(self, documents: List[Document]) -> List[TextNode]:
        """
        Splits a list of LlamaIndex Document objects into TextNode chunks. (Nodes reperesent "chunks")
        """
        print(f"Chunking {len(documents)} documents into nodes (chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})...")
        nodes = self.parser.get_nodes_from_documents(documents)
        print(f"Created {len(nodes)} text nodes.")
        return nodes

# Example Usage (for testing chunker directly)
if __name__ == "__main__":
    # Create some dummy documents for testing
    doc1 = Document(text="This is the first sentence of a long document. It discusses company policy. The policy is very detailed regarding remote work guidelines. Employees should refer to section 8 for international travel rules.", metadata={"source": "policy_doc"})
    doc2 = Document(text="Product A offers great features. It has a real-time collaboration tool and integrates seamlessly with other services.", metadata={"source": "product_manual"})

    documents = [doc1, doc2]

    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10) # Smaller chunks for demonstration
    nodes = chunker.get_nodes_from_documents(documents)

    for i, node in enumerate(nodes):
        print(f"\n--- Node {i+1} ---")
        print(f"Content: {node.text}")
        print(f"Metadata: {node.metadata}")
        print(f"Node ID: {node.id_}")