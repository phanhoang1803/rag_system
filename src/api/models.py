# intelligent_rag_system/src/api/models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    """Represents a single message in the chat history."""
    role: str = Field(..., description="Role of the sender (e.g., 'user', 'assistant').")
    content: str = Field(..., description="Content of the message.")

class QueryRequest(BaseModel):
    """Represents the request body for a user query."""
    query: str = Field(..., description="The user's input query.")
    chat_history: Optional[List[ChatMessage]] = Field(
        None, description="Optional chat history for multi-turn conversation."
    )

class SourceNode(BaseModel):
    """Represents a retrieved source document node."""
    text: str = Field(..., description="The text content of the retrieved node.")
    metadata: Dict[str, Any] = Field(..., description="Metadata associated with the node.")
    score: Optional[float] = Field(None, description="Relevance score of the node.")

class QueryResponse(BaseModel):
    """Represents the response body for a query."""
    answer: str = Field(..., description="The synthesized answer from the RAG pipeline.")
    source_nodes: List[SourceNode] = Field(..., description="List of retrieved source nodes.")
    rewritten_query: str = Field(..., description="The query after rewriting by the pipeline.")
    extracted_entities: List[Dict[str, str]] = Field(..., description="List of entities extracted from the query.")
    chat_history: List[ChatMessage] = Field(..., description="Updated chat history.")
    # Add other metadata here if needed, e.g., latency metrics
