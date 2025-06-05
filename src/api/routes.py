# intelligent_rag_system/src/api/routes.py

from fastapi import APIRouter, Depends, HTTPException, status
from src.api.models import QueryRequest, QueryResponse, ChatMessage, SourceNode
from src.api.dependencies import get_rag_pipeline
from src.rag_system.rag_pipeline import RAGPipeline
from typing import List, Dict, Any

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline) # Inject RAGPipeline instance
) -> QueryResponse:
    """
    Processes a user query through the RAG pipeline.

    Args:
        request: The QueryRequest object containing the user's query and chat history.
        rag_pipeline: The singleton instance of RAGPipeline.

    Returns:
        A QueryResponse object containing the synthesized answer, source nodes, and other metadata.
    """
    try:
        # Convert Pydantic ChatMessage objects to raw dicts for conversation_manager
        raw_chat_history = []
        if request.chat_history:
            for msg in request.chat_history:
                raw_chat_history.append({"role": msg.role, "content": msg.content})
        
        # Manually set chat history for the current request
        # In a real multi-user app, you'd manage history per user session,
        # but for this demo, we'll reset or set it for the current pipeline instance.
        # For simplicity, let's assume the pipeline's conversation_manager handles state
        # if multiple requests from the same "session" hit the same instance.
        # For robust multi-user, you'd pass and retrieve state from external store/db.
        
        # Clear existing history first to ensure the passed history is the only one considered for this turn
        rag_pipeline.conversation_manager.clear_history()
        for msg in raw_chat_history:
            rag_pipeline.conversation_manager.add_message(msg['role'], msg['content'])


        # Run the RAG pipeline
        pipeline_response = rag_pipeline.run_pipeline(request.query)

        # Convert pipeline's raw dict outputs back to Pydantic models for the API response
        return QueryResponse(
            answer=pipeline_response["answer"],
            source_nodes=[SourceNode(**node) for node in pipeline_response["source_nodes"]],
            rewritten_query=pipeline_response["rewritten_query"],
            extracted_entities=pipeline_response["extracted_entities"],
            chat_history=[ChatMessage(**msg) for msg in rag_pipeline.conversation_manager.get_chat_history_raw()]
        )
    except Exception as e:
        print(f"Error processing query: {e}")
        # Log the full traceback in a real application
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your query: {str(e)}"
        )

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "ok", "message": "RAG API is up and running!"}

