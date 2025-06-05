# intelligent_rag_system/src/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as api_router # Alias to avoid naming conflict
from config.settings import settings

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.1.0",
        description="Enterprise RAG System API for knowledge base querying.",
    )

    # Configure CORS to allow requests from your Streamlit frontend (or other origins)
    # In a production environment, restrict origins to only your frontend's URL
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins for development. CHANGE THIS IN PRODUCTION!
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
        allow_headers=["*"],  # Allows all headers
    )

    # Include API router
    app.include_router(api_router, prefix="/api")

    # Add a root endpoint for basic access
    @app.get("/", tags=["Root"])
    async def read_root():
        return {"message": "Welcome to the Enterprise RAG API! Access /api/health for health check or /docs for API documentation."}

    return app

app = create_app()

# To run this application:
# Make sure you are in the intelligent_rag_system directory.
# Run: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
# The --reload option is useful for development as it restarts the server on code changes.
